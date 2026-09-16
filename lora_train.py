#!/usr/bin/env python3
"""Axis C: train one LoRA expert per factor, so the weight change can be read.

Uniform LoRA on *every* block module (attn1, attn2, ff, norm1/2/3 across all 28
blocks) -- deliberately not the placement the analysis suggests, because the
point is to measure where training moves weights when nothing is pre-selected.
`weight_delta_analysis.py` then reads that out.

Objective, derived from the pipeline's own scalings (see
pipeline_cosmos2_video2world.py: c_in = c_skip = 1 - t, c_out = -t, t = s/(s+1)):

    z        = x0 + s * n                          s from the inference sigma schedule
    x0_pred  = (1-t) z - t F(z*(1-t), t, text)
    => the network's target is  F = n - x0          (flow-matching velocity)

    loss = || F(z*(1-t), t, text) - (n - x0) ||^2

The first latent frame is handed over unchanged at t_conditioning, exactly as the
pipeline does, so the task is "given frame 0 and the caption, predict the rest".

Data note: agd_dataset clips are nuScenes keyframes at 2 fps. Rather than passing
them off as 16 fps video, `--fps 2` is handed to the model, which feeds fps into
the temporal RoPE -- the clip and its conditioning then agree. The startup
diagnostic prints the initial loss at both fps so the choice is measured, not
assumed.

Experts trained (one caption per sample -- never all eight against the same
video, which would teach the model to ignore the factors):

    A     captions["A"]      base + appearance phrase
    G     captions["G"]      base + geometry phrase
    D     captions["D"]      base + dynamics phrase (can_bus GT wording)
    base  captions["base"]   control: what moves under fine-tuning alone

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=1 $PY lora_train.py --factor A --seed 0 --out ./runs/lora
  CUDA_VISIBLE_DEVICES=1 $PY lora_train.py --factor A --seed 0 --steps 8 --smoke
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cosmos_common import build_pipe, factor_phrases, latent_shape

TARGETS = (
    r"transformer_blocks\.\d+\.(attn1|attn2)\.(to_q|to_k|to_v|to_out\.0)"
    r"|transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)"
    r"|transformer_blocks\.\d+\.norm[123]\.linear_[12]"
)
FACTORS = ("A", "G", "D", "base")

MODULE_PATTERNS = {
    "attn1": r"attn1\.(to_q|to_k|to_v|to_out\.0)",
    "attn2": r"attn2\.(to_q|to_k|to_v|to_out\.0)",
    "ff": r"ff\.net\.(0\.proj|2)",
    "adaln": r"norm[123]\.linear_[12]",
}


def parse_layers(spec, n_layers):
    """'4-7' | '4,5,6,7' | 'all' -> sorted block indices."""
    if spec in (None, "", "all"):
        return list(range(n_layers))
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    assert out and max(out) < n_layers, f"{spec}: outside 0..{n_layers - 1}"
    return sorted(out)


def build_targets(layers, modules, n_layers):
    """peft matches a str `target_modules` with re.fullmatch on the module path, so
    restricting the block index to an alternation restricts LoRA to those blocks.
    `(0|1|2|3)` cannot match `10` because the pattern requires a `.` right after."""
    lay = r"\d+" if len(layers) == n_layers else "(" + "|".join(map(str, layers)) + ")"
    pats = [rf"transformer_blocks\.{lay}\.{MODULE_PATTERNS[m]}" for m in modules]
    return "|".join(pats)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True,
                    help="A/G/D/base, or any tag when --template is given")
    ap.add_argument("--template", default=None,
                    help="fixed caption for every sample, e.g. 'The red car drives "
                         "along the road.' Same videos, one word swapped -> dW_red - "
                         "dW_green is attributable to that word alone.")
    ap.add_argument("--out", default="./runs/lora")
    ap.add_argument("--agd", default=None, help="agd_dataset root (default $AGD)")
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=252)          # 63 samples x 4
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=13,          # (13-1)/4+1 = 4 latent
                    help="must satisfy (frames-1) %% 4 == 0")
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--fps", type=int, default=16,
                    help="clips are 2 Hz keyframes, but the model's prior fits the "
                         "16 fps labelling 3x better (measured: initial loss 1742 vs "
                         "5400) -- RoPE at fps=2 is far outside the trained range")
    ap.add_argument("--sched_steps", type=int, default=35)
    ap.add_argument("--sigma_min", type=float, default=0.0,
                    help="restrict training to sigma >= this (eDiff-I style band)")
    ap.add_argument("--sigma_max", type=float, default=float("inf"),
                    help="restrict training to sigma <= this")
    ap.add_argument("--sigma_dist", default="lognormal", choices=["lognormal", "schedule"],
                    help="lognormal: EDM training distribution ln(s)~N(-1.2,1.2) for "
                         "sigma_data=1. 'schedule' samples the 35 Karras inference "
                         "sigmas uniformly, which piles up at low sigma where n is "
                         "unpredictable and the loss is irreducible.")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--layers", default="all",
                    help="restrict LoRA to these blocks: '4-7', '4,5,6,7' or 'all'. "
                         "B-LoRA (ECCV 2024) found that confining LoRA to two of "
                         "SDXL's eleven blocks makes style and content separate on "
                         "their own; this is the knob for that experiment.")
    ap.add_argument("--modules", default="attn1,attn2,ff",
                    help="comma list from " + ",".join(MODULE_PATTERNS) +
                         ". attn1 takes no text, so keeping it in gives a built-in "
                         "null: factor structure appearing there is an artifact.")
    ap.add_argument("--tag", default=None, help="filename tag (default factor_sSEED)")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
def clean_samples(root: Path, text_field: str):
    """Samples whose A/G/D phrases are consistent with their combination labels."""
    out = []
    for cj in sorted(root.glob("*/caption.json")):
        meta = json.loads(cj.read_text())
        ph = factor_phrases(meta, text_field)
        agd = meta["captions"]["AGD"][text_field]
        ok = all(ph[k] and ph[k] in agd for k in ("A", "G", "D"))
        for c in ("AGD", "GD", "AD", "AG"):
            t = meta["captions"][c][text_field]
            for f in ("A", "G", "D"):
                if (ph[f] in t) != bool(meta["captions"][c][f]):
                    ok = False
        if ok:
            out.append(cj.parent)
    return out


def read_video(path: Path, n_frames, height, width):
    """agd_dataset mp4 -> [1, 3, T, H, W] in [-1, 1], centre-cropped to H."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    assert len(frames) >= n_frames, f"{path}: {len(frames)} < {n_frames} frames"
    v = np.stack(frames[:n_frames])                       # [T, H0, W0, 3]
    t, h0, w0, _ = v.shape
    if h0 != height or w0 != width:                       # 720 -> 704: centre crop
        top = max((h0 - height) // 2, 0)
        left = max((w0 - width) // 2, 0)
        v = v[:, top:top + height, left:left + width]
    x = torch.from_numpy(v).permute(3, 0, 1, 2).float().unsqueeze(0)   # [1,3,T,H,W]
    return x / 127.5 - 1.0


def encode_video(pipe, video, device):
    """VAE encode + the pipeline's latent normalisation."""
    lat = pipe.vae.encode(video.to(device, dtype=pipe.vae.dtype)).latent_dist.mode()
    cfg = pipe.vae.config
    mean = torch.tensor(cfg.latents_mean).view(1, cfg.z_dim, 1, 1, 1).to(device, lat.dtype)
    std = torch.tensor(cfg.latents_std).view(1, cfg.z_dim, 1, 1, 1).to(device, lat.dtype)
    return ((lat - mean) / std * pipe.scheduler.config.sigma_data).float()


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    assert (args.frames - 1) % 4 == 0, "(frames-1) must be divisible by 4"
    import os
    root = Path(args.agd or os.environ["AGD"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.template is None and args.factor not in FACTORS:
        raise SystemExit(f"--factor must be one of {FACTORS} unless --template is given")
    samples = clean_samples(root, args.text_field)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"[data] {len(samples)} clean samples   factor={args.factor}")

    pipe = build_pipe()
    # Building the pipeline (the guardrail's model init) leaves autograd globally
    # disabled, so every forward would silently produce a graph-less tensor and
    # backward would fail with "does not require grad". Turn it back on.
    torch.set_grad_enabled(True)
    tfm = pipe.transformer
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    device = torch.device("cuda")
    band = "" if args.sigma_max == float("inf") and args.sigma_min == 0 else \
        f"   sigma band [{args.sigma_min}, {args.sigma_max}]"
    print(f"[model] latent {tuple(shape)}   fps={args.fps}{band}")

    # ---- precompute targets and text embeddings (kept in RAM, not on disk) --- #
    t0 = time.time()
    data = []
    with torch.no_grad():
        for i, s in enumerate(samples):
            meta = json.loads((s / "caption.json").read_text())
            caption = args.template or meta["captions"][args.factor][args.text_field]
            emb = pipe._get_t5_prompt_embeds(caption, device=device)
            video = read_video(s / "video.mp4", args.frames, args.height, args.width)
            x0 = encode_video(pipe, video, device)
            assert tuple(x0.shape) == tuple(shape), f"{s.name}: {tuple(x0.shape)}"
            data.append({"x0": x0.cpu(), "emb": emb.cpu().half(),
                         "id": s.name, "caption": caption})
            if i == 0:
                print(f"[caption:{args.factor}] {caption}")
            if (i + 1) % 20 == 0:
                print(f"  encoded {i+1}/{len(samples)}", flush=True)
    print(f"[data] encoded in {time.time()-t0:.0f}s")

    # the VAE and the text encoder are done with -- free the GPU for training
    pipe.vae.to("cpu")
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # ---- LoRA on the selected blocks and module types ---------------------- #
    from peft import LoraConfig
    modules = [m for m in args.modules.split(",") if m]
    assert all(m in MODULE_PATTERNS for m in modules), args.modules
    layers = parse_layers(args.layers, len(tfm.transformer_blocks))
    targets = build_targets(layers, modules, len(tfm.transformer_blocks))
    print(f"[scope] blocks {layers[0]}-{layers[-1]} ({len(layers)} of "
          f"{len(tfm.transformer_blocks)})   modules {modules}")
    cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.0,
                     bias="none", target_modules=targets, init_lora_weights=True)
    tfm.add_adapter(cfg)
    got = sorted({int(n.split("transformer_blocks.")[1].split(".")[0])
                  for n, p in tfm.named_parameters() if p.requires_grad})
    assert got == layers, f"LoRA landed on blocks {got}, expected {layers}"
    # Reentrant checkpointing drops the graph when no *input* requires grad, which
    # is exactly the LoRA case (only weights are trainable) -- use the
    # non-reentrant implementation.
    import functools
    from torch.utils.checkpoint import checkpoint
    tfm.enable_gradient_checkpointing(
        gradient_checkpointing_func=functools.partial(checkpoint, use_reentrant=False))
    train_params = [p for p in tfm.parameters() if p.requires_grad]
    n_lora = sum(p.numel() for p in train_params)
    n_mod = len({n.rsplit(".lora_", 1)[0] for n, p in tfm.named_parameters()
                 if p.requires_grad})
    print(f"[lora] rank {args.rank}  {n_mod} modules  {n_lora/1e6:.1f} M trainable")

    pipe.scheduler.set_timesteps(args.sched_steps, device=device)
    sigmas = pipe.scheduler.sigmas[:-1].float().to(device)     # inference schedule
    sigma_conditioning = torch.tensor(1e-4, device=device)
    t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

    cond_indicator = torch.zeros(1, 1, shape[2], 1, 1, device=device)
    cond_indicator[:, :, 0] = 1.0                              # first latent frame
    cond_mask = cond_indicator.expand(1, 1, shape[2], shape[3], shape[4]).contiguous()
    padding_mask = torch.zeros(1, 1, args.height, args.width, device=device)

    def batch_loss(item, sigma, fps):
        x0 = item["x0"].to(device).float()
        emb = item["emb"].to(device, dtype=torch.bfloat16)
        n = torch.randn_like(x0)
        z = x0 + sigma * n
        t = sigma / (sigma + 1)
        c_in = 1 - t
        inp = z * c_in
        inp = cond_indicator * x0 + (1 - cond_indicator) * inp
        ts = t.view(1, 1, 1, 1, 1).expand(1, -1, shape[2], -1, -1)
        ts = cond_indicator * t_conditioning + (1 - cond_indicator) * ts
        pred = tfm(hidden_states=inp.to(torch.bfloat16),
                   timestep=ts.to(torch.bfloat16),
                   encoder_hidden_states=emb,
                   fps=fps,
                   condition_mask=cond_mask.to(torch.bfloat16),
                   padding_mask=padding_mask.to(torch.bfloat16),
                   return_dict=False)[0].float()
        target = n - x0
        # the conditioning frame is given, not predicted -- exclude it
        m = (1 - cond_indicator).expand_as(x0)
        return ((pred - target) ** 2 * m).sum() / m.sum()

    # ---- diagnostic: is fps=2 or fps=16 closer to the model's prior? -------- #
    with torch.no_grad():
        for probe_fps in dict.fromkeys((args.fps, 2, 16)):
            ls = [batch_loss(data[i], sigmas[len(sigmas) // 2], probe_fps).item()
                  for i in range(min(6, len(data)))]
            print(f"[fps diag] fps={probe_fps:2d}  initial loss "
                  f"{np.mean(ls):.4f} +- {np.std(ls):.4f}")

    # ---- train ------------------------------------------------------------- #
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    steps = 8 if args.smoke else args.steps
    order = np.random.permutation(len(data))
    hist = []
    t0 = time.time()
    for step in range(steps):
        item = data[order[step % len(data)]]
        if step % len(data) == len(data) - 1:
            order = np.random.permutation(len(data))
        for _ in range(1000):                     # rejection-sample into the band
            if args.sigma_dist == "lognormal":
                sigma = torch.exp(torch.randn(1, device=device) * 1.2 - 1.2)[0]
            else:
                sigma = sigmas[torch.randint(len(sigmas), (1,), device=device)][0]
            if args.sigma_min <= float(sigma) <= args.sigma_max:
                break
        with torch.enable_grad():
            loss = batch_loss(item, sigma, args.fps)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"step": step, "loss": loss.item(), "sigma": sigma.item(),
                     "grad_norm": gn, "id": item["id"]})
        if step % 10 == 0 or step == steps - 1:
            recent = np.mean([h["loss"] for h in hist[-10:]])
            print(f"  step {step:4d}/{steps}  loss {loss.item():.4f}  "
                  f"(mean10 {recent:.4f})  sigma {sigma.item():7.3f}  "
                  f"|g| {gn:.3f}  {time.time()-t0:.0f}s", flush=True)

    # ---- save the adapter (ΔW = B@A) and the run record -------------------- #
    # fp32, not fp16: same-seed runs share order/sigma/noise so dW_f - dW_base is an
    # exact paired difference, but the caption is only ~0.15% of the loss, so the two
    # adapters differ by ~1e-3 relative -- fp16 (~5e-4) would bury that.
    tag = args.tag or f"{args.factor}_s{args.seed}"
    sd = {k: v.detach().cpu().float() for k, v in tfm.state_dict().items()
          if "lora_" in k}
    torch.save({"state_dict": sd,
                "config": {k: v for k, v in vars(args).items()},
                "n_samples": len(data), "history": hist,
                "targets": targets, "layers": layers,
                "modules": modules},
               out / f"lora_{tag}.pt")
    print(f"\n[save] {out}/lora_{tag}.pt  ({len(sd)} tensors, "
          f"{sum(v.numel() for v in sd.values())/1e6:.1f} M)  "
          f"{time.time()-t0:.0f}s")
    first10 = np.mean([h["loss"] for h in hist[:10]])
    last10 = np.mean([h["loss"] for h in hist[-10:]])
    print(f"[loss] first10 {first10:.4f} -> last10 {last10:.4f} "
          f"({(1-last10/first10)*100:+.1f}%)")


if __name__ == "__main__":
    main()
