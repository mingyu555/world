#!/usr/bin/env python3
"""Same-noise A/G/D factor-addition trace for Cosmos-Predict2-2B-Video2World.

One conditioning image + the 8 A/G/D caption combinations of one agd_dataset
sample are generated from the *same* initial latent noise, and the denoising
trajectory is logged per timestep (latents) and per transformer layer x
timestep (hidden-state summaries + JL sketch).  See plan sections 16-19:

    D_z(t)    = || z_t^complex - z_t^base ||
    D_h(l,t)  = || h_{l,t}^complex - h_{l,t}^base ||

Only the conditional (positive-prompt) forward pass of each CFG step is logged.

Usage
-----
  export HF_HOME=/mnt/ssd4/youngjae/cvpr2027_yj/hf_cache
  CUDA_VISIBLE_DEVICES=2 python3 run_same_noise.py \
      --sample /mnt/ssd4/youngjae/cvpr2027_yj/agd_dataset/scene-0014_f16 \
      --out    /mnt/ssd4/youngjae/cvpr2027_yj/cosmos_exp/scene-0014_f16 \
      --null_run
  # low memory / quick smoke test:
  #   --frames 13 --steps 8 --height 704 --width 1280 --vae_tiling
"""
import argparse, gc, json, os, sys, time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_utils import Sketcher, block_list, as_hidden

MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
CONDS = ["base", "A", "G", "D", "AG", "AD", "GD", "AGD"]
NEG = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky movements, low frame "
    "rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
    "visual noise, and flickering. Overall, the video is of poor quality."
)


# --------------------------------------------------------------------------- #
# tracing
# --------------------------------------------------------------------------- #
class Tracer:
    """Hooks every transformer block; records the conditional pass of each step."""

    def __init__(self, transformer, latent_shape, sketch_k=64, full_every=0, out_dir=None):
        self.name, self.blocks = block_list(transformer)
        self.n_layers = len(self.blocks)
        self.latent_T = latent_shape[2]
        self.sketcher = Sketcher(k=sketch_k)
        self.full_every = full_every
        self.out_dir = out_dir
        self.handles = []
        # step bookkeeping -- the pipeline calls the transformer twice per step
        # (cond then uncond); call 0 of each step is the conditional one.
        self.step = 0
        self.call_in_step = 0
        self.enabled = True
        self.rec = {"norm": [], "tok_mean": [], "frame_norm": [], "sketch": []}
        self._buf = None
        self.seq_len = None
        self.dim = None

    def attach(self):
        for i, blk in enumerate(self.blocks):
            self.handles.append(blk.register_forward_hook(self._make_hook(i)))
        # a pre-hook on block 0 counts forward passes
        self.handles.append(self.blocks[0].register_forward_pre_hook(self._pre_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _pre_hook(self, mod, args):
        if self.call_in_step == 0:
            self._buf = {k: [None] * self.n_layers for k in self.rec}

    def _make_hook(self, layer_idx):
        def hook(mod, args, out):
            if not self.enabled or self.call_in_step != 0:
                return
            h = as_hidden(out)
            if h.dim() == 3:            # [B, S, D]
                h = h[0]
            elif h.dim() == 4:          # [B, T, S, D] (unlikely)
                h = h[0].reshape(-1, h.shape[-1])
            h = h.detach()
            S, D = h.shape
            self.seq_len, self.dim = S, D
            hf = h.float()
            self._buf["norm"][layer_idx] = hf.norm().cpu()
            self._buf["tok_mean"][layer_idx] = hf.mean(0).cpu().half()
            T = self.latent_T
            if S % T == 0:
                per_frame = hf.view(T, S // T, D).norm(dim=(1, 2))
            else:                        # fall back: single bucket
                per_frame = hf.norm().view(1)
            self._buf["frame_norm"][layer_idx] = per_frame.cpu()
            self._buf["sketch"][layer_idx] = self.sketcher(hf)
            if self.full_every and self.step % self.full_every == 0 and self.out_dir:
                fp = Path(self.out_dir)
                fp.mkdir(parents=True, exist_ok=True)
                torch.save(h.cpu().half(), fp / f"l{layer_idx:02d}_t{self.step:03d}.pt")
        return hook

    def end_of_call(self):
        """Called after each transformer forward (from the wrapper)."""
        if self.call_in_step == 0 and self._buf is not None:
            for k in self.rec:
                self.rec[k].append(torch.stack(self._buf[k]))
            self._buf = None
        self.call_in_step += 1

    def end_of_step(self):
        self.step += 1
        self.call_in_step = 0

    def summary(self):
        out = {}
        for k, v in self.rec.items():
            if v:
                out[k] = torch.stack(v)      # [n_steps, n_layers, ...]
        return out


def wrap_transformer(transformer, tracer):
    """Count forward calls so cond/uncond passes can be told apart."""
    orig = transformer.forward

    def fwd(*a, **kw):
        out = orig(*a, **kw)
        tracer.end_of_call()
        return out

    transformer.forward = fwd
    return orig


# --------------------------------------------------------------------------- #
class _PassThroughSafety:
    """Stand-in for CosmosSafetyChecker.

    The pipeline hard-requires a safety checker (`safety_checker=None` raises),
    so bypassing it means supplying an object with the same three entry points.
    Worth doing here for two reasons beyond speed: the blocklist false-positives
    on benign driving text (it rejected "a stop line about 3 m ahead" because
    "3 m" matches the 3M trademark), and the real checker holds ~7 GB of VRAM
    that a shared card cannot spare.  Enabled only via `--no_guardrail`; note
    that NVIDIA's Open Model License asks for it to stay on.
    """

    def to(self, *a, **k):
        return self

    def check_text_safety(self, prompt):
        return True

    def check_video_safety(self, video):
        return video


def build_pipe(dtype, offload, vae_tiling, no_guardrail=False):
    from diffusers import Cosmos2VideoToWorldPipeline

    extra = {"safety_checker": _PassThroughSafety()} if no_guardrail else {}
    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(MODEL, torch_dtype=dtype, **extra)
    # diffusers >=0.35: `_execution_device` can raise AttributeError behind the
    # property; pin it so accelerate hooks do not shadow it.
    try:
        _ = pipe._execution_device
    except Exception:
        type(pipe)._execution_device = property(lambda self: torch.device("cuda:0"))
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception as e:
            print(f"[warn] vae tiling unavailable: {e}")
    return pipe


def latent_shape(pipe, batch, frames, height, width):
    """Match Cosmos2VideoToWorldPipeline.prepare_latents exactly."""
    C = pipe.transformer.config.in_channels - 1          # 16 (last channel = cond mask)
    ct = pipe.vae_scale_factor_temporal
    cs = pipe.vae_scale_factor_spatial
    T = (frames - 1) // ct + 1
    return (batch, C, T, height // cs, width // cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="agd_dataset/<sample_id> directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--frames", type=int, default=93)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sketch_k", type=int, default=96)
    ap.add_argument("--full_every", type=int, default=0,
                    help="also dump exact hidden states every N steps (huge)")
    ap.add_argument("--null_run", action="store_true",
                    help="re-run 'base' a second time -> numerical noise floor")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--vae_tiling", action="store_true")
    ap.add_argument("--no_guardrail", action="store_true",
                    help="bypass the Cosmos safety checker (frees ~7 GB and avoids "
                         "blocklist false positives on driving text)")
    ap.add_argument("--no_video", action="store_true", help="skip decoding/saving mp4")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    sample = Path(args.sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sample / "caption.json").read_text())
    image = Image.open(sample / "cond_image.jpg").convert("RGB")

    conds = [c for c in args.conds.split(",") if c]
    runs = [(c, meta["captions"][c][args.text_field]) for c in conds]
    if args.null_run:
        runs.append(("null", meta["captions"]["base"][args.text_field]))

    dtype = torch.bfloat16
    print(f"[load] {MODEL}")
    pipe = build_pipe(dtype, args.offload, args.vae_tiling, args.no_guardrail)

    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[latent] {shape}")

    # ---- shared initial noise (unit variance; prepare_latents scales by sigma_max)
    zpath = out / "z_T_shared.pt"
    if zpath.exists():
        z_shared = torch.load(zpath)
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z_shared = torch.randn(shape, generator=g, dtype=torch.float32)
        torch.save(z_shared, zpath)

    (out / "config.json").write_text(json.dumps({
        "model": MODEL, "sample": str(sample), "text_field": args.text_field,
        "frames": args.frames, "height": args.height, "width": args.width,
        "steps": args.steps, "guidance": args.guidance, "fps": args.fps,
        "seed": args.seed, "latent_shape": list(shape), "sketch_k": args.sketch_k,
        "conds": [r[0] for r in runs],
        "prompts": {c: t for c, t in runs},
    }, indent=1))

    for cond, prompt in runs:
        tp = out / f"{cond}_trace.pt"
        if args.resume and tp.exists():
            print(f"[skip] {cond}")
            continue
        print(f"\n=== {cond} ===\n{prompt}")
        t0 = time.time()

        tracer = Tracer(pipe.transformer, shape, args.sketch_k,
                        args.full_every, out / f"full_{cond}")
        tracer.attach()
        orig_fwd = wrap_transformer(pipe.transformer, tracer)

        zs = []

        def cb(p, i, t, kw):
            zs.append(kw["latents"].detach().float().cpu())
            tracer.end_of_step()
            return kw

        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            res = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=NEG,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                fps=args.fps,
                generator=gen,
                latents=z_shared.clone().to("cuda", dtype=dtype),
                output_type="pil" if not args.no_video else "latent",
                callback_on_step_end=cb,
                callback_on_step_end_tensor_inputs=["latents"],
            )

        tracer.detach()
        pipe.transformer.forward = orig_fwd

        rec = tracer.summary()
        rec["z"] = torch.stack(zs).half()          # [n_steps, B, C, T, H, W]
        rec["seq_len"] = tracer.seq_len
        rec["dim"] = tracer.dim
        rec["n_layers"] = tracer.n_layers
        rec["prompt"] = prompt
        rec["cond"] = cond
        torch.save(rec, tp)
        print(f"[save] {tp}  ({tp.stat().st_size/1e6:.0f} MB)  "
              f"layers={tracer.n_layers} steps={rec['norm'].shape[0]} "
              f"S={tracer.seq_len} D={tracer.dim}  {time.time()-t0:.0f}s")

        if not args.no_video:
            try:
                from diffusers.utils import export_to_video
                export_to_video(res.frames[0], str(out / f"{cond}.mp4"), fps=args.fps)
            except Exception as e:
                print(f"[warn] video export failed: {e}")

        del res, zs, rec
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[done]", out)


if __name__ == "__main__":
    main()
