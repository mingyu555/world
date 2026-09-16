#!/usr/bin/env python3
"""B-LoRA's block analysis, adapted: inject one factor's caption into one block
window and measure how the *generated video* changes.

B-LoRA (ECCV 2024, Sec. 3) located style and content by injecting a different
text prompt into the cross-attention layers of a single SDXL transformer block
while every other block kept the original prompt, generating the image, and
comparing CLIP image-text similarity against both prompts. Over 400 prompt pairs
they found blocks W2 and W4 changed the *content* and W5 changed the *color*.

The same question here, with three differences forced by the setting:

  28 blocks, not 11      -> seven windows of four, matching `blora_scan.sh`
  video, not an image    -> the readout has to separate spatial from temporal change
  A/G/D, not style/content -> the metrics are chosen to match the factors

Every other probe in this directory measured *how much* a latent moved. This one
measures *what kind* of change reached the pixels, which is what makes a result
readable as "window w controls appearance".

    reference   the base caption in all 28 blocks                  V0
    injected    the base caption everywhere except window w, which
                gets base + one factor phrase                      V
    ceiling     the factor caption in all 28 blocks

Metrics, all as V against V0 on decoded pixels:

    A  d_lum     |mean luminance difference|
       d_hist    total-variation distance of the per-channel colour histogram
    G  d_struct  1 - SSIM on z-scored luminance (brightness and contrast removed,
                 so this responds to structure rather than to appearance)
    D  d_flow    |change in mean Farneback optical-flow magnitude|
       d_flowdir mean angular change of the flow field
    *  d_siglip  sim(V, factor caption) - sim(V0, factor caption), SigLIP so400m
                 -- the B-LoRA statistic: did injecting here move the video
                 toward the injected caption?

The 3x3 structure is its own control: injecting A at window w should move the
appearance metrics more than injecting G or D does. Two hard controls come free:

    base-into-window     injecting the base caption must reproduce V0 exactly
    attn1 has no text    (checked by `block_patch`, not here)

Cosmos runs classifier-free guidance as a *second* transformer call with
`negative_prompt_embeds`, so the override is gated to the conditional pass -- an
ungated override would rewrite the negative prompt too.

    source env.sh
    CUDA_VISIBLE_DEVICES=0 $PY layer_prompt_image_probe.py --samples 4 --out ./runs/lpi
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from block_patch import BlockPatcher
from cosmos_common import NEG, build_pipe, latent_shape
from lora_train import clean_samples

WINDOWS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27"]
FACTORS = ("A", "G", "D")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/lpi")
    ap.add_argument("--agd", default=None)
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--sample_stride", type=int, default=7,
                    help="spread the chosen samples over the dataset")
    ap.add_argument("--only", default=None,
                    help="comma list of sample directory names, overriding "
                         "--samples/--offset/--stride. The ceiling screen shows "
                         "only ~17%% of samples have a dynamics effect that "
                         "reaches the pixels at all, so the window grid is worth "
                         "running only on those.")
    ap.add_argument("--sample_offset", type=int, default=0,
                    help="skip this many of the strided samples, so several GPUs "
                         "can each take a disjoint slice")
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--factors", default="A,G,D")
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ceiling", action="store_true",
                    help="skip the all-28-blocks ceiling generation")
    ap.add_argument("--siglip", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--no_siglip", action="store_true")
    ap.add_argument("--skip_control", action="store_true",
                    help="skip the base-into-all-blocks no-op check. Only for the "
                         "ceiling screen, where it has already been verified.")
    ap.add_argument("--tag", default="lpi")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
def lum(x):
    """[..., 3] in [0,1] -> luminance."""
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def hist3(x, bins=32):
    return np.stack([np.histogram(x[..., c], bins=bins, range=(0, 1))[0]
                     / x[..., c].size for c in range(3)])


def flow_fields(V):
    """Farneback flow between consecutive frames of [T,H,W,3] in [0,1]."""
    import cv2
    g = [(np.clip(lum(f), 0, 1) * 255).astype(np.uint8) for f in V]
    return [cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            for a, b in zip(g[:-1], g[1:])]


def video_metrics(V, V0, flow0=None):
    """V, V0: [T,H,W,3] float in [0,1]. Returns the factor-matched distances."""
    from skimage.metrics import structural_similarity as ssim

    l, l0 = lum(V), lum(V0)
    d_lum = float(abs(l.mean() - l0.mean()))
    d_hist = float(np.abs(hist3(V) - hist3(V0)).sum() / 2 / 3)

    def z(a):
        s = a.std()
        return (a - a.mean()) / (s if s > 1e-8 else 1.0)

    ss = []
    for t in range(len(V)):
        a_, b_ = z(l0[t]), z(l[t])
        rng = float(max(np.ptp(a_), np.ptp(b_)))
        ss.append(ssim(a_, b_, data_range=rng if rng > 1e-8 else 1.0))
    d_struct = float(1 - np.mean(ss))

    f1 = flow_fields(V)
    f0 = flow0 if flow0 is not None else flow_fields(V0)
    m1 = np.mean([np.linalg.norm(f, axis=-1).mean() for f in f1])
    m0 = np.mean([np.linalg.norm(f, axis=-1).mean() for f in f0])
    d_flow = float(abs(m1 - m0))
    ang = []
    for a, b in zip(f0, f1):
        na = np.linalg.norm(a, axis=-1)
        nb = np.linalg.norm(b, axis=-1)
        keep = (na > 0.5) & (nb > 0.5)          # ignore static pixels
        if keep.sum() < 100:
            continue
        c = (a[keep] * b[keep]).sum(-1) / (na[keep] * nb[keep])
        ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())
    d_flowdir = float(np.mean(ang)) if ang else 0.0
    return {"d_lum": d_lum, "d_hist": d_hist, "d_struct": d_struct,
            "d_flow": d_flow, "d_flowdir": d_flowdir}, f1


class SigLIP:
    """sim(video, caption): mean over frames of the cosine in SigLIP space."""

    def __init__(self, name, device):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.float32).to(device).eval()
        self.device = device

    @staticmethod
    def _feat(o):
        """get_*_features returns a bare tensor on some transformers versions and a
        ModelOutput on others."""
        if torch.is_tensor(o):
            return o
        if getattr(o, "pooler_output", None) is not None:
            return o.pooler_output
        return o.last_hidden_state.mean(1)

    @torch.no_grad()
    def text(self, caps):
        t = self.proc(text=caps, padding="max_length", truncation=True,
                      max_length=64, return_tensors="pt").to(self.device)
        e = self._feat(self.model.get_text_features(**t))
        return e / e.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def video(self, V, every=3):
        ims = [Image.fromarray((np.clip(V[t], 0, 1) * 255).astype(np.uint8))
               for t in range(0, len(V), every)]
        p = self.proc(images=ims, return_tensors="pt").to(self.device)
        e = self._feat(self.model.get_image_features(**p))
        e = e / e.norm(dim=-1, keepdim=True)
        return e.mean(0, keepdim=True)


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.agd or os.environ["AGD"])
    device = torch.device("cuda")

    samples = clean_samples(root, args.text_field)
    if args.only:
        want = [w for w in args.only.split(",") if w]
        by = {s.name: s for s in samples}
        missing = [w for w in want if w not in by]
        assert not missing, f"not clean samples: {missing}"
        picked = [by[w] for w in want]
    else:
        strided = samples[::args.sample_stride]
        picked = strided[args.sample_offset:args.sample_offset + args.samples]
        assert picked, f"offset {args.sample_offset} past {len(strided)} strided"
    print(f"[data] {len(samples)} clean samples, using {len(picked)}: "
          + " ".join(s.name for s in picked))

    windows = [w for w in args.windows.split(",") if w]
    factors = [f for f in args.factors.split(",") if f]
    sites = [(w, [int(x) for x in range(int(w.split('-')[0]), int(w.split('-')[1]) + 1)])
             for w in windows]
    if not args.no_ceiling:
        sites.append(("all", list(range(28))))
    print(f"[sites] {[s[0] for s in sites]}   factors {factors}")
    n_gen = len(picked) * (1 + 1 + len(sites) * len(factors))
    print(f"[plan] {n_gen} generations "
          f"({args.frames} frames, {args.steps} steps, cfg {args.guidance})")

    pipe = build_pipe()
    tfm = pipe.transformer
    assert len(tfm.transformer_blocks) == 28
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    patcher = BlockPatcher(tfm)
    patcher.attach()

    # Gate the override to the conditional pass: the pipeline calls the
    # transformer a second time with negative_prompt_embeds, and rewriting that
    # would change the guidance direction instead of the prompt.
    orig_fwd = tfm.forward
    gate = {"pos": None, "map": {}}

    def fwd(*a, **kw):
        # The conditional pass comes first in every step (pipeline lines 709/728),
        # so the first encoder_hidden_states of a generation is the positive
        # embedding. The pipeline re-encodes the prompt per call, hence the
        # per-generation re-capture rather than a one-off.
        e = kw.get("encoder_hidden_states")
        if gate["pos"] is None:
            gate["pos"] = e
        on = bool(gate["map"]) and e is gate["pos"]
        patcher.text = gate["map"] if on else {}
        patcher.enabled = on
        try:
            return orig_fwd(*a, **kw)
        finally:
            patcher.enabled = False
            patcher.text = {}

    tfm.forward = fwd

    sig = None if args.no_siglip else SigLIP(args.siglip, device)
    if sig is not None:
        print(f"[siglip] {args.siglip} loaded")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z0 = torch.randn(shape, generator=g, dtype=torch.float32)

    def generate(image, base_cap, override_cap=None, layers=None):
        """One generation. `override_cap` is fed to `layers` only; the pipeline's
        own prompt (base_cap) reaches every other block."""
        gate["map"] = {}
        gate["pos"] = None
        if override_cap is not None:
            with torch.no_grad():
                emb = pipe._get_t5_prompt_embeds(override_cap, device=device)
            gate["map"] = {l: emb for l in layers}
        with torch.no_grad():
            res = pipe(image=image, prompt=base_cap, negative_prompt=NEG,
                       height=args.height, width=args.width,
                       num_frames=args.frames, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, fps=args.fps,
                       generator=torch.Generator(device="cpu").manual_seed(args.seed),
                       latents=z0.clone().to(device, dtype=torch.bfloat16),
                       output_type="np")
        gate["map"] = {}
        gate["pos"] = None
        v = res.frames[0]
        return np.asarray(v, dtype=np.float32)

    rows = []
    t_all = time.time()
    for si, s in enumerate(picked):
        meta = json.loads((s / "caption.json").read_text())
        caps = {k: meta["captions"][k][args.text_field]
                for k in ("base",) + tuple(factors)}
        image = Image.open(s / "cond_image.jpg").convert("RGB")
        print(f"\n[{si+1}/{len(picked)}] {s.name}")
        print(f"  base: {caps['base']}")
        for f in factors:
            print(f"  {f}:    {caps[f]}")

        t0 = time.time()
        V0 = generate(image, caps["base"])
        flow0 = flow_fields(V0)
        print(f"  reference generated in {time.time()-t0:.0f}s  "
              f"shape {V0.shape}  mean lum {lum(V0).mean():.4f}")

        if not args.skip_control:
            Vc = generate(image, caps["base"], caps["base"], list(range(28)))
            null = float(np.abs(Vc - V0).max())
            print(f"  [control] base-into-all-blocks max|dV| = {null:.3e}")
            assert null == 0.0, f"override is not a no-op for the base caption: {null}"

        base_sim = {}
        if sig is not None:
            tv = sig.text([caps[f] for f in factors] + [caps["base"]])
            iv0 = sig.video(V0)
            s0 = (iv0 @ tv.T)[0].tolist()
            base_sim = dict(zip(list(factors) + ["base"], s0))
            print("  [siglip ref] " + "  ".join(f"{k}={v:+.4f}"
                                                for k, v in base_sim.items()))

        for name, layers in sites:
            for f in factors:
                t0 = time.time()
                V = generate(image, caps["base"], caps[f], layers)
                m, _ = video_metrics(V, V0, flow0)
                if sig is not None:
                    iv = sig.video(V)
                    s1 = (iv @ tv.T)[0].tolist()
                    cur = dict(zip(list(factors) + ["base"], s1))
                    m["d_siglip"] = cur[f] - base_sim[f]
                    m["d_siglip_base"] = cur["base"] - base_sim["base"]
                m.update({"sample": s.name, "site": name, "factor": f,
                          "n_layers": len(layers), "secs": time.time() - t0})
                rows.append(m)
                print(f"    {name:6s} {f}  lum {m['d_lum']:.4f}  "
                      f"hist {m['d_hist']:.4f}  struct {m['d_struct']:.4f}  "
                      f"flow {m['d_flow']:.4f}  dir {m['d_flowdir']:5.2f}"
                      + (f"  siglip {m['d_siglip']:+.4f}" if sig else "")
                      + f"  ({m['secs']:.0f}s)", flush=True)
                (out / f"lpi_{args.tag}.json").write_text(json.dumps(
                    {"meta": {"config": {k: v for k, v in vars(args).items()},
                              "samples": [p.name for p in picked],
                              "sites": [s[0] for s in sites], "factors": factors},
                     "rows": rows}))

    tfm.forward = orig_fwd
    patcher.detach()
    print(f"\n[save] {out / f'lpi_{args.tag}.json'}   "
          f"{len(rows)} cells   {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
