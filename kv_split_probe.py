#!/usr/bin/env python3
"""Does the factor arrive through the keys or through the values?

The text-only probe found a dissociation: appearance separates in `to_v` at
every layer while geometry and dynamics separate in `to_k` at L20-L26. If that
holds causally, A and D need LoRA on *different projections*, not just different
layers.

`to_k` and `to_v` are separate Linear modules that happen to be called with the
same tensor, so a forward pre-hook can feed them different text:

    k-only   to_k sees the factor prompt, to_v sees the base prompt
    v-only   the other way round
    both     the ordinary case, for reference

    restore(mode, L, w) = <v_inj - v_base, v_full - v_base> / || v_full - v_base ||^2

so the three modes partition the factor's effect at that block set into a key
part and a value part. `both` is the ceiling; if k-only ~= both then the factor
enters through attention placement, and if v-only ~= both it rides on the content
that attention fetches.

Restricted to the two sites the activation probe found (A at L2-L5, D at
L20-L26) plus the whole stack as a reference, so it costs one tenth of a full
sweep.

    source env.sh
    CUDA_VISIBLE_DEVICES=1 $PY kv_split_probe.py --out ./runs/kv1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from cosmos_common import NEG, build_pipe, latent_shape
from minimal_pair_probe import BASE, PHRASES, build_prompts

SITES = {"A_site_L2-5": [2, 3, 4, 5],
         "D_site_L20-26": [20, 21, 22, 23, 24, 25, 26],
         "all_L0-27": list(range(28))}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/kv")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--image", default=None)
    ap.add_argument("--step_every", type=int, default=8)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="kv")
    ap.add_argument("--per_layer", default=None,
                    help="single-layer resolution over this range, e.g. '0-13'. "
                         "to_k and to_v are the only modules that take text, so "
                         "this is the full text-entry grid: layer x {k,v} x step.")
    ap.add_argument("--modes", default="k_only,v_only,both")
    return ap.parse_args()


def main():
    args = parse_args()
    import os
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Path(args.image or (Path(os.environ["AGD"]) / "scene-0626_f24" / "cond_image.jpg"))
    image = Image.open(img).convert("RGB")
    prompts = build_prompts(args.base, PHRASES)
    words = [w for g, w, _ in prompts if g != "base"]
    grp = {w: g for g, w, _ in prompts if g != "base"}
    global SITES
    if args.per_layer:
        a, b = args.per_layer.split("-")
        SITES = {f"L{l}": [l] for l in range(int(a), int(b) + 1)}

    pipe = build_pipe()
    tfm = pipe.transformer
    device = torch.device("cuda")
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[base] {args.base}\n[model] {len(tfm.transformer_blocks)} blocks   "
          f"latent {tuple(shape)}   image {img.name}")

    with torch.no_grad():
        emb = {w: pipe._get_t5_prompt_embeds(p, device=device) for _, w, p in prompts}

    # substitution table: (layer, "k"|"v") -> tensor to feed that projection
    sub = {}

    def pre_hook(layer, which):
        def hook(module, inp):
            t = sub.get((layer, which))
            return (t,) + tuple(inp[1:]) if t is not None else None
        return hook

    handles = []
    for l, blk in enumerate(tfm.transformer_blocks):
        handles.append(blk.attn2.to_k.register_forward_pre_hook(pre_hook(l, "k")))
        handles.append(blk.attn2.to_v.register_forward_pre_hook(pre_hook(l, "v")))
    print(f"[hooks] {len(handles)} projections instrumented")

    probe_steps = list(range(0, args.steps, args.step_every))
    modes = tuple(m for m in args.modes.split(",") if m)
    n = 1 + 1 + len(words) + len(SITES) * len(modes) * len(words)
    print(f"[plan] {n} forwards x {len(probe_steps)} steps = {n * len(probe_steps)}")

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z0 = torch.randn(shape, generator=g, dtype=torch.float32)

    orig = tfm.forward
    state = {"step": 0, "call": 0, "busy": False}
    res, controls = {}, []

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(kw):
        t0 = time.time()
        kwb = dict(kw)
        kwb["encoder_hidden_states"] = emb["base"]
        sub.clear()
        v_base = flat(orig(**kwb))

        # control: substituting the base text into both projections changes nothing
        for l in range(len(tfm.transformer_blocks)):
            sub[(l, "k")] = emb["base"]
            sub[(l, "v")] = emb["base"]
        d0 = float((flat(orig(**kwb)) - v_base).norm())
        controls.append(d0)
        sub.clear()

        d_full = {}
        for w in words:
            kwv = dict(kw)
            kwv["encoder_hidden_states"] = emb[w]
            d_full[w] = flat(orig(**kwv)) - v_base

        for site, layers in SITES.items():
            for mode in modes:
                for w in words:
                    sub.clear()
                    for l in layers:
                        if mode in ("k_only", "both"):
                            sub[(l, "k")] = emb[w]
                        if mode in ("v_only", "both"):
                            sub[(l, "v")] = emb[w]
                    d = flat(orig(**kwb)) - v_base
                    r = float((d @ d_full[w]) /
                              d_full[w].dot(d_full[w]).clamp_min(1e-20))
                    res.setdefault(site, {}).setdefault(mode, {}) \
                       .setdefault(w, []).append(r)
                    sub.clear()
        print(f"  [kv] step {state['step']:3d}  "
              f"{len(SITES) * len(modes) * len(words)} injections in "
              f"{time.time() - t0:.0f}s   null={d0:.2e}", flush=True)

    def fwd(*a, **kw):
        v = orig(*a, **kw)
        if state["busy"]:
            return v
        if state["call"] == 0 and state["step"] in probe_steps:
            state["busy"] = True
            try:
                with torch.no_grad():
                    probe(kw)
            finally:
                state["busy"] = False
        state["call"] += 1
        return v

    tfm.forward = fwd

    def cb(p, i, t, kwargs):
        state["step"] += 1
        state["call"] = 0
        return kwargs

    t0 = time.time()
    with torch.no_grad():
        pipe(image=image, prompt=args.base, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps, generator=torch.Generator(device="cpu").manual_seed(args.seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    tfm.forward = orig
    for h in handles:
        h.remove()

    rec = {"meta": {"base": args.base, "image": str(img), "words": words,
                    "groups": grp, "sites": SITES, "modes": list(modes),
                    "probe_steps": probe_steps,
                    "config": {k: v for k, v in vars(args).items()}},
           "restore": res, "null": controls}
    fp = out / f"kv_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   {time.time() - t0:.0f}s")
    print(f"[control] base-into-both moved the output by at most {max(controls):.2e}")


if __name__ == "__main__":
    main()
