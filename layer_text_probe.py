#!/usr/bin/env python3
"""Which block's *reading* of the prompt produces the factor's effect on the output?

Everything measured so far asked how differently a block writes. That is not the
same question as how much of the factor's effect on the model output that block
accounts for -- a block can write differently and change nothing downstream.

`attn2` takes the text as an argument, so the causal question is directly
askable: hand one block the factor prompt and every other block the base prompt.
Whatever moves in the output is what that block's reading of the text
contributes.

    v_base        base text in every block
    v_full(w)     word w's text in every block          the factor's total effect
    v_inj(L, w)   base text everywhere except blocks L, which get w

    restore(L, w) = <v_inj - v_base, v_full - v_base> / || v_full - v_base ||^2

restore is a projection, so it reads as "the share of this word's effect on the
output that blocks L account for", and shares from disjoint block sets are
comparable. Two controls come for free:

  base at L      handing L the base text changes nothing -> restore = 0 exactly
  filler words   four same-length, same-shape, meaningless phrases give the null
                 that a factor's restore has to beat

No training, no weight change. One frozen trajectory; the injections are extra
forwards at the same latent, so nothing accumulates across steps.

Coarse-to-fine: `--group 4` tests seven blocks of four layers (129 forwards per
probed step); rerun with `--group 1 --layers <range>` to split whichever block
wins.

    source env.sh
    CUDA_VISIBLE_DEVICES=1 $PY layer_text_probe.py --out ./runs/ltp1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from block_patch import BlockPatcher
from cosmos_common import NEG, build_pipe, latent_shape
from minimal_pair_probe import BASE, PHRASES, build_prompts


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/ltp")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--image", default=None)
    ap.add_argument("--group", type=int, default=4, help="layers per injected block set")
    ap.add_argument("--layers", default=None,
                    help="restrict to these layers, e.g. '16-27' (default: all)")
    ap.add_argument("--step_every", type=int, default=8)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="ltp")
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
    groups = {w: g for g, w, _ in prompts if g != "base"}

    pipe = build_pipe()
    tfm = pipe.transformer
    nl = len(tfm.transformer_blocks)
    device = torch.device("cuda")
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[base] {args.base}")
    print(f"[model] {nl} blocks   latent {tuple(shape)}   image {img.name}")

    if args.layers:
        a, b = args.layers.split("-")
        pool = list(range(int(a), int(b) + 1))
    else:
        pool = list(range(nl))
    sets = [pool[i:i + args.group] for i in range(0, len(pool), args.group)]
    print(f"[sets] {len(sets)} block sets: " +
          "  ".join(f"L{s[0]}" + (f"-{s[-1]}" if len(s) > 1 else "") for s in sets))

    with torch.no_grad():
        emb = {w: pipe._get_t5_prompt_embeds(p, device=device)
               for _, w, p in prompts}
    n_fwd = (1 + len(words) + len(sets) * len(words) + 1)
    probe_steps = list(range(0, args.steps, args.step_every))
    print(f"[plan] {n_fwd} forwards x {len(probe_steps)} steps = "
          f"{n_fwd * len(probe_steps)}")

    zpath = out / "z_T_shared.pt"
    if zpath.exists():
        z0 = torch.load(zpath)
        assert tuple(z0.shape) == tuple(shape)
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn(shape, generator=g, dtype=torch.float32)
        torch.save(z0, zpath)

    patcher = BlockPatcher(tfm)
    patcher.attach()
    orig = tfm.forward
    state = {"step": 0, "call": 0, "busy": False}
    res = {}          # setname -> word -> [per step]
    refnorm = {w: [] for w in words}
    controls = []

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(kw):
        t0 = time.time()
        kwb = dict(kw)
        kwb["encoder_hidden_states"] = emb["base"]
        v_base = flat(orig(**kwb))

        # control: overriding a block with the base text must change nothing
        with patcher.text_override({l: emb["base"] for l in pool}):
            d0 = float((flat(orig(**kwb)) - v_base).norm())
        controls.append(d0)

        d_full = {}
        for w in words:
            kwv = dict(kw)
            kwv["encoder_hidden_states"] = emb[w]
            d_full[w] = flat(orig(**kwv)) - v_base
            refnorm[w].append(float(d_full[w].norm()))

        for s in sets:
            name = f"L{s[0]}" + (f"-{s[-1]}" if len(s) > 1 else "")
            for w in words:
                with patcher.text_override({l: emb[w] for l in s}):
                    d = flat(orig(**kwb)) - v_base
                r = float((d @ d_full[w]) / d_full[w].dot(d_full[w]).clamp_min(1e-20))
                res.setdefault(name, {}).setdefault(w, []).append(r)
        print(f"  [ltp] step {state['step']:3d}  {len(sets) * len(words)} injections "
              f"in {time.time() - t0:.0f}s   null={d0:.2e}", flush=True)

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
    patcher.detach()

    rec = {"meta": {"base": args.base, "image": str(img), "words": words,
                    "groups": groups, "sets": [f"L{s[0]}" + (f"-{s[-1]}" if len(s) > 1 else "")
                                               for s in sets],
                    "set_layers": {f"L{s[0]}" + (f"-{s[-1]}" if len(s) > 1 else ""): s
                                   for s in sets},
                    "probe_steps": probe_steps, "n_layers": nl,
                    "config": {k: v for k, v in vars(args).items()}},
           "restore": res, "ref_norm": refnorm, "null": controls}
    fp = out / f"ltp_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   {time.time() - t0:.0f}s")
    print(f"[control] base-text override moved the output by at most "
          f"{max(controls):.2e} (must be 0)")


if __name__ == "__main__":
    main()
