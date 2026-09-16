#!/usr/bin/env python3
"""Branch-resolved local divergence and write magnitude (axis "D_u", cheap version).

`cosmos_exp/analyze_same_noise.py` recovers a layer-local divergence D_u from the
stored sketches of nine full trajectories (~200 MB each). The same quantity, and
a strictly finer version of it, falls out of the patcher's captures in one run:

    delta_f(l,b)     what block l branch b writes to the residual stream under AGD
    delta_ref(l,b)   ... under AGD\\g, at the *same* latent z_t

    local_div(g,l,b) = || delta_AGD(l,b) - delta_ref(l,b) || / || h_l ||
    write(l,b)       = || delta_AGD(l,b) || / || h_l ||

`local_div` is the branch-resolved, depth-unbiased divergence: it is what that
one block does differently because of factor g, with no accumulation from below
(unlike D_h) and split across attn1 / attn2 / ff (unlike D_u).

Cost: 1 + |factors| forwards per probed step (20 forwards for 3 factors x 5
steps), and the output is a few KB.

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=1 $PY local_div_probe.py \\
      --sample $AGD/scene-0626_f24 --out ./runs/scene-0626_f24 --frames 45
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from block_patch import BRANCHES, BlockPatcher
from cosmos_common import NEG, REMOVE, build_pipe, factor_phrases, latent_shape


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--factors", default="A,G,D")
    ap.add_argument("--step_every", type=int, default=7)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="localdiv")
    ap.add_argument("--allow_dirty", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    sample, out = Path(args.sample), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((sample / "caption.json").read_text())
    tf = args.text_field
    prompt = meta["captions"]["AGD"][tf]
    phrases = factor_phrases(meta, tf)
    image = Image.open(sample / "cond_image.jpg").convert("RGB")
    factors = [f for f in args.factors.split(",") if f]

    dirty = [f"{f}: contaminated" for f in factors
             if phrases[f] and (phrases[f] not in prompt
                                or phrases[f] in meta["captions"][REMOVE[f]][tf])]
    if dirty and not args.allow_dirty:
        raise SystemExit(f"[abort] {dirty}  (--allow_dirty to measure anyway)")

    pipe = build_pipe()
    n_layers = len(pipe.transformer.transformer_blocks)
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[model] {n_layers} blocks   latent {tuple(shape)}")

    device = torch.device("cuda")
    with torch.no_grad():
        emb_remove = {g: pipe._get_t5_prompt_embeds(meta["captions"][REMOVE[g]][tf],
                                                    device=device)
                      for g in factors}

    zpath = out / "z_T_shared.pt"
    if zpath.exists():
        z0 = torch.load(zpath)
        assert tuple(z0.shape) == tuple(shape), "z_T_shared shape mismatch"
        print(f"[noise] reusing {zpath}")
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn(shape, generator=g, dtype=torch.float32)
        torch.save(z0, zpath)

    patcher = BlockPatcher(pipe.transformer, cache_device="cpu")
    patcher.attach()

    probe_steps = list(range(0, args.steps, args.step_every))
    print(f"[plan] {(1 + len(factors)) * len(probe_steps)} extra forwards")
    orig = pipe.transformer.forward
    state = {"step": 0, "call": 0, "busy": False}
    write = defaultdict(list)                       # (l,b) -> per step
    gram = defaultdict(list)                        # "A-G" -> cos per step
    ref_norm = defaultdict(list)
    local = defaultdict(lambda: defaultdict(list))  # (l,b) -> factor -> per step

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(kw):
        t0 = time.time()
        with patcher.capture():
            v_full = flat(orig(**kw))
        full = dict(patcher.store)
        h = dict(patcher.h_norm)
        for (l, b), d in full.items():
            write[(l, b)].append(d.float().norm().item() / max(h.get(l, 1.0), 1e-8))
        d_ref = {}
        for g in factors:
            kw_g = dict(kw)
            kw_g["encoder_hidden_states"] = emb_remove[g]
            with patcher.capture():
                d_ref[g] = flat(orig(**kw_g)) - v_full
            for (l, b), d in patcher.store.items():
                dd = (full[(l, b)].float() - d.float()).norm().item()
                local[(l, b)][g].append(dd / max(h.get(l, 1.0), 1e-8))
            patcher.clear()
        # Gram matrix of the three "remove factor" directions. If these are
        # themselves nearly parallel, no intervention anywhere can be specific to
        # one factor -- the ceiling is set by the contrast design, not the model.
        for i, g1 in enumerate(factors):
            for g2 in factors[i:]:
                c = torch.nn.functional.cosine_similarity(d_ref[g1], d_ref[g2], dim=0)
                gram[f"{g1}-{g2}"].append(c.item())
            ref_norm[g1].append(d_ref[g1].norm().item())
        del full, d_ref
        print(f"  [localdiv] step {state['step']:3d}  ref-Gram "
              + " ".join(f"{k}={v[-1]:+.2f}" for k, v in gram.items() if "-" in k
                         and k.split("-")[0] != k.split("-")[1])
              + f"  {1 + len(factors)} forwards "
              f"in {time.time() - t0:.0f}s", flush=True)

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

    pipe.transformer.forward = fwd

    def cb(p, i, t, kwargs):
        state["step"] += 1
        state["call"] = 0
        return kwargs

    t0 = time.time()
    with torch.no_grad():
        pipe(image=image, prompt=prompt, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps, generator=torch.Generator(device="cpu").manual_seed(args.seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    pipe.transformer.forward = orig
    patcher.detach()

    rec = {
        "meta": {"sample": str(sample), "sample_id": meta["sample_id"],
                 "prompt": prompt, "factors": factors, "n_layers": n_layers,
                 "branches": list(BRANCHES), "probe_steps": probe_steps,
                 "latent_shape": list(shape),
                 "config": {k: v for k, v in vars(args).items()}},
        "write": {f"{l}|{b}": v for (l, b), v in write.items()},
        "local_div": {f"{l}|{b}": dict(d) for (l, b), d in local.items()},
        "ref_gram": dict(gram),
        "ref_norm": dict(ref_norm),
    }
    fp = out / f"localdiv_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
