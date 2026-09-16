#!/usr/bin/env python3
"""Which weights have leverage on each factor?  (axis B; plan sections 25-26)

`factor_layer_probe.py` (axis A) measures where a factor's effect *currently*
flows.  This measures where a factor's effect *could be changed from* -- i.e.
which parameters would receive gradient if we trained for that factor.  It is
the cheap predictor of the post-training weight-change localisation (axis C),
and it needs no training data at all.

At the reference latent z_t of the AGD trajectory, with the factor-removed
prediction as a detached target:

    v_ref(g)  = f(z_t, AGD\\g)                    no grad, detached
    L_g       = 1/2 || f(z_t, AGD) - v_ref(g) ||^2
    grad_g    = dL_g / dW                        one backward per (factor, step)

Because dL/dW = J_W^T (v_full - v_ref), the gradient norm at a module is exactly
how much that module can move the output *along the factor's own difference
direction*.  Reported three ways:

    grad_norm    ||g||                    raw
    leverage     ||g|| * ||W||            first-order effect of a *relative*
                                          weight change -- the scale-free score,
                                          and the one to rank placements by
    rel          ||g|| / ||W||

Bonus, free: cos(grad_A, grad_G) per module is a *pre-training* estimate of
parameter-space interference between two factors at that module -- the quantity
plan section 40 currently defines only through a performance drop, and what
section 55's orthogonality loss would target.

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=1 $PY grad_probe.py \\
      --sample $AGD/scene-0626_f24 --out ./runs/scene-0626_f24 \\
      --frames 45 --step_every 7
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from cosmos_common import NEG, REMOVE, build_pipe, factor_phrases, latent_shape

# transformer_blocks.<l>.<group>...  ->  the (layer, group) a parameter belongs to
BLOCK_RE = re.compile(r"transformer_blocks\.(\d+)\.(attn1|attn2|ff|norm1|norm2|norm3)\.(.+)")


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
    ap.add_argument("--tag", default="grad")
    ap.add_argument("--no_checkpointing", action="store_true")
    ap.add_argument("--allow_dirty", action="store_true")
    ap.add_argument("--z_shared", default=None)
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

    print(f"[sample] {sample.name}   text_field={tf}")
    dirty = []
    for f in factors:
        ref = meta["captions"][REMOVE[f]][tf]
        if phrases[f] and phrases[f] not in prompt:
            dirty.append(f"{f}: phrase is not verbatim in the AGD caption")
        if phrases[f] and phrases[f] in ref:
            dirty.append(f"{f}: phrase LEAKS into its removal reference {REMOVE[f]}")
    if dirty:
        msg = "contaminated factor contrast:\n  " + "\n  ".join(dirty)
        if not args.allow_dirty:
            raise SystemExit(f"[abort] {msg}\n  (--allow_dirty to measure anyway)")
        print(f"[warn] {msg}")

    pipe = build_pipe()
    tfm = pipe.transformer
    n_layers = len(tfm.transformer_blocks)
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[model] {n_layers} blocks   latent {tuple(shape)}")

    if not args.no_checkpointing:
        tfm.enable_gradient_checkpointing()
        print("[mem] gradient checkpointing on")

    device = torch.device("cuda")
    with torch.no_grad():
        emb_remove = {g: pipe._get_t5_prompt_embeds(meta["captions"][REMOVE[g]][tf],
                                                    device=device)
                      for g in factors}

    # parameters we track: every weight inside a block, grouped by (layer, group)
    tracked, groups = [], []
    for name, p in tfm.named_parameters():
        m = BLOCK_RE.match(name)
        if m is None or p.ndim < 2:          # skip norms' scalars / non-matrices
            p.requires_grad_(False)
            continue
        p.requires_grad_(True)
        tracked.append(p)
        groups.append((int(m.group(1)), m.group(2), name))
    w_norm = {}
    for (l, grp, name), p in zip(groups, tracked):
        w_norm[name] = p.detach().float().norm().item()
    print(f"[grad] tracking {len(tracked)} weight matrices "
          f"({len(set((l, g) for l, g, _ in groups))} layer x group cells)")

    probe_steps = list(range(0, args.steps, args.step_every))
    print(f"[plan] {len(factors)} factors x {len(probe_steps)} steps "
          f"= {len(factors) * len(probe_steps)} backward passes")

    if args.z_shared and Path(args.z_shared).exists():
        z0 = torch.load(args.z_shared)
        assert tuple(z0.shape) == tuple(shape)
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn(shape, generator=g, dtype=torch.float32)

    orig = tfm.forward
    state = {"step": 0, "call": 0, "busy": False}
    # (layer, group, name) -> factor -> list over steps
    acc = defaultdict(lambda: defaultdict(list))
    # flattened gradient per factor, for cross-factor cosine per (layer, group)
    cos_acc = defaultdict(lambda: defaultdict(list))
    ref_norm = {g: [] for g in factors}

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(kw):
        t0 = time.time()
        with torch.no_grad():
            v_full_ng = flat(orig(**kw))
            targets = {}
            for g in factors:
                kw_g = dict(kw)
                kw_g["encoder_hidden_states"] = emb_remove[g]
                targets[g] = flat(orig(**kw_g))
                ref_norm[g].append((targets[g] - v_full_ng).norm().item())

        per_factor_grads = {}
        for g in factors:
            with torch.enable_grad():
                v = flat(orig(**kw))
                loss = 0.5 * (v - targets[g]).pow(2).sum()
                gr = torch.autograd.grad(loss, tracked, retain_graph=False,
                                         allow_unused=True)
            cell = defaultdict(list)
            for (l, grp, name), gi in zip(groups, gr):
                if gi is None:
                    continue
                gn = gi.detach().float().norm().item()
                acc[(l, grp, name)][g].append(gn)
                cell[(l, grp)].append(gi.detach().flatten())
            per_factor_grads[g] = {k: torch.cat(v) for k, v in cell.items()}
            del gr, cell
            torch.cuda.empty_cache()

        for i, g1 in enumerate(factors):
            for g2 in factors[i + 1:]:
                for key in per_factor_grads[g1]:
                    a, b = per_factor_grads[g1][key], per_factor_grads[g2].get(key)
                    if b is None:
                        continue
                    c = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0)
                    cos_acc[key][f"{g1}-{g2}"].append(c.item())
        del per_factor_grads
        torch.cuda.empty_cache()
        print(f"  [grad] step {state['step']:3d}  {len(factors)} backwards in "
              f"{time.time() - t0:.0f}s", flush=True)

    def fwd(*a, **kw):
        v = orig(*a, **kw)
        if state["busy"]:
            return v
        if state["call"] == 0 and state["step"] in probe_steps:
            state["busy"] = True
            try:
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
        pipe(image=image, prompt=prompt, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps, generator=torch.Generator(device="cpu").manual_seed(args.seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    tfm.forward = orig

    rec = {
        "meta": {"sample": str(sample), "sample_id": meta["sample_id"],
                 "prompt": prompt, "phrases": phrases, "factors": factors,
                 "n_layers": n_layers, "probe_steps": probe_steps,
                 "latent_shape": list(shape),
                 "config": {k: v for k, v in vars(args).items()}},
        "w_norm": w_norm,
        "grad": {f"{l}|{grp}|{name}": dict(d) for (l, grp, name), d in acc.items()},
        "cos": {f"{l}|{grp}": dict(d) for (l, grp), d in cos_acc.items()},
        "ref_norm": ref_norm,
    }
    fp = out / f"grad_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
