#!/usr/bin/env python3
"""Which layer carries Appearance / Geometry / Dynamics?  (plan sections 26-28)

Teacher-forced, branch-resolved activation patching on Cosmos-Predict2-2B.

For one sample we run the AGD trajectory once.  At the probed denoising steps,
*at the reference latent* z_t, we compare raw transformer outputs:

    v_full      = f(z_t, AGD)
    v_ref(g)    = f(z_t, AGD\\g)                     factor g deleted from the caption
    v_patch     = f(z_t, AGD) with the residual delta of one (layer, branch)
                  replaced by the delta the AGD\\g stream wrote there

    restore(g, l, b, t) = <v_patch - v_full, v_ref(g) - v_full> / ||v_ref(g) - v_full||^2

`restore` is the fraction of factor g's effect on the model output that flows
through that (layer, branch).  Because a Cosmos block is pure residual, patching
every tap point reproduces the reference stream exactly, so the scores are on a
common 0..1 scale and are close to additive (verified in test_block_patch.py:
sum of single-tap restores = 1.0005 on a toy stack).

Why not compare finished videos, and why not only blind cross-attention:

  * end-of-trajectory comparison measures chaotic amplification -- a step-0
    difference of 4e-4 grows ~1000x by step 34 and saturates, which is why
    cosmos_exp/causal_ablation.py gave align 0.25 for every layer group alike;
  * blinding attn2 to a token span (cosmos_exp/causal_probe.py) only tests where
    the text is *read in*.  Text enters Cosmos only through attn2, but the factor
    is *rendered* by attn1 (spatio-temporal) and ff.  Patching covers all three.

Two controls are computed for free:

  own   patching the AGD stream's own delta      -> restore must be 0 (numerical floor)
  L0/attn1  self attention at layer 0 sees identical input in both streams
            -> restore must be 0 (structural floor)

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=0 $PY factor_layer_probe.py \\
      --sample $AGD/scene-0014_f16 --out ./runs/scene-0014_f16 \\
      --frames 29 --group 4 --step_every 7

  # single-layer resolution once the coarse pass points somewhere
  CUDA_VISIBLE_DEVICES=0 $PY factor_layer_probe.py ... --group 1 --tag L1
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from block_patch import BRANCHES, BlockPatcher, restore_scores
from cosmos_common import NEG, REMOVE, build_pipe, factor_phrases, latent_shape


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="agd_dataset/<sample_id> directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--factors", default="A,G,D")
    ap.add_argument("--branches", default="attn1,attn2,ff",
                    help="comma list from attn1,attn2,ff; 'sum' patches all three together")
    ap.add_argument("--group", type=int, default=4, help="layers per tap group")
    ap.add_argument("--step_every", type=int, default=7)
    # generation settings -- Cosmos defaults except --frames, which is the memory knob
    ap.add_argument("--frames", type=int, default=29,
                    help="93 = Cosmos default (~40 GB); 29 -> latent T=8; 13 -> T=4")
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_device", default="cpu",
                    help="where captured deltas live; 'cuda' is faster if it fits")
    ap.add_argument("--vae_tiling", action="store_true")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--allow_dirty", action="store_true",
                    help="run even if a factor phrase leaks into its removal reference")
    ap.add_argument("--z_shared", default=None,
                    help="reuse an existing z_T_shared.pt (must match --frames)")
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

    if args.branches == "sum":
        branch_sets = [("sum", list(BRANCHES))]
    else:
        branch_sets = [(b, [b]) for b in args.branches.split(",") if b]
    for _, bs in branch_sets:
        for b in bs:
            assert b in BRANCHES, f"unknown branch {b}"

    print(f"[sample] {sample.name}   text_field={tf}")
    print(f"[prompt] {prompt}")
    # The whole measurement rests on AGD\g really lacking factor g. 38 of the 100
    # agd_dataset samples violate that: some combination captions leak a factor
    # phrase they are labelled 0 for (scene-0014_f16's GD still contains the
    # appearance phrase, so "remove A" would measure word order, not appearance).
    dirty = []
    for f in factors:
        ref = meta["captions"][REMOVE[f]][tf]
        print(f"   phrase {f}: {phrases[f]!r}")
        print(f"   remove {f} -> {REMOVE[f]}: {ref}")
        if phrases[f] and phrases[f] not in prompt:
            dirty.append(f"{f}: phrase is not verbatim in the AGD caption")
        if phrases[f] and phrases[f] in ref:
            dirty.append(f"{f}: phrase LEAKS into its removal reference {REMOVE[f]}")
    if dirty:
        msg = "contaminated factor contrast:\n  " + "\n  ".join(dirty)
        if not args.allow_dirty:
            raise SystemExit(f"[abort] {msg}\n  (--allow_dirty to measure anyway)")
        print(f"[warn] {msg}")

    pipe = build_pipe(vae_tiling=args.vae_tiling)
    n_layers = len(pipe.transformer.transformer_blocks)
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[model] {n_layers} blocks   latent {tuple(shape)}")

    device = torch.device("cuda")
    with torch.no_grad():
        emb_remove = {g: pipe._get_t5_prompt_embeds(meta["captions"][REMOVE[g]][tf],
                                                    device=device)
                      for g in factors}

    # tap configurations: (layer group) x (branch set)
    groups = [list(range(i, min(i + args.group, n_layers)))
              for i in range(0, n_layers, args.group)]
    configs = []
    for bname, bs in branch_sets:
        for L in groups:
            tag = f"L{L[0]}" + (f"-{L[-1]}" if len(L) > 1 else "")
            configs.append({"layers": L, "branches": bs, "bname": bname,
                            "tag": f"{bname}@{tag}"})
    probe_steps = list(range(0, args.steps, args.step_every))
    n_fwd = len(probe_steps) * (2 + len(factors) * (2 + len(configs)))
    print(f"[plan] {len(configs)} tap configs x {len(factors)} factors "
          f"x {len(probe_steps)} steps  ->  ~{n_fwd} extra forwards")

    # shared initial noise (unit variance; prepare_latents multiplies by sigma_max)
    if args.z_shared and Path(args.z_shared).exists():
        z0 = torch.load(args.z_shared)
        assert tuple(z0.shape) == tuple(shape), f"z_shared shape {tuple(z0.shape)} != {tuple(shape)}"
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn(shape, generator=g, dtype=torch.float32)
        torch.save(z0, out / "z_T_shared.pt")

    patcher = BlockPatcher(pipe.transformer, cache_device=args.cache_device)
    patcher.attach()
    # only the taps we actually patch need to be cached (saves the copy of every
    # branch of every layer on each captured stream)
    null_key = (max(1, n_layers // 2), "attn2")           # tap used by the null control
    patcher.capture_keys = {(l, b) for c in configs for l in c["layers"]
                            for b in c["branches"]} | {null_key}

    results = {}          # tag -> factor -> list of per-step score dicts
    controls = {"own": [], "ref_norm": {g: [] for g in factors}, "v_norm": []}
    write_norm = []       # per probed step: {(layer, branch): ||delta|| / ||h_layer||}
    orig = pipe.transformer.forward
    state = {"step": 0, "call": 0, "busy": False}

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(v_full_raw, kw):
        t0 = time.time()
        v_full = flat(v_full_raw)
        controls["v_norm"].append(v_full.norm().item())

        # ---- AGD's own deltas: magnitudes (the denominator each factor writes into)
        with patcher.capture():
            _ = orig(**kw)
        own_norm, h_norm = {}, dict(patcher.h_norm)
        for (l, b), d in patcher.store.items():
            own_norm[f"{l}|{b}"] = d.float().norm().item()
        null_delta = {null_key: patcher.store[null_key].clone()}
        write_norm.append({"delta": own_norm, "h": {str(k): v for k, v in h_norm.items()}})
        patcher.clear()

        # ---- null control: re-inject AGD's own delta -> must be exactly 0
        with patcher.injection(null_delta):
            d_own = flat(orig(**kw)) - v_full
        controls["own"].append(d_own.norm().item())
        del null_delta

        # ---- pass 1: every reference direction first, so the specificity of a
        # tap can be measured against all factors (these vectors are small).
        d_ref = {}
        for g in factors:
            kw_g = dict(kw)
            kw_g["encoder_hidden_states"] = emb_remove[g]
            d_ref[g] = flat(orig(**kw_g)) - v_full
            controls["ref_norm"][g].append(d_ref[g].norm().item())

        # ---- pass 2: one captured stream in memory at a time (these are large)
        for g in factors:
            kw_g = dict(kw)
            kw_g["encoder_hidden_states"] = emb_remove[g]
            with patcher.capture():
                _ = orig(**kw_g)

            for c in configs:
                subs = {(l, b): patcher.store[(l, b)]
                        for l in c["layers"] for b in c["branches"]}
                with patcher.injection(subs):
                    d_patch = flat(orig(**kw)) - v_full
                sc = restore_scores(d_patch, d_ref[g])
                # specificity: does this tap also explain the other factors?
                sc["cross"] = {
                    h: restore_scores(d_patch, d_ref[h])["restore"]
                    for h in factors if h != g
                }
                sc["step"] = state["step"]
                results.setdefault(c["tag"], {}).setdefault(g, []).append(sc)
            patcher.clear()

        print(f"  [probe] step {state['step']:3d}  "
              f"{len(configs) * len(factors)} patches in {time.time() - t0:.0f}s  "
              f"null={controls['own'][-1]:.2e}", flush=True)

    def fwd(*a, **kw):
        v = orig(*a, **kw)
        if state["busy"]:
            return v
        if state["call"] == 0 and state["step"] in probe_steps:
            state["busy"] = True
            try:
                with torch.no_grad():
                    probe(v, kw)
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
        "meta": {
            "sample": str(sample), "sample_id": meta["sample_id"],
            "prompt": prompt, "phrases": phrases,
            "remove_prompts": {g: meta["captions"][REMOVE[g]][tf] for g in factors},
            "factors": factors, "n_layers": n_layers, "group": args.group,
            "branch_sets": [b for b, _ in branch_sets], "probe_steps": probe_steps,
            "latent_shape": list(shape),
            "config": {k: v for k, v in vars(args).items()},
        },
        "configs": {c["tag"]: {"layers": c["layers"], "branches": c["branches"]}
                    for c in configs},
        "results": results,
        "controls": controls,
        "write_norm": write_norm,
    }
    fp = out / f"factor_layer_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   total {time.time() - t0:.0f}s")
    print(f"[control] own-delta re-injection |d| = "
          f"{max(controls['own']):.3e} (must be ~0)")


if __name__ == "__main__":
    main()
