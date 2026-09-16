#!/usr/bin/env python3
"""Inside attn2: does the factor arrive through to_k or through to_v?

Cosmos `attn2` has no biases anywhere, and the pipeline zero-pads the T5
sequence to 512. Two things follow mechanically:

    to_v(0) = 0                 a padding position contributes nothing to the sum
    norm_k(to_k(0)) = 0         a padding position's logit is exactly q.0 = 0

So a padding position is not attended-to content -- it is a fixed `512 - r`
count of `exp(0) = 1` terms sitting in the softmax denominator:

    out = sum_{j real} P_j v_j ,   mass_real = S / (S + 512 - r)
    S   = sum_{j real} exp(q.k_j / sqrt(d))

which divides the whole cross-attention branch down by 20-90x. That is the
mechanical reason `attn2` writes only ~3% of ||h|| while carrying most of the
factor sensitivity: its direction is 100% text-derived `v`, just scaled small.

Because padding is exactly neutral in both projections, `to_k` and `to_v` can be
fed *different* prompts with no alignment problem -- token counts need not match,
since a position the keys use but the values do not simply contributes zero:

    out(kv)  k from the factor prompt, v from the factor prompt   the full effect
    out(kb)  k from the factor prompt, v from the base prompt     placement only
    out(bv)  k from the base prompt,   v from the factor prompt   payload only

    d   = out(kv) - out(bb)
    r_k = <out(kb) - out(bb), d> / ||d||^2
    r_v = <out(bv) - out(bb), d> / ||d||^2

r_k and r_v are projections onto the factor's own effect, so they are comparable
and sum to 1 plus an interaction term. r_v near 1 means the factor rides on the
content the attention fetches; r_k near 1 means it rides on where the attention
looks (and on how much mass escapes the padding).

`mass_real` is reported separately: it depends on the keys alone, so a prompt
that adds tokens raises it mechanically -- which is exactly why the filler group
has to carry the same token count.

All 28 layers, every prompt, one captured `q` per layer per step: one forward
total, unlike `kv_split_probe.py`, which re-runs the whole model per injection.
The tradeoff is that this is local to the branch and says nothing about
downstream propagation, which is the point -- it isolates the projection.

    source env.sh
    CUDA_VISIBLE_DEVICES=1 $PY kv_branch_probe.py --out ./runs/kvb
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cosmos_common import NEG, build_pipe, latent_shape
from minimal_pair_probe import BASE, PHRASES, build_prompts

FACTORS = ("A", "G", "D")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/kvb")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--image", default=None)
    ap.add_argument("--step_every", type=int, default=8)
    ap.add_argument("--nq", type=int, default=4096, help="queries sampled per layer")
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="kvb")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Path(args.image or (Path(os.environ["AGD"]) / "scene-0626_f24" / "cond_image.jpg"))
    image = Image.open(img).convert("RGB")

    prompts = build_prompts(args.base, PHRASES)
    words = [w for g, w, _ in prompts if g != "base"]
    grp = {w: g for g, w, _ in prompts if g != "base"}

    pipe = build_pipe()
    tfm = pipe.transformer
    tok = pipe.tokenizer
    device = torch.device("cuda")
    nl = len(tfm.transformer_blocks)

    with torch.no_grad():
        emb, real = {}, {}
        for _, w, p in prompts:
            emb[w] = pipe._get_t5_prompt_embeds(p, device=device)
            real[w] = int(tok(p, padding="max_length", max_length=512, truncation=True,
                              return_tensors="pt")["attention_mask"][0].sum())

    # the two structural facts this probe rests on, checked rather than assumed
    a0 = tfm.transformer_blocks[0].attn2
    assert all(getattr(a0, n).bias is None for n in ("to_q", "to_k", "to_v"))
    z = torch.zeros(1, 2, emb["base"].shape[-1], device=device,
                    dtype=next(a0.to_v.parameters()).dtype)
    with torch.no_grad():
        assert float(a0.to_v(z).norm()) == 0.0
        kz = a0.to_k(z).unflatten(-1, (a0.heads, -1))
        assert float(a0.norm_k(kz).norm()) == 0.0
    pad_norm = max(float(emb[w][0, real[w]:].norm()) for w in emb)
    print(f"[structure] attn2 has no biases; to_v(0)=0, norm_k(to_k(0))=0; "
          f"max ||T5 padding||={pad_norm:.2e}")
    assert pad_norm == 0.0, "padding is not exactly zero; the decomposition breaks"
    print(f"[tokens] base {real['base']} real, variants "
          f"{sorted(set(real[w] for w in words))} of 512")

    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    probe_steps = list(range(0, args.steps, args.step_every))
    print(f"[plan] {nl} layers x {len(words)} words x {len(probe_steps)} steps, "
          f"{args.nq} queries")

    grab = {}
    handles = []
    for l, blk in enumerate(tfm.transformer_blocks):
        def hook(module, inp, l=l):
            grab[l] = inp[0].detach()
        handles.append(blk.attn2.to_q.register_forward_pre_hook(hook))

    rows = []          # one dict per (step, layer, word)
    massrow = []       # one dict per (step, layer), base prompt
    state = {"step": 0}

    def analyse():
        t0 = time.time()
        for l, blk in enumerate(tfm.transformer_blocks):
            a2 = blk.attn2
            H = a2.heads
            with torch.no_grad():
                q = a2.to_q(grab[l])
                D = q.shape[-1] // H
                q = q.unflatten(-1, (H, D)).transpose(1, 2)
                if a2.norm_q is not None:
                    q = a2.norm_q(q)
                nq = min(args.nq, q.shape[2])
                q = q[:, :, :nq].float()

                def keys(e):
                    k = a2.to_k(e).unflatten(-1, (H, D)).transpose(1, 2)
                    if a2.norm_k is not None:
                        k = a2.norm_k(k)
                    return k.float()

                def vals(e):
                    return a2.to_v(e).unflatten(-1, (H, D)).transpose(1, 2).float()

                def probs(k, nreal):
                    """softmax over all 512 positions, done on the real ones only:
                    padding logits are exactly 0, so they contribute a constant
                    `512 - nreal` to the denominator and nothing to the numerator."""
                    ex = (q @ k[:, :, :nreal].transpose(-1, -2) / D ** 0.5).exp()
                    S = ex.sum(-1, keepdim=True)
                    denom = S + (512 - nreal)
                    return ex / denom, S / denom

                nb = real["base"]
                kb, vb = keys(emb["base"]), vals(emb["base"])
                pb, mb = probs(kb, nb)
                out_bb = pb @ vb[:, :, :nb]
                massrow.append({"step": state["step"], "layer": l,
                                "mass_base": float(mb.mean()),
                                "out_norm": float(out_bb.norm())})
                for w in words:
                    n = real[w]
                    kf, vf = keys(emb[w]), vals(emb[w])
                    pf, mf = probs(kf, n)
                    out_kv = pf @ vf[:, :, :n]                 # factor k, factor v
                    out_kb = pf @ vb[:, :, :n]                 # factor k, base   v
                    out_bv = pb @ vf[:, :, :nb]                # base   k, factor v
                    d = out_kv - out_bb
                    df = d.flatten()
                    dsq = df.dot(df).clamp_min(1e-20)
                    rk = float((out_kb - out_bb).flatten().dot(df) / dsq)
                    rv = float((out_bv - out_bb).flatten().dot(df) / dsq)
                    rows.append({
                        "step": state["step"], "layer": l, "word": w, "group": grp[w],
                        "d": float(d.norm()), "out_norm": float(out_bb.norm()),
                        "rel": float(d.norm() / out_bb.norm().clamp_min(1e-20)),
                        "r_k": rk, "r_v": rv, "r_int": 1.0 - rk - rv,
                        "mag_k": float((out_kb - out_bb).norm() / d.norm().clamp_min(1e-20)),
                        "mag_v": float((out_bv - out_bb).norm() / d.norm().clamp_min(1e-20)),
                        "mass_base": float(mb.mean()), "mass_f": float(mf.mean()),
                    })
        print(f"  [kvb] step {state['step']:3d}  {nl} layers in {time.time()-t0:.0f}s",
              flush=True)

    orig = tfm.forward
    st = {"call": 0, "busy": False}

    def fwd(*a, **kw):
        v = orig(*a, **kw)
        if not st["busy"] and st["call"] == 0 and state["step"] in probe_steps:
            st["busy"] = True
            try:
                analyse()
            finally:
                st["busy"] = False
        st["call"] += 1
        return v

    tfm.forward = fwd

    def cb(p, i, t, kwargs):
        state["step"] += 1
        st["call"] = 0
        return kwargs

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z0 = torch.randn(shape, generator=g, dtype=torch.float32)
    t0 = time.time()
    with torch.no_grad():
        pipe(image=image, prompt=args.base, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps,
             generator=torch.Generator(device="cpu").manual_seed(args.seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    tfm.forward = orig
    for h in handles:
        h.remove()

    rec = {"meta": {"base": args.base, "image": str(img), "words": words,
                    "groups": grp, "real": real, "n_layers": nl,
                    "probe_steps": probe_steps,
                    "config": {k: v for k, v in vars(args).items()}},
           "rows": rows, "mass": massrow}
    fp = out / f"kvb_{args.tag}.json"
    fp.write_text(json.dumps(rec))
    print(f"[save] {fp}   {time.time()-t0:.0f}s")

    # ------------------------------- read-out ------------------------------- #
    R = rows
    steps = sorted({r["step"] for r in R})

    print("\n=== 패딩이 cross-attn 분기를 얼마나 나누는가 (실토큰 확률질량) ===")
    print(f"  {'layer':6s}" + "".join(f"{'s' + str(s):>10s}" for s in steps))
    for l in range(0, nl, 3):
        v = [next(m["mass_base"] for m in massrow
                  if m["layer"] == l and m["step"] == s) for s in steps]
        print(f"  L{l:<5d}" + "".join(f"{x:>10.4f}" for x in v))

    print("\n=== 인자 효과가 어느 projection 을 타는가 (d 로의 사영) ===")
    print(f"  {'layer':6s}{'|d|':>8s}{'rel':>8s}{'r_k':>8s}{'r_v':>8s}"
          f"{'r_int':>8s}{'mass':>8s}")
    for l in range(nl):
        sub = [r for r in R if r["layer"] == l]
        print(f"  L{l:<5d}{np.mean([r['d'] for r in sub]):>8.3f}"
              f"{np.mean([r['rel'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_k'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_v'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_int'] for r in sub]):>8.3f}"
              f"{np.mean([r['mass_base'] for r in sub]):>8.4f}")

    print("\n=== factor 별 (전 레이어 평균) ===")
    print(f"  {'group':8s}{'|d|':>8s}{'rel':>8s}{'r_k':>8s}{'r_v':>8s}{'r_int':>8s}")
    for f in FACTORS + ("filler",):
        sub = [r for r in R if r["group"] == f]
        print(f"  {f:8s}{np.mean([r['d'] for r in sub]):>8.3f}"
              f"{np.mean([r['rel'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_k'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_v'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_int'] for r in sub]):>8.3f}")

    print("\n=== factor x layer 대역: r_v (괄호는 filler 대비) ===")
    bands = [("L0-6", range(0, 7)), ("L7-13", range(7, 14)),
             ("L14-19", range(14, 20)), ("L20-27", range(20, 28))]
    print(f"  {'group':8s}" + "".join(f"{b:>14s}" for b, _ in bands))
    fil = {b: np.mean([r["r_v"] for r in R
                       if r["group"] == "filler" and r["layer"] in lay])
           for b, lay in bands}
    for f in FACTORS:
        row = []
        for b, lay in bands:
            v = np.mean([r["r_v"] for r in R
                         if r["group"] == f and r["layer"] in lay])
            row.append(f"{v:.3f}({v - fil[b]:+.3f})".rjust(14))
        print(f"  {f:8s}" + "".join(row))
    print(f"  {'filler':8s}" + "".join(f"{fil[b]:>14.3f}" for b, _ in bands))

    print("\n=== 스텝별 (전 레이어 평균) ===")
    print(f"  {'step':6s}{'r_k':>8s}{'r_v':>8s}{'mass':>8s}{'rel':>8s}")
    for s_ in steps:
        sub = [r for r in R if r["step"] == s_]
        print(f"  s{s_:<5d}{np.mean([r['r_k'] for r in sub]):>8.3f}"
              f"{np.mean([r['r_v'] for r in sub]):>8.3f}"
              f"{np.mean([r['mass_base'] for r in sub]):>8.4f}"
              f"{np.mean([r['rel'] for r in sub]):>8.3f}")


if __name__ == "__main__":
    main()
