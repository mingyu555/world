#!/usr/bin/env python3
"""Where is a factor readable from the prompt alone? No image, no latent, no training.

Text reaches the DiT through exactly two projections per block: `attn2.to_k` and
`attn2.to_v`. Both act on the T5 embedding and nothing else, so

    K_l = norm_k(to_k_l(T5(prompt)))        V_l = to_v_l(T5(prompt))

are computable from the prompt by itself -- no conditioning image, no z_t, no
denoising step, no fine-tuning. If the four appearance prompts land in a
different place at layer l than the four geometry prompts, that layer *reads*
the factor. Whether it then acts on it is a separate question, which the
activation probes answer.

Two levels are reported:

  T5      the encoder output itself, before any DiT weight touches it.
          This is the ceiling: a factor the text encoder does not separate
          cannot be separated by any layer downstream.
  K / V   per layer, after that block's own projection.

Statistic, per level and per layer: each prompt is summarised by its mean over
*real* tokens (padding excluded -- 91% of cross-attention mass sits on padding,
so including it would drown the signal), taken relative to the base prompt.
Group separation is the same permutation F used by the activation probes, with
the same filler group as the control.

    source env.sh
    CUDA_VISIBLE_DEVICES=1 $PY text_only_probe.py --out ./runs/textonly
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cosmos_common import build_pipe
from minimal_pair_probe import BASE, PHRASES, build_prompts

FACTORS = ("A", "G", "D")
RNG = np.random.default_rng(0)


def f_ratio(V, labels, groups):
    """One-way F for every row of V at once (rows = layers/levels)."""
    idx = {g: np.where(labels == g)[0] for g in groups}
    n, k = V.shape[1], len(groups)
    gm = V.mean(1, keepdims=True)
    btw = np.zeros(V.shape[0])
    wth = np.zeros(V.shape[0])
    for g in groups:
        sub = V[:, idx[g]]
        mu = sub.mean(1)
        btw += sub.shape[1] * (mu - gm[:, 0]) ** 2
        wth += ((sub - mu[:, None]) ** 2).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (btw / (k - 1)) / (wth / (n - k))


def perm_p(V, labels, groups, n_perm=20000):
    f0 = f_ratio(V, labels, groups)
    ge = np.ones(V.shape[0])
    for _ in range(n_perm):
        ge += f_ratio(V, RNG.permutation(labels), groups) >= f0
    return f0, ge / (n_perm + 1)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/textonly")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(args.base, PHRASES)
    words = [w for g, w, _ in prompts if g != "base"]
    groups = np.array([g for g, w, _ in prompts if g != "base"])
    print(f"[base] {args.base}")
    print(f"[groups] " + "  ".join(f"{g}={sum(groups == g)}"
                                  for g in dict.fromkeys(groups)))

    pipe = build_pipe()
    tfm = pipe.transformer
    tok = pipe.tokenizer
    device = torch.device("cuda")
    nl = len(tfm.transformer_blocks)

    # ---- T5 embeddings, real tokens only ---------------------------------- #
    emb, mask = {}, {}
    with torch.no_grad():
        for _, w, p in prompts:
            enc = tok(p, padding="max_length", max_length=512, truncation=True,
                      return_tensors="pt")
            m = enc["attention_mask"][0].bool()
            emb[w] = pipe._get_t5_prompt_embeds(p, device=device)[0]   # [512, 1024]
            mask[w] = m.to(device)
    nreal = {w: int(mask[w].sum()) for w in emb}
    print(f"[tokens] real tokens: base {nreal['base']}, "
          f"variants {sorted(set(nreal[w] for w in words))}  (padding excluded)")

    # ---- level 0: the T5 output itself ------------------------------------ #
    def summarise(t, m):
        return t[m].float().mean(0)

    rows = {}
    b = summarise(emb["base"], mask["base"])
    rows["T5"] = torch.stack([summarise(emb[w], mask[w]) - b for w in words])

    # ---- levels 1..: per-layer K and V ------------------------------------ #
    with torch.no_grad():
        for l, blk in enumerate(tfm.transformer_blocks):
            a2 = blk.attn2
            for which, proj in (("K", a2.to_k), ("V", a2.to_v)):
                vecs = []
                for w in words + ["base"]:
                    x = proj(emb[w].to(next(proj.parameters()).dtype))
                    if which == "K" and getattr(a2, "norm_k", None) is not None:
                        h = a2.heads
                        x = a2.norm_k(x.unflatten(-1, (h, -1))).flatten(-2)
                    vecs.append(summarise(x, mask[w]))
                base_v = vecs[-1]
                rows[f"{which}{l}"] = torch.stack([v - base_v for v in vecs[:-1]])

    # ---- separation test -------------------------------------------------- #
    names = list(rows)
    # each row: 16 prompts x dim -> pairwise cosine, then the group statistic on
    # the per-prompt norm and on the within/between cosine
    all_groups = list(dict.fromkeys(groups))
    normV = np.stack([rows[n].norm(dim=1).cpu().numpy() for n in names])
    f_norm, p_norm = perm_p(normV, groups, all_groups)
    q_norm = bh(p_norm)

    gram, sep = {}, {}
    for n in names:
        M = rows[n]
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-20)
        G = (M @ M.T).cpu().numpy()
        gram[n] = G
        s = {}
        for g in all_groups:
            i = np.where(groups == g)[0]
            o = np.where(groups != g)[0]
            win = np.mean([G[a, b2] for a in i for b2 in i if a != b2])
            bet = np.mean([G[a, b2] for a in i for b2 in o])
            s[g] = win - bet
        sep[n] = s

    print(f"\n=== 그룹 응집 마진 (그룹 내 cos − 그룹 간 cos) ===")
    print(f"  {'level':7s}" + "".join(f"{g:>10s}" for g in all_groups)
          + f"{'‖·‖ F':>9s}{'q':>8s}")
    order = ["T5"] + [f"{w}{l}" for l in range(nl) for w in ("K", "V")]
    for i, n in enumerate(names):
        pass
    idx = {n: i for i, n in enumerate(names)}
    for n in ["T5"] + [f"K{l}" for l in range(0, nl, 3)] + [f"V{l}" for l in range(0, nl, 3)]:
        if n not in sep:
            continue
        i = idx[n]
        print(f"  {n:7s}" + "".join(f"{sep[n][g]:>10.3f}" for g in all_groups)
              + f"{f_norm[i]:>9.2f}{q_norm[i]:>8.3f}")

    print(f"\n=== factor 별 최고 응집 layer (filler 마진을 뺀 값) ===")
    for g in FACTORS:
        best = sorted(((sep[n][g] - sep[n]["filler"], n) for n in sep if n != "T5"),
                      reverse=True)[:6]
        print(f"  {g}: " + "  ".join(f"{n}({v:+.3f})" for v, n in best))
    print(f"  T5 자체: " + "  ".join(f"{g}={sep['T5'][g] - sep['T5']['filler']:+.3f}"
                                    for g in FACTORS))

    res = {"base": args.base, "words": words, "groups": groups.tolist(),
           "levels": names,
           "sep": {n: {g: float(v) for g, v in sep[n].items()} for n in sep},
           "F_norm": f_norm.tolist(), "q_norm": q_norm.tolist(),
           "nreal": nreal}
    (out / "textonly.json").write_text(json.dumps(res, indent=1))
    print(f"\n[save] {out/'textonly.json'}")


if __name__ == "__main__":
    main()
