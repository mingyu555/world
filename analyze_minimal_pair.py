#!/usr/bin/env python3
"""Which (layer, branch) responds to a factor rather than to a word?

Reads `minimal_pair_probe.py` output: 16 prompts (4 phrases x 4 groups) that
differ from one fixed base prompt by one appended phrase, and the per-(layer,
branch) effect each one has.

Per cell, the question is whether the four groups separate. The test statistic
is the one-way F ratio across groups -- between-group variance over within-group
variance -- and the null is the same statistic with the word-to-group labels
shuffled, so it needs no distributional assumption:

    F(cell) = MS_between / MS_within        over the 4 groups of 4 phrases
    p(cell) = fraction of label shuffles reaching F or higher

84 cells are tested at once, so p is controlled with Benjamini-Hochberg FDR.

Three further readings, each answering a question the earlier analyses could not:

  filler margin   (mean over factor groups - mean over filler) / filler
                  The filler phrases are the same shape and length and carry no
                  factor content, so this is what is left after prompt length and
                  syntax are accounted for -- plan section 10's control.
  length coupling corr(effect, added token count) across the 16 phrases.
                  A cell that just tracks prompt length shows up here.
  output Gram     within-group vs between-group cosine of the output differences.
                  Do the four appearance phrases move the output in a common
                  direction that the geometry phrases do not?

    python analyze_minimal_pair.py ./runs/minpair
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

BRANCHES = ("attn1", "attn2", "ff")
FACTORS = ("A", "G", "D")
RNG = np.random.default_rng(0)


def f_ratio(vals, labels, groups):
    """One-way F across `groups`; vals and labels are aligned 1-D arrays."""
    gm = vals.mean()
    between = within = 0.0
    dfb = len(groups) - 1
    dfw = len(vals) - len(groups)
    for g in groups:
        v = vals[labels == g]
        if len(v) == 0:
            return np.nan
        between += len(v) * (v.mean() - gm) ** 2
        within += ((v - v.mean()) ** 2).sum()
    if within <= 0 or dfw <= 0:
        return np.nan
    return (between / dfb) / (within / dfw)


def bh(p):
    """Benjamini-Hochberg: returns the q-value for each p."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tag", default="minpair")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()
    d = Path(args.run_dir)
    rec = torch.load(d / f"minpair_{args.tag}.pt", weights_only=False)
    meta = rec["meta"]
    words = meta["words"]
    groups = np.array(meta["groups"])
    nl = meta["n_layers"]
    added = np.array([meta["added_tokens"][w] for w in words], float)

    print(f"[base] {meta['base']}")
    print(f"[groups] " + "  ".join(
        f"{g}={sum(groups == g)}" for g in dict.fromkeys(groups)))
    print(f"[added tokens] " + "  ".join(
        f"{g}:{sorted(set(added[groups == g].astype(int)))}"
        for g in dict.fromkeys(groups)))

    # cell -> per-word effect, averaged over the probed steps
    cells = sorted({k for w in words for k in rec["effect"][w]},
                   key=lambda s: (s.split("|")[1], int(s.split("|")[0])))
    E = {}
    for c in cells:
        E[c] = np.array([np.mean(rec["effect"][w][c]) for w in words])

    all_groups = list(dict.fromkeys(groups))
    obs, ps = {}, {}
    for c in cells:
        v = E[c]
        f0 = f_ratio(v, groups, all_groups)
        obs[c] = f0
        if not np.isfinite(f0):
            ps[c] = 1.0
            continue
        hits = 0
        for _ in range(args.perm):
            f = f_ratio(v, RNG.permutation(groups), all_groups)
            if np.isfinite(f) and f >= f0:
                hits += 1
        ps[c] = (hits + 1) / (args.perm + 1)
    q = dict(zip(cells, bh([ps[c] for c in cells])))

    sig = [c for c in cells if q[c] <= args.q]
    print(f"\n=== 그룹 분리 검정: {len(cells)}칸 중 FDR q<={args.q} 통과 {len(sig)}칸 ===")
    if not sig:
        print("  없음 — 어느 (layer, branch) 도 factor 그룹을 유의하게 구분하지 못한다")
    for c in sorted(sig, key=lambda c: -obs[c]):
        l, b = int(c.split("|")[0]), c.split("|")[1]
        v = E[c]
        means = {g: v[groups == g].mean() for g in all_groups}
        top = max(FACTORS, key=lambda g: means[g])
        fac = np.mean([means[g] for g in FACTORS])
        marg = (means[top] - means["filler"]) / max(means["filler"], 1e-12) * 100
        print(f"  {b:6s} L{l:<3d} F={obs[c]:6.2f} q={q[c]:.3f}  "
              + "  ".join(f"{g}={means[g]:.4f}" for g in all_groups)
              + f"   최대={top}  filler 대비 {marg:+.0f}%")

    # ---- branch level: pooled over layers ---------------------------------- #
    print(f"\n=== branch별 요약 (layer 평균) ===")
    print(f"  {'branch':7s}" + "".join(f"{g:>10s}" for g in all_groups)
          + f"{'F':>8s}{'p':>8s}   factor/filler")
    for b in BRANCHES:
        cs = [c for c in cells if c.split("|")[1] == b]
        if not cs:
            continue
        v = np.mean([E[c] for c in cs], axis=0)
        f0 = f_ratio(v, groups, all_groups)
        hits = sum(1 for _ in range(args.perm)
                   if (lambda f: np.isfinite(f) and f >= f0)(
                       f_ratio(v, RNG.permutation(groups), all_groups)))
        p = (hits + 1) / (args.perm + 1)
        means = {g: v[groups == g].mean() for g in all_groups}
        ratio = np.mean([means[g] for g in FACTORS]) / max(means["filler"], 1e-12)
        print(f"  {b:7s}" + "".join(f"{means[g]:>10.4f}" for g in all_groups)
              + f"{f0:>8.2f}{p:>8.4f}   {ratio:>6.2f}x")

    # ---- length coupling --------------------------------------------------- #
    print(f"\n=== 길이 결합: corr(effect, 추가 토큰 수) ===")
    for b in BRANCHES:
        cs = [c for c in cells if c.split("|")[1] == b]
        rs = [np.corrcoef(E[c], added)[0, 1] for c in cs if np.std(E[c]) > 0]
        if rs:
            print(f"  {b:7s} 평균 r={np.mean(rs):+.3f}   최대 |r|={np.max(np.abs(rs)):.3f}")

    # ---- output-space clustering ------------------------------------------- #
    G = np.array(rec["gram"]).mean(0)
    print(f"\n=== 출력 공간: 그룹 내 vs 그룹 간 코사인 (출력 차이 방향) ===")
    print(f"  {'group':8s}{'내부':>9s}{'외부':>9s}{'margin':>9s}")
    for g in all_groups:
        idx = np.where(groups == g)[0]
        oth = np.where(groups != g)[0]
        win = np.mean([G[i, j] for i in idx for j in idx if i != j])
        bet = np.mean([G[i, j] for i in idx for j in oth])
        print(f"  {g:8s}{win:>9.3f}{bet:>9.3f}{win - bet:>9.3f}")

    res = {"meta": meta, "F": obs, "p": ps, "q": q,
           "effect": {c: E[c].tolist() for c in cells},
           "gram_mean": G.tolist()}
    (d / "minpair_analysis.json").write_text(json.dumps(res, indent=1))
    print(f"\n[save] {d/'minpair_analysis.json'}")


if __name__ == "__main__":
    main()
