#!/usr/bin/env python3
"""Pool the minimal-pair runs: does a cell separate the factors in every setting?

Each run uses a different base prompt and a different conditioning image. A cell
that only separates the groups in one setting is a property of that sentence or
that image, not of the factor. So two things are reported:

  pooled     effects averaged over runs, then the permutation F test
             (label shuffles, BH-FDR over all cells)
  agreement  in how many runs the same factor leads that cell

Only cells that pass the pooled test *and* lead with the same factor in every
run are treated as located. All phrases add exactly 4 T5 tokens, so the length
confound that ate the first geometry result is gone by construction; the check
is printed anyway.

    python aggregate_minpair.py runs/mp1 runs/mp2 runs/mp3
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
    gm = vals.mean()
    between = within = 0.0
    for g in groups:
        v = vals[labels == g]
        if len(v) == 0:
            return np.nan
        between += len(v) * (v.mean() - gm) ** 2
        within += ((v - v.mean()) ** 2).sum()
    dfb, dfw = len(groups) - 1, len(vals) - len(groups)
    if within <= 0 or dfw <= 0:
        return np.nan
    return (between / dfb) / (within / dfw)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(p)[::-1]):
        prev = min(prev, p[i] * n / (n - rank))
        q[i] = prev
    return q



def perm_test_rows(V, groups, all_groups, n_perm=4000, rng=None):
    """Permutation p-value of the one-way F, for every row of V at once.

    V is [n_cells, n_phrases]; labels are shuffled across phrases, which is the
    exchangeability the null needs (a cell's response to a phrase carries no
    group information under H0).
    """
    rng = rng or np.random.default_rng(0)
    idx = {g: np.where(groups == g)[0] for g in all_groups}
    n, k = V.shape[1], len(all_groups)
    dfb, dfw = k - 1, n - k

    def F(order):
        gm = V.mean(1, keepdims=True)
        between = np.zeros(V.shape[0])
        within = np.zeros(V.shape[0])
        for g in all_groups:
            sub = V[:, order[idx[g]]]
            mu = sub.mean(1)
            between += sub.shape[1] * (mu - gm[:, 0]) ** 2
            within += ((sub - mu[:, None]) ** 2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (between / dfb) / (within / dfw)

    base_order = np.arange(n)
    f0 = F(base_order)
    ge = np.ones(V.shape[0])
    for _ in range(n_perm):
        ge += F(rng.permutation(base_order)) >= f0
    return f0, ge / (n_perm + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--q", type=float, default=0.10)
    args = ap.parse_args()

    recs = []
    for r in args.runs:
        d = Path(r)
        fp = next(d.glob("minpair_*.pt"))
        recs.append((d.name, torch.load(fp, weights_only=False)))
    words = recs[0][1]["meta"]["words"]
    groups = np.array(recs[0][1]["meta"]["groups"])
    for name, rec in recs:
        assert rec["meta"]["words"] == words, f"{name}: phrase set differs"
        print(f"[{name}] base: {rec['meta']['base']}   image: "
              f"{Path(rec['meta']['image']).parent.name}")
    added = np.array([recs[0][1]["meta"]["added_tokens"][w] for w in words], float)
    print(f"[tokens] 추가 토큰 = {sorted(set(added.astype(int)))}"
          f"  (모두 동일하면 길이 교란 없음)")

    cells = sorted(set.intersection(*[set(rec["effect"][words[0]]) for _, rec in recs]),
                   key=lambda s: (s.split("|")[1], int(s.split("|")[0])))
    all_groups = list(dict.fromkeys(groups))

    # per run, per cell: the 16 phrase effects (averaged over probed steps)
    per_run = []
    for name, rec in recs:
        E = {c: np.array([np.mean(rec["effect"][w][c]) for w in words]) for c in cells}
        per_run.append((name, E))

    # pooled: normalise each run to its own mean so runs weigh equally
    pooled = {}
    for c in cells:
        vs = []
        for _, E in per_run:
            v = E[c]
            vs.append(v / max(v.mean(), 1e-20))
        pooled[c] = np.mean(vs, axis=0)

    obs, ps = {}, {}
    for c in cells:
        f0 = f_ratio(pooled[c], groups, all_groups)
        obs[c] = f0
        if not np.isfinite(f0):
            ps[c] = 1.0
            continue
        hits = sum(1 for _ in range(args.perm)
                   if np.isfinite(f := f_ratio(pooled[c], RNG.permutation(groups),
                                               all_groups)) and f >= f0)
        ps[c] = (hits + 1) / (args.perm + 1)
    q = dict(zip(cells, bh([ps[c] for c in cells])))

    def leader(v):
        m = {g: v[groups == g].mean() for g in FACTORS}
        return max(m, key=m.get)

    sig = [c for c in cells if q[c] <= args.q]
    print(f"\n=== 통합 검정: {len(cells)}칸 중 FDR q<={args.q} 통과 {len(sig)}칸 ===")
    print(f"  {'cell':11s}{'F':>7}{'q':>7}{'최대':>5}{'런일치':>7}  "
          f"{'A':>8}{'G':>8}{'D':>8}{'filler':>8}  filler대비")
    located = []
    for c in sorted(sig, key=lambda c: -obs[c]):
        l, b = c.split("|")[0], c.split("|")[1]
        v = pooled[c]
        m = {g: v[groups == g].mean() for g in all_groups}
        top = leader(v)
        leads = [leader(E[c]) for _, E in per_run]
        agree = sum(1 for x in leads if x == top)
        marg = (m[top] - m["filler"]) / max(m["filler"], 1e-12) * 100
        star = "*" if agree == len(per_run) else " "
        if agree == len(per_run):
            located.append((b, int(l), top, marg))
        print(f"  {b:6s}L{l:<4s}{obs[c]:7.2f}{q[c]:7.3f}{top:>5}{agree}/{len(per_run)}{star:>3}  "
              + "".join(f"{m[g]:>8.3f}" for g in all_groups)
              + f"  {marg:>+7.0f}%")

    print(f"\n=== 세 설정 모두에서 같은 factor 가 이긴 칸: {len(located)} ===")
    for t in FACTORS:
        cs = sorted([(b, l, mg) for b, l, top, mg in located if top == t], key=lambda x: x[1])
        print(f"  {t}: " + ("  ".join(f"{b}/L{l}({mg:+.0f}%)" for b, l, mg in cs)
                            if cs else "없음"))

    print(f"\n=== branch별 (layer 평균, 통합) ===")
    print(f"  {'branch':7s}" + "".join(f"{g:>9s}" for g in all_groups)
          + f"{'F':>8}{'p':>8}  factor/filler")
    for b in BRANCHES:
        cs = [c for c in cells if c.split("|")[1] == b]
        if not cs:
            continue
        v = np.mean([pooled[c] for c in cs], axis=0)
        f0 = f_ratio(v, groups, all_groups)
        hits = sum(1 for _ in range(args.perm)
                   if np.isfinite(f := f_ratio(v, RNG.permutation(groups), all_groups))
                   and f >= f0)
        p = (hits + 1) / (args.perm + 1)
        m = {g: v[groups == g].mean() for g in all_groups}
        ratio = np.mean([m[g] for g in FACTORS]) / max(m["filler"], 1e-12)
        print(f"  {b:7s}" + "".join(f"{m[g]:>9.3f}" for g in all_groups)
              + f"{f0:>8.2f}{p:>8.4f}  {ratio:>6.2f}x")

    # ---- timestep 분해: 같은 검정을 관측 step 마다 ----------------------- #
    steps = recs[0][1]["meta"]["probe_steps"]
    print(f"\n=== timestep 분해 ({len(steps)} 관측 step, step 마다 84칸 검정) ===")
    print(f"  {'step':>5}  {'통과칸':>6}  factor 별 국소화된 (branch/layer)")
    per_step_hits = {}
    for si, st in enumerate(steps):
        V = []
        for c in cells:
            vs = []
            for _, rec in recs:
                v = np.array([rec["effect"][w][c][si] for w in words], float)
                vs.append(v / max(v.mean(), 1e-20))
            V.append(np.mean(vs, axis=0))
        V = np.array(V)
        f0, pv = perm_test_rows(V, groups, all_groups, n_perm=4000)
        qq = bh(pv)
        hit = [(cells[i], V[i]) for i in range(len(cells)) if qq[i] <= args.q]
        per_step_hits[st] = []
        by = {t: [] for t in FACTORS}
        for c, v in hit:
            top = leader(v)
            m = {g: v[groups == g].mean() for g in all_groups}
            if m[top] <= m["filler"]:
                continue                      # filler 를 못 넘으면 버린다
            b, l = c.split("|")[1], int(c.split("|")[0])
            by[top].append((b, l))
            per_step_hits[st].append([b, l, top])
        txt = "   ".join(f"{t}: " + (",".join(f"{b}/L{l}" for b, l in sorted(by[t], key=lambda x: x[1]))
                                     or "-") for t in FACTORS)
        print(f"  {st:>5}  {len(per_step_hits[st]):>6}  {txt}")

    print(f"\n=== 출력 공간: 그룹 내 vs 그룹 간 코사인 (런 평균) ===")
    print(f"  {'group':8s}{'내부':>9s}{'외부':>9s}{'margin':>9s}   런별 margin")
    for g in all_groups:
        idx = np.where(groups == g)[0]
        oth = np.where(groups != g)[0]
        margins = []
        for _, rec in recs:
            G = np.array(rec["gram"]).mean(0)
            win = np.mean([G[i, j] for i in idx for j in idx if i != j])
            bet = np.mean([G[i, j] for i in idx for j in oth])
            margins.append(win - bet)
        G = np.mean([np.array(rec["gram"]).mean(0) for _, rec in recs], axis=0)
        win = np.mean([G[i, j] for i in idx for j in idx if i != j])
        bet = np.mean([G[i, j] for i in idx for j in oth])
        print(f"  {g:8s}{win:>9.3f}{bet:>9.3f}{win - bet:>9.3f}   "
              + "  ".join(f"{m:+.3f}" for m in margins))

    out = Path(args.runs[0]).parent / "minpair_pooled.json"
    out.write_text(json.dumps({
        "runs": [n for n, _ in recs], "words": words, "groups": groups.tolist(),
        "F": obs, "p": ps, "q": q,
        "pooled_effect": {c: pooled[c].tolist() for c in cells},
        "located": [[b, l, t, mg] for b, l, t, mg in located],
        "per_step": per_step_hits}, indent=1))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
