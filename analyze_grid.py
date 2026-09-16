#!/usr/bin/env python3
"""The window grid, conditioned on the ceiling.

`runs/screen` holds the ceiling for every clean sample: how much each factor's
caption changes the video when injected into all 28 blocks. It found a dynamics
effect that reaches the pixels in 10 of 59 samples, an appearance effect in 3 and
a geometry effect in 1 -- so an unconditioned grid mostly averages samples where
there is nothing to localise, which is exactly what diluted the first pass.

`runs/grid` holds the seven-window grid for the 10 dynamics samples. Each cell is
divided by that sample's own ceiling, giving "how much of the effect this window
accounts for", which is comparable across samples with wildly different effect
sizes (the ceiling flow ratio ranges from 2.2x to 86x).

Statistics, all paired within sample:
  recovery    metric(window) / metric(ceiling), per sample
  contrast    D injection vs the larger of A/G at the same window
  sign test   in how many samples does D beat both A and G here (binomial, p=1/3
              under the null that the three factors are exchangeable)
  permutation factor labels shuffled within sample

    source env.sh
    $PY analyze_grid.py
"""
import json
import glob
from math import comb

import numpy as np

F = ("A", "G", "D")
SITES = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27"]
MET = "d_flow"
ALT = ["d_flowdir", "d_struct"]


def load(pat, site_filter=None):
    rows = [r for f in glob.glob(pat) for r in json.load(open(f))["rows"]]
    return rows


def main():
    sel = json.load(open("runs/screen/selected.json"))
    strong = sel["strong_D"]
    scr = load("runs/screen/lpi_*.json")
    grid = load("runs/grid/lpi_*.json")

    def ceil(s, f, m):
        x = [r[m] for r in scr if r["sample"] == s and r["site"] == "all"
             and r["factor"] == f]
        return float(x[0]) if x else np.nan

    def cell(s, site, f, m):
        x = [r[m] for r in grid if r["sample"] == s and r["site"] == site
             and r["factor"] == f]
        return float(x[0]) if x else np.nan

    have = [s for s in strong
            if all(not np.isnan(cell(s, si, f, MET))
                   for si in SITES for f in F)]
    print(f"[data] {len(grid)} grid cells; {len(have)}/{len(strong)} "
          f"dynamics samples complete")
    if not have:
        return

    for m in [MET] + ALT:
        print(f"\n=== [{m}] window 가 ceiling 의 D 효과를 몇 % 복원하나 (샘플별) ===")
        print(f"  {'site':8s}" + "".join(f"{s.split('-')[1][:4]:>8s}" for s in have)
              + f"{'중앙값':>9s}{'평균':>8s}")
        for si in SITES:
            rec = [cell(s, si, "D", m) / max(ceil(s, "D", m), 1e-12) for s in have]
            print(f"  {si:8s}" + "".join(f"{x * 100:>7.0f}%" for x in rec)
                  + f"{np.median(rec) * 100:>8.0f}%{np.mean(rec) * 100:>7.0f}%")

    print(f"\n=== [{MET}] D 주입 vs max(A,G) 주입, 같은 window ===")
    print(f"  {'site':8s}{'D':>9s}{'max(A,G)':>10s}{'비율':>7s}"
          f"{'D 승':>7s}{'p(부호)':>9s}{'p(순열)':>9s}")
    rng = np.random.default_rng(0)
    res = []
    for si in SITES:
        d = np.array([cell(s, si, "D", MET) for s in have])
        o = np.array([max(cell(s, si, "A", MET), cell(s, si, "G", MET))
                      for s in have])
        wins = int((d > o).sum())
        n = len(have)
        # binomial: under exchangeability D is the largest of three with p = 1/3
        p_sign = sum(comb(n, k) * (1 / 3) ** k * (2 / 3) ** (n - k)
                     for k in range(wins, n + 1))
        # paired permutation on the within-sample normalised triple
        M = []
        for s in have:
            t = np.array([cell(s, si, f, MET) for f in F])
            mu = np.abs(t).mean()
            M.append(t / mu if mu > 1e-12 else t)
        M = np.array(M)
        obs = M.mean(0)[2] - M.mean(0)[:2].max()
        ge = 1
        for _ in range(20000):
            P = np.array([rng.permutation(r) for r in M])
            ge += (P.mean(0)[2] - P.mean(0)[:2].max()) >= obs
        p_perm = ge / 20001
        res.append((si, p_sign, p_perm))
        print(f"  {si:8s}{d.mean():>9.3f}{o.mean():>10.3f}"
              f"{d.mean() / max(o.mean(), 1e-12):>7.2f}{wins:>4d}/{n}"
              f"{p_sign:>9.4f}{p_perm:>9.4f}")

    ps = np.array([r[2] for r in res])
    nq = len(ps)
    q = np.empty(nq)
    prev = 1.0
    for rank, i in enumerate(np.argsort(ps)[::-1]):
        prev = min(prev, ps[i] * nq / (nq - rank))
        q[i] = prev
    print(f"\n  BH-FDR: " + "  ".join(f"{r[0]}:q={qq:.3f}"
                                     + ("*" if qq < 0.10 else "")
                                     for r, qq in zip(res, q)))


if __name__ == "__main__":
    main()
