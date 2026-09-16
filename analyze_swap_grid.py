#!/usr/bin/env python3
"""Window grid vs ceiling, joined across the two runs.

`runs/swap45` holds the ceiling (the swapped caption in all 28 blocks) for every
screened sample; `runs/swapgrid` holds the seven-window grid for the eight
samples with the largest ceiling effect. Each window cell is divided by that
sample's own ceiling, so windows are comparable across samples whose effect sizes
differ by an order of magnitude.

Reported per factor and metric:
  recovery    median over samples of window / ceiling
  argmax      how often each window is the sample's largest, against chance 1/7
  band        L12-19 (the region the first, uncorrected pass pointed at) against
              the other five windows, sign test at chance 2/7

    source env.sh
    $PY analyze_swap_grid.py
"""
import glob
import json
from math import comb

import numpy as np

MET = ("d_toward", "d_lum", "d_hist", "d_struct", "d_flow", "d_flowdir")


def order_sites(rows):
    """Sites as they appear along the stack, taken from the data rather than a
    fixed list, so the same read-out serves the four-block and two-block grids."""
    return sorted({r["site"] for r in rows if r["site"] != "all"},
                  key=lambda s: int(s.split("-")[0]))


def sign_p(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    ceil_rows = [r for f in glob.glob("runs/swap45/swap_*.json")
                 for r in json.load(open(f))["rows"] if r["site"] == "all"]
    import sys
    gdir = sys.argv[1] if len(sys.argv) > 1 else "runs/swapgrid"
    grid = [r for f in glob.glob(f"{gdir}/swap_*.json")
            for r in json.load(open(f))["rows"] if r["site"] != "all"]
    print(f"[grid] {gdir}")
    print(f"[data] {len(ceil_rows)} ceiling rows, {len(grid)} window cells")
    if not grid:
        return
    ce = {(r["factor"], r["sample"]): r for r in ceil_rows}

    for fac in ("A", "G", "D"):
        SITES = order_sites([r for r in grid if r["factor"] == fac])
        if not SITES:
            continue
        band = [s for s in SITES if 12 <= int(s.split("-")[0]) <= 19] or SITES[-2:]
        smp = sorted({r["sample"] for r in grid if r["factor"] == fac})
        smp = [s for s in smp
               if (fac, s) in ce
               and all(any(r["sample"] == s and r["factor"] == fac
                           and r["site"] == si for r in grid) for si in SITES)]
        if not smp:
            continue
        print(f"\n=== {fac}: window / ceiling, {len(smp)} 샘플 ===")
        print(f"  {'site':8s}" + "".join(f"{m.replace('d_', ''):>11s}" for m in MET))
        rec = {}
        for si in SITES:
            row = []
            for m in MET:
                v = []
                for s in smp:
                    a = [r[m] for r in grid if r["sample"] == s
                         and r["factor"] == fac and r["site"] == si]
                    b = ce[(fac, s)][m]
                    if a and abs(b) > 1e-9:
                        v.append(a[0] / b)
                rec[(si, m)] = v
                row.append(f"{np.median(v) * 100:>10.0f}%" if v else f"{'-':>11s}")
            print(f"  {si:8s}" + "".join(row))

        print(f"\n  샘플별 최대 window (chance 1/{len(SITES)}):")
        for m in MET:
            am = []
            for s in smp:
                v = [next((r[m] for r in grid if r["sample"] == s
                           and r["factor"] == fac and r["site"] == si), np.nan)
                     for si in SITES]
                am.append(SITES[int(np.nanargmax(v))])
            top = max(set(am), key=am.count)
            k, n = am.count(top), len(am)
            print(f"    {m:11s} " + " ".join(f"{a:>7s}" for a in am)
                  + f"   최빈 {top} {k}/{n}  p={sign_p(k, n, 1 / len(SITES)):.3f}")

        rest = [s for s in SITES if s not in band]
        if rest:
            pb = len(band) / len(SITES)
            print(f"\n  {'+'.join(band)} vs 나머지 {len(rest)} (chance {pb:.2f}):")
            for m in MET:
                w = 0
                for s in smp:
                    g = {si: next((r[m] for r in grid if r["sample"] == s
                                   and r["factor"] == fac and r["site"] == si),
                                  np.nan) for si in SITES}
                    w += max(g[x] for x in band) > max(g[x] for x in rest)
                n = len(smp)
                print(f"    {m:11s} {w}/{n}   p={sign_p(w, n, pb):.4f}")


if __name__ == "__main__":
    main()
