#!/usr/bin/env python3
"""When in the trajectory does the factor's text act?

Each row is one (sample, factor, step band) with the swapped caption injected into
all 28 blocks but only while the denoising step index is inside that band. The
band `0-19` is the control: it must reproduce the full-trajectory ceiling from
`runs/swap45` exactly, since gating on "every step" is a no-op.

Reported per factor and metric, as a share of that sample's own control band, so
samples with wildly different effect sizes are comparable:

    share(band) = metric(band) / metric(0-19)

and the argmax over the three thirds, sign-tested against chance 1/3.

    source env.sh
    $PY analyze_stepband.py
"""
import glob
import json
from math import comb

import numpy as np

MET = ("d_toward", "d_lum", "d_hist", "d_struct", "d_flow", "d_flowdir")
CTRL = "0-19"
BANDS = ["0-6", "7-13", "14-19"]


def sign_p(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def bh(ps):
    ps = np.asarray(ps, float)
    n = len(ps)
    q = np.empty(n)
    prev = 1.0
    for r, i in enumerate(np.argsort(ps)[::-1]):
        prev = min(prev, ps[i] * n / (n - r))
        q[i] = prev
    return q


def main():
    rows = [r for f in glob.glob("runs/stepband/swap_*.json")
            for r in json.load(open(f))["rows"]]
    if not rows:
        print("no rows under runs/stepband")
        return
    for r in rows:
        r["b"] = r["site"].split("@s")[-1]
    print(f"[data] {len(rows)} cells")

    # control: the full-range band must equal the earlier full-trajectory ceiling,
    # since gating on "every step" is a no-op. Keyed on the caption pair as well as
    # the sample: geometry here uses the hand-written future-topology pairs while
    # runs/swap45 used the dataset inventory, so a sample-only key would compare
    # two different prompts and the check would fail for the wrong reason.
    ceil = {}
    for src in ("runs/swap45", "runs/grecond"):
        for f in glob.glob(f"{src}/swap_*.json"):
            for r in json.load(open(f))["rows"]:
                if r["site"] == "all":
                    ceil[(r["sample"], r["phrase_old"], r["phrase_new"])] = r
    checked = matched = 0
    err = 0.0
    for r in rows:
        if r["b"] != CTRL:
            continue
        checked += 1
        c = ceil.get((r["sample"], r["phrase_old"], r["phrase_new"]))
        if c is None:
            continue
        matched += 1
        err = max(err, max(abs(r[m] - c[m]) for m in MET if m in c))
    print(f"[control] {matched}/{checked} full-range bands matched a stored "
          f"ceiling with the same caption pair: max abs diff {err:.2e}")
    if matched:
        assert err == 0.0, f"step gating changed the full-range result by {err}"

    for fac in ("A", "D", "G"):
        smp = sorted({r["sample"] for r in rows if r["factor"] == fac})
        smp = [s for s in smp
               if all(any(r["sample"] == s and r["factor"] == fac and r["b"] == b
                          for r in rows) for b in [CTRL] + BANDS)]
        if not smp:
            continue

        def cell(s, b, m):
            return next(r[m] for r in rows if r["sample"] == s
                        and r["factor"] == fac and r["b"] == b)

        print(f"\n=== {fac}: 각 스텝 구간이 전 구간 효과의 몇 %인가 "
              f"({len(smp)} 샘플, 중앙값) ===")
        print(f"  {'band':10s}" + "".join(f"{m.replace('d_', ''):>11s}" for m in MET))
        for b in [CTRL] + BANDS:
            line = []
            for m in MET:
                v = [cell(s, b, m) / cell(s, CTRL, m) for s in smp
                     if abs(cell(s, CTRL, m)) > 1e-9]
                line.append(f"{np.median(v) * 100:>10.0f}%" if v else f"{'-':>11s}")
            print(f"  {'s' + b:10s}" + "".join(line))

        print(f"\n  샘플별 최대 구간 (chance 1/3, 6지표 BH-FDR):")
        ps, out = [], []
        for m in MET:
            am = [BANDS[int(np.argmax([cell(s, b, m) for b in BANDS]))] for s in smp]
            top = max(set(am), key=am.count)
            k = am.count(top)
            p = sign_p(k, len(smp), 1 / 3)
            ps.append(p)
            out.append((m, am, top, k, p))
        for (m, am, top, k, p), q in zip(out, bh(ps)):
            print(f"    {m:11s} " + " ".join(f"{a:>6s}" for a in am)
                  + f"   최빈 {top} {k}/{len(smp)}  p={p:.3f} q={q:.3f}"
                  + ("  *" if q < 0.10 else ""))


if __name__ == "__main__":
    main()
