#!/usr/bin/env python3
"""Read out `layer_prompt_image_probe.py`: which block window controls which factor.

The probe injects one factor's caption into one four-block window and measures
the decoded video against the reference. This turns those numbers into the table
B-LoRA's Figure 3 is: window on one axis, what changed on the other.

Two normalisations matter.

Windows differ in overall influence -- injecting anything into L0-3 moves the
video more than injecting anything into L24-27 -- so a raw metric would just rank
windows by influence. Dividing by the window's own mean over the three factors
removes that:

    rel(w, f, m) = metric(w, f, m) / mean_g metric(w, g, m)

and rel > 1 means factor f drives metric m at window w more than its neighbours
do. The diagonal of the (factor x metric) block is then directly readable: A
should own the colour metrics, D the flow metrics.

Metrics also differ in scale, so the second view z-scores each metric across all
(window, factor) cells before comparing them.

`all` is the ceiling: the factor caption in every block. A window whose rel is
near the ceiling's accounts for that factor on its own.

    source env.sh
    $PY analyze_lpi.py runs/lpi
"""

import json
import sys
from pathlib import Path

import numpy as np

FACTORS = ("A", "G", "D")
METRICS = ("d_lum", "d_hist", "d_struct", "d_flow", "d_flowdir", "d_siglip")
# which factor each metric is supposed to be sensitive to
OWNER = {"d_lum": "A", "d_hist": "A", "d_struct": "G",
         "d_flow": "D", "d_flowdir": "D", "d_siglip": None}


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/lpi")
    files = sorted(root.glob("lpi_*.json"))
    rows, samples, sites = [], [], []
    for f in files:
        d = json.loads(f.read_text())
        rows += d["rows"]
        samples += d["meta"]["samples"]
        sites = d["meta"]["sites"] or sites
    if not rows:
        print(f"no rows under {root}")
        return
    have = sorted({r["sample"] for r in rows})
    sites = [s for s in sites if any(r["site"] == s for r in rows)]
    mets = [m for m in METRICS if m in rows[0]]
    print(f"[data] {len(rows)} cells   {len(have)} samples   "
          f"{len(sites)} sites   metrics {mets}")
    n_per = {(s, f): sum(1 for r in rows if r["site"] == s and r["factor"] == f)
             for s in sites for f in FACTORS}
    if len(set(n_per.values())) > 1:
        print(f"[warn] unbalanced cells: {sorted(set(n_per.values()))}")

    def cell(site, fac, met):
        v = [r[met] for r in rows if r["site"] == site and r["factor"] == fac]
        return float(np.mean(v)) if v else np.nan

    print("\n=== 원값: 주입 window x factor, 지표별 평균 ===")
    for met in mets:
        own = OWNER[met]
        print(f"\n  [{met}]" + (f"  (기대 우세: {own})" if own else "  (SigLIP)"))
        print(f"    {'site':8s}" + "".join(f"{f:>10s}" for f in FACTORS)
              + f"{'우세':>7s}")
        for s in sites:
            v = [cell(s, f, met) for f in FACTORS]
            i = int(np.nanargmax(v))
            print(f"    {s:8s}" + "".join(f"{x:>10.4f}" for x in v)
                  + f"{FACTORS[i]:>7s}")

    print("\n=== rel(w,f,m) = 지표 / 그 window 의 3 factor 평균  (>1 이면 우세) ===")
    print(f"  {'site':8s}" + "".join(f"{m.replace('d_', '') + '/' + f:>13s}"
                                     for m in mets for f in FACTORS))
    for s in sites:
        line = []
        for m in mets:
            v = np.array([cell(s, f, m) for f in FACTORS], dtype=float)
            mu = np.nanmean(np.abs(v))
            line += [f"{x / mu:>13.2f}" if mu > 1e-12 else f"{'-':>13s}" for x in v]
        print(f"  {s:8s}" + "".join(line))

    print("\n=== 대각 검정: 각 지표가 자기 factor 에서 가장 커지는 window ===")
    print(f"  {'metric':10s}{'owner':>6s}{'best site':>12s}{'rel':>7s}"
          f"{'2nd factor rel':>16s}{'대각?':>7s}")
    hits = 0
    tests = 0
    for m in mets:
        own = OWNER[m]
        if own is None:
            continue
        tests += 1
        best, brel, bsec = None, -np.inf, None
        for s in sites:
            if s == "all":
                continue
            v = np.array([cell(s, f, m) for f in FACTORS], dtype=float)
            mu = np.nanmean(np.abs(v))
            if mu <= 1e-12:
                continue
            rel = v[FACTORS.index(own)] / mu
            if rel > brel:
                brel = rel
                best = s
                bsec = max(v[i] / mu for i, f in enumerate(FACTORS) if f != own)
        ok = brel > bsec
        hits += ok
        print(f"  {m:10s}{own:>6s}{best:>12s}{brel:>7.2f}{bsec:>16.2f}"
              f"{'o' if ok else 'x':>7s}")
    print(f"  대각 성립 {hits}/{tests}")

    if "all" in sites:
        print("\n=== window / ceiling: 그 window 하나가 전체 효과의 몇 %인가 ===")
        print(f"  {'site':8s}" + "".join(f"{m.replace('d_', ''):>11s}" for m in mets))
        for s in sites:
            if s == "all":
                continue
            line = []
            for m in mets:
                num = np.mean([cell(s, f, m) for f in FACTORS])
                den = np.mean([cell("all", f, m) for f in FACTORS])
                line.append(f"{num / den * 100:>10.1f}%" if abs(den) > 1e-12
                            else f"{'-':>11s}")
            print(f"  {s:8s}" + "".join(line))

    print("\n=== 짝지은 순열검정: factor 가 지표를 설명하는가 (샘플 내 permute) ===")
    tests = paired_test(rows, sites, mets)
    print(f"  {'site':8s}{'metric':11s}{'n':>3s}" +
          "".join(f"{f:>8s}" for f in FACTORS) + f"{'p':>8s}{'q':>7s}")
    for o in sorted(tests, key=lambda x: x["q"]):
        star = " *" if o["q"] < 0.10 else ""
        print(f"  {o['site']:8s}{o['metric']:11s}{o['n']:>3d}" +
              "".join(f"{x:>8.3f}" for x in o["mean"]) +
              f"{o['p']:>8.4f}{o['q']:>7.3f}{star}")
    sig = [o for o in tests if o["q"] < 0.10]
    print(f"  q<0.10 통과: {len(sig)}/{len(tests)}")

    print("\n=== ceiling 패턴 재현도: window 의 (factor x 지표) 패턴 vs all ===")
    agree, cvec = ceiling_agreement(rows, sites, mets)
    print("  ceiling 패턴 (factor 별 편차): " +
          "  ".join(f"{m.replace('d_','')}:" +
                    "/".join(f"{cvec[i*3+j]:+.2f}" for j in range(3))
                    for i, m in enumerate(mets)))
    for s, v in sorted(agree.items(), key=lambda kv: -kv[1]):
        print(f"  {s:8s} cos = {v:+.3f}")

    print("\n=== 샘플 간 일관성: 지표별 우세 window 가 샘플마다 같은가 ===")
    print(f"  {'metric':10s}{'owner':>6s}  샘플별 최대 window")
    for m in mets:
        own = OWNER[m] or FACTORS[0]
        per = []
        for sm in have:
            best, brel = None, -np.inf
            for s in sites:
                if s == "all":
                    continue
                v = [r[m] for r in rows if r["sample"] == sm and r["site"] == s
                     and r["factor"] == own]
                if not v:
                    continue
                if np.mean(v) > brel:
                    brel = float(np.mean(v))
                    best = s
            per.append(best)
        top = max(set(per), key=per.count)
        print(f"  {m:10s}{own:>6s}  " + " ".join(str(p) for p in per)
              + f"   (최빈 {top}: {per.count(top)}/{len(per)})")


def paired_test(rows, sites, mets, n_perm=20000):
    """Is the metric explained by which factor was injected?

    Samples differ enormously in how much any injection moves them, so the test
    is paired within a sample: each cell is divided by that (sample, site)'s mean
    over the three factors, and the permutation shuffles factor labels *within*
    each sample. Unpaired shuffling would let between-sample variance masquerade
    as a factor effect.
    """
    rng = np.random.default_rng(0)
    samples = sorted({r["sample"] for r in rows})
    out = []
    for site in sites:
        for met in mets:
            M = []                      # [sample, factor]
            for sm in samples:
                v = [[r[met] for r in rows if r["sample"] == sm
                      and r["site"] == site and r["factor"] == f] for f in FACTORS]
                if any(len(x) != 1 for x in v):
                    continue
                row = np.array([x[0] for x in v], float)
                mu = np.abs(row).mean()
                if mu <= 1e-12:
                    continue
                M.append(row / mu)
            if len(M) < 3:
                continue
            M = np.array(M)
            def stat(X):
                m = X.mean(0)
                return float(((m - m.mean()) ** 2).sum())
            f0 = stat(M)
            ge = 1
            for _ in range(n_perm):
                P = np.array([rng.permutation(r) for r in M])
                ge += stat(P) >= f0
            out.append({"site": site, "metric": met, "n": len(M), "stat": f0,
                        "p": ge / (n_perm + 1),
                        "mean": M.mean(0).tolist()})
    ps = np.array([o["p"] for o in out])
    n = len(ps)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(np.argsort(ps)[::-1]):
        prev = min(prev, ps[i] * n / (n - rank))
        q[i] = prev
    for o, qq in zip(out, q):
        o["q"] = float(qq)
    return out


def ceiling_agreement(rows, sites, mets):
    """Does window w reproduce the ceiling's factor-specific pattern?

    The ceiling ("all", the factor caption in every block) is the only place the
    factor's real effect is guaranteed to be present. Stack a window's
    within-window-normalised (factor x metric) values into one vector and take
    the cosine with the ceiling's. A window that acts on the factors the way the
    full model does scores near 1; a window that only adds noise scores near 0.
    """
    def vec(site):
        v = []
        for m in mets:
            x = np.array([np.mean([r[m] for r in rows if r["site"] == site
                                   and r["factor"] == f]) for f in FACTORS], float)
            mu = np.abs(x).mean()
            v += list(x / mu - 1) if mu > 1e-12 else [0, 0, 0]
        return np.array(v)
    c = vec("all")
    return {s: float(vec(s) @ c / (np.linalg.norm(vec(s)) * np.linalg.norm(c) + 1e-20))
            for s in sites if s != "all"}, c


if __name__ == "__main__":
    main()
