#!/usr/bin/env python3
"""Read out `swap_probe.py`: does swapping the factor phrase change the video?

Each row is one (sample, factor, site). The reference already commits to a value
of the factor, so this is a true minimal pair -- the swap replaces a
length-matched phrase core with the most semantically distant one in the
dataset's inventory for that slot.

The ceiling rows (site "all") answer the prior question the window grid depends
on: is there an effect to localise at all? Reported per factor:

  toward > 0    the video moved toward the swapped caption in SigLIP space.
                Sign test against 1/2, since under the null the video is as
                likely to drift either way.
  own metric    the pixel metric matched to the factor, against the median of
                the other two factors' effect on that same metric -- which is
                the specificity the first pass never established.
  frame 0       must be exactly 0: the pipeline overwrites the first latent frame
                with the conditioning image, so a nonzero value would mean the
                injection leaked somewhere it should not.

    source env.sh
    $PY analyze_swap.py runs/swap45
"""
import glob
import sys
from math import comb

import numpy as np

OWN = {"A": "d_hist", "G": "d_struct", "D": "d_flow"}
MET = ("d_lum", "d_hist", "d_struct", "d_flow", "d_flowdir", "d_toward")


def sign_p(k, n, p=0.5):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def main():
    import json
    root = sys.argv[1] if len(sys.argv) > 1 else "runs/swap45"
    rows = [r for f in glob.glob(f"{root}/swap_*.json")
            for r in json.load(open(f))["rows"]]
    if not rows:
        print(f"no rows under {root}")
        return
    cfg = json.load(open(sorted(glob.glob(f"{root}/swap_*.json"))[0]))["meta"]["config"]
    print(f"[data] {len(rows)} cells   frames={cfg['frames']}  steps={cfg['steps']}")
    f0 = [r["d_lum_f0"] for r in rows if "d_lum_f0" in r]
    if f0:
        print(f"[control] frame 0 luminance change: max {max(f0):.2e} (must be 0)")
        assert max(f0) == 0.0

    for fac in ("A", "G", "D"):
        ce = [r for r in rows if r["factor"] == fac and r["site"] == "all"]
        if not ce:
            continue
        n = len(ce)
        tw = np.array([r["d_toward"] for r in ce])
        k = int((tw > 0).sum())
        own = OWN[fac]
        print(f"\n=== {fac}: 구절 교체 ceiling ({n} 샘플) ===")
        print(f"  toward > 0 : {k}/{n}   p(부호)={sign_p(k, n):.4f}   "
              f"평균 {tw.mean():+.4f}  중앙값 {np.median(tw):+.4f}")
        print(f"  {'sample':16s}{'cos':>6s}{'words':>8s}" +
              "".join(f"{m.replace('d_', ''):>10s}" for m in MET))
        for r in sorted(ce, key=lambda x: -x["d_toward"]):
            print(f"  {r['sample']:16s}{r['phrase_cos']:>6.2f}"
                  f"{str(tuple(r['words'])):>8s}" +
                  "".join(f"{r[m]:>10.4f}" for m in MET))
        print(f"  {'평균':16s}{'':>6s}{'':>8s}" +
              "".join(f"{np.mean([r[m] for r in ce]):>10.4f}" for m in MET))

    # cross-factor specificity: does each factor move its own metric most?
    facs = [f for f in ("A", "G", "D")
            if any(r["factor"] == f and r["site"] == "all" for r in rows)]
    if len(facs) > 1:
        print(f"\n=== 특이성: 각 factor 가 자기 지표를 가장 움직이나 (ceiling 평균) ===")
        print(f"  {'metric':11s}" + "".join(f"{f:>10s}" for f in facs) + f"{'우세':>7s}")
        for m in MET:
            v = [np.mean([r[m] for r in rows
                          if r["factor"] == f and r["site"] == "all"]) for f in facs]
            print(f"  {m:11s}" + "".join(f"{x:>10.4f}" for x in v)
                  + f"{facs[int(np.argmax(v))]:>7s}")
        hits = [f for f in facs
                if OWN[f] in MET and
                np.mean([r[OWN[f]] for r in rows
                         if r["factor"] == f and r["site"] == "all"]) ==
                max(np.mean([r[OWN[f]] for r in rows
                             if r["factor"] == g and r["site"] == "all"])
                    for g in facs)]
        print(f"  자기 지표 1위인 factor: {hits}")

    sites = [s for s in ("0-3", "4-7", "8-11", "12-15", "16-19", "20-23",
                         "24-27") if any(r["site"] == s for r in rows)]
    if sites:
        print(f"\n=== window 별 (ceiling 대비 복원율, 샘플 중앙값) ===")
        for fac in facs:
            print(f"\n  [{fac}]  " + "".join(f"{m.replace('d_', ''):>11s}" for m in MET))
            smp = sorted({r["sample"] for r in rows if r["factor"] == fac})
            for s in sites:
                line = []
                for m in MET:
                    rec = []
                    for sm in smp:
                        a = [r[m] for r in rows if r["sample"] == sm
                             and r["factor"] == fac and r["site"] == s]
                        b = [r[m] for r in rows if r["sample"] == sm
                             and r["factor"] == fac and r["site"] == "all"]
                        if a and b and abs(b[0]) > 1e-12:
                            rec.append(a[0] / b[0])
                    line.append(f"{np.median(rec) * 100:>10.0f}%" if rec
                                else f"{'-':>11s}")
                print(f"  {s:8s}" + "".join(line))


if __name__ == "__main__":
    main()
