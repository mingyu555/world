#!/usr/bin/env python3
"""Read out `kv_split_probe.py`: per layer, does the factor's effect on the model
output arrive through `attn2.to_k` or `attn2.to_v`?

`restore` is a projection of the injected difference onto the factor's full
difference, so k-only and v-only shares are comparable and roughly additive at
one site. Three things are worth separating:

  level       the raw share, which the filler group also has -- length,
              punctuation and token count all move the output
  vs filler   the factor-attributable part: mean(restore | factor) - mean(restore | filler)
  separation  whether the four prompts *within* a factor agree, which is what a
              per-factor expert would need

The permutation test is the same one the activation probes use: shuffle the
group labels, recompute the between/within F on the restore values, and count
how often the shuffled F beats the observed one. BH-FDR over all
(layer, mode) cells.

    source env.sh
    $PY analyze_kv.py runs/kvf_lo/kv_lo.pt runs/kvf_hi/kv_hi.pt
"""

import sys
from pathlib import Path

import numpy as np
import torch

FACTORS = ("A", "G", "D")
RNG = np.random.default_rng(0)


def f_ratio(V, labels, groups):
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
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return
    res, grp, modes, steps = {}, {}, None, None
    for p in paths:
        r = torch.load(p, weights_only=False)
        m = r["meta"]
        grp.update(m["groups"])
        modes = modes or tuple(m["modes"])
        steps = steps or m["probe_steps"]
        null = max(r["null"])
        print(f"[{p.name}] {len(r['restore'])} sites, modes {modes}, "
              f"steps {steps}, control null={null:.2e}")
        assert null == 0.0, f"{p}: base-into-both moved the output by {null}"
        for site, d in r["restore"].items():
            res.setdefault(site, {}).update(d)

    words = [w for w in grp]
    labels = np.array([grp[w] for w in words])
    sites = sorted(res, key=lambda s: (len(s), s))
    allg = list(dict.fromkeys(labels))
    print(f"[groups] " + "  ".join(f"{g}={sum(labels == g)}" for g in allg))

    # cells: (site, mode) -> per-word mean restore over steps
    cells, names = [], []
    for site in sites:
        for mode in modes:
            if mode not in res[site]:
                continue
            row = [float(np.mean(res[site][mode][w])) for w in words]
            cells.append(row)
            names.append((site, mode))
    V = np.array(cells)
    # A site covering every layer, injected into both projections, reproduces the
    # factor's full effect by construction: restore == 1 for every word, zero
    # variance. That is the probe's identity check, not a finding, so it is
    # reported separately and kept out of the multiple-comparison set.
    ident = [i for i in range(len(V)) if V[i].std() < 1e-9]
    for i in ident:
        print(f"[identity] {names[i][0]} {names[i][1]}: restore = "
              f"{V[i].mean():.6f} +- {V[i].std():.1e} for all {len(words)} words")
    keep = np.array([i for i in range(len(V)) if i not in ident])
    f = np.full(len(V), np.nan)
    q = np.ones(len(V))
    if len(keep):
        fk, pk = perm_p(V[keep], labels, allg)
        qk = bh(pk)
        f[keep], q[keep] = fk, qk

    fil = labels == "filler"
    print(f"\n=== 레이어 x projection: restore (filler 대비) ===")
    hdr = "".join(f"{m:>22s}" for m in modes)
    print(f"  {'site':8s}{hdr}{'F':>8s}{'q':>7s}")
    idx = {n: i for i, n in enumerate(names)}
    for site in sites:
        cols = []
        Fq = ""
        for mode in modes:
            if (site, mode) not in idx:
                cols.append(" " * 22)
                continue
            i = idx[(site, mode)]
            r = V[i]
            base = r[fil].mean()
            best = max(FACTORS, key=lambda g: r[labels == g].mean() - base)
            cols.append(f"{r.mean():6.3f} |f {base:5.2f}| {best}{r[labels == best].mean() - base:+.3f}"
                        .rjust(22))
            Fq = f"{f[i]:>8.2f}{q[i]:>7.3f}"
        print(f"  {site:8s}" + "".join(cols) + Fq)

    print(f"\n=== factor 별: k_only vs v_only, 사이트별 (filler 뺀 값) ===")
    print(f"  {'site':8s}" + "".join(f"{g + '/' + m[0]:>10s}"
                                     for g in FACTORS for m in modes))
    for site in sites:
        row = []
        for g in FACTORS:
            for mode in modes:
                if (site, mode) not in idx:
                    row.append(f"{'-':>10s}")
                    continue
                r = V[idx[(site, mode)]]
                row.append(f"{r[labels == g].mean() - r[fil].mean():>10.3f}")
        print(f"  {site:8s}" + "".join(row))

    sig = [(q[i], names[i], f[i]) for i in range(len(names))
           if i not in ident and q[i] < 0.10]
    print(f"\n=== q<0.10 통과 셀: {len(sig)}/{len(keep)} (항등 {len(ident)}개 제외) ===")
    for qq, (site, mode), ff in sorted(sig):
        r = V[idx[(site, mode)]]
        mar = {g: r[labels == g].mean() - r[fil].mean() for g in FACTORS}
        win = max(mar, key=mar.get)
        print(f"  {site:8s} {mode:8s} F={ff:6.2f} q={qq:.3f}  "
              f"우세={win}({mar[win]:+.3f})  " +
              "  ".join(f"{g}{mar[g]:+.3f}" for g in FACTORS))

    print(f"\n=== 사이트 무관 요약: 어느 projection 이 더 많이 복원하는가 ===")
    for mode in modes:
        vals = [V[idx[(s, mode)]].mean() for s in sites if (s, mode) in idx]
        print(f"  {mode:8s} 평균 restore={np.mean(vals):.3f}   "
              f"최대 {max(vals):.3f} @ {sites[int(np.argmax(vals))]}")
    if len(modes) == 2:
        a, b = modes
        d = [V[idx[(s, a)]].mean() - V[idx[(s, b)]].mean()
             for s in sites if (s, a) in idx and (s, b) in idx]
        print(f"  {a} > {b} 인 사이트: {sum(x > 0 for x in d)}/{len(d)}   "
              f"평균 차 {np.mean(d):+.3f}")


if __name__ == "__main__":
    main()
