#!/usr/bin/env python3
"""Do the axes agree on where each factor lives?  (plan section 22)

Puts every available measurement on the same (factor x layer) grid and reports
Spearman rank correlations between them:

    restore     axis A, causal      factor_layer_probe   -> analysis.json
    local_div   local divergence    local_div_probe      -> localdiv_*.pt
    write_f     cross-attn read-in  cosmos_exp/analyze_factor_attention
                                                         -> factor_write.json
    leverage    axis B, gradient    grad_probe           -> grad_*.pt

If they agree, the localisation is trustworthy and the cheap measures can stand
in for the expensive one. If they disagree -- e.g. the text is read in early but
causal effect sits late -- that disagreement is itself the evidence for a
sub-expert structure (plan H3), so print it either way.

    python compare_axes.py ./runs/scene-0626_f24
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

BRANCHES = ("attn1", "attn2", "ff")


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a[ok])).astype(float)
    rb = np.argsort(np.argsort(b[ok])).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def load_restore(d, factors, n_layers):
    fp = d / "analysis.json"
    if not fp.exists():
        return None, None
    a = json.loads(fp.read_text())
    per_layer = {f: np.array(a["layer_profile"][f]) for f in factors if f in a["layer_profile"]}
    per_branch = {f: {b: np.array(v) for b, v in a["branch_layer_profile"][f].items()}
                  for f in factors if f in a.get("branch_layer_profile", {})}
    return per_layer, per_branch


def load_localdiv(d, factors, n_layers, tag="localdiv"):
    fps = sorted(d.glob(f"localdiv_*.pt"))
    if not fps:
        return None, None
    rec = torch.load(fps[0], weights_only=False)
    per_branch = {f: {b: np.zeros(n_layers) for b in BRANCHES} for f in factors}
    for key, per_f in rec["local_div"].items():
        l, b = key.split("|")
        for f, vals in per_f.items():
            if f in per_branch and b in per_branch[f]:
                per_branch[f][b][int(l)] = float(np.mean(vals))
    per_layer = {f: sum(per_branch[f][b] for b in BRANCHES) for f in factors}
    return per_layer, per_branch


def load_write(d, factors, n_layers):
    """cosmos_exp/analyze_factor_attention output (attn2 only)."""
    fp = d / "factor_write.json"
    if not fp.exists():
        return None, None
    w = json.loads(fp.read_text())
    per_layer, per_branch = {}, {}
    for f in factors:
        if f in w.get("factors", {}):
            v = np.array(w["factors"][f]["layer_profile"], float)
            per_layer[f] = v
            per_branch[f] = {"attn2": v}
    return per_layer, per_branch


def load_grad(d, factors, n_layers):
    fps = sorted(d.glob("grad_*.pt"))
    if not fps:
        return None, None
    rec = torch.load(fps[0], weights_only=False)
    wn = rec["w_norm"]
    per_branch = {f: {b: np.zeros(n_layers) for b in BRANCHES} for f in factors}
    for key, per_f in rec["grad"].items():
        l, grp, name = key.split("|", 2)
        b = grp if grp in BRANCHES else None       # norm1/2/3 folded in below
        for f, vals in per_f.items():
            if f not in per_branch:
                continue
            # leverage = ||g|| * ||W||  (effect of a relative weight change)
            lev = float(np.mean(vals)) * wn.get(name, 0.0)
            if b is not None:
                per_branch[f][b][int(l)] += lev
            else:                                   # adaLN gates -> their branch
                tgt = {"norm1": "attn1", "norm2": "attn2", "norm3": "ff"}.get(grp)
                if tgt:
                    per_branch[f][tgt][int(l)] += lev
    per_layer = {f: sum(per_branch[f][b] for b in BRANCHES) for f in factors}
    return per_layer, per_branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--factors", default="A,G,D")
    ap.add_argument("--n_layers", type=int, default=28)
    args = ap.parse_args()
    d = Path(args.run_dir)
    factors = args.factors.split(",")
    nl = args.n_layers

    loaders = {
        "restore": load_restore, "local_div": load_localdiv,
        "write_f": load_write, "leverage": load_grad,
    }
    layer, branch = {}, {}
    for name, fn in loaders.items():
        pl, pb = fn(d, factors, nl)
        if pl:
            layer[name], branch[name] = pl, pb
            print(f"[have] {name}")
        else:
            print(f"[miss] {name}")
    # Derived axis: patching a deep layer injects a delta computed on an already
    # diverged upstream stream, so `restore` is biased toward depth. Dividing by
    # the local divergence asks instead: per unit of local perturbation, how much
    # of the factor's effect does this location explain?
    if "restore" in branch and "local_div" in branch:
        deb = {}
        for f in factors:
            if f not in branch["restore"] or f not in branch["local_div"]:
                continue
            deb[f] = {}
            for b in BRANCHES:
                r = branch["restore"][f].get(b)
                dv = branch["local_div"][f].get(b)
                if r is None or dv is None:
                    continue
                dd = np.where(np.abs(dv) > 1e-9, dv, np.nan)
                deb[f][b] = r / dd
        if deb:
            branch["restore/div"] = deb
            layer["restore/div"] = {f: np.nansum([deb[f][b] for b in deb[f]], axis=0)
                                    for f in deb}
            print("[derived] restore/div  (depth-bias-corrected)")

    if len(layer) < 2:
        raise SystemExit("need at least two axes to compare")

    fps = sorted(d.glob("localdiv_*.pt"))
    if fps:
        rec = torch.load(fps[0], weights_only=False)
        gram = rec.get("ref_gram", {})
        off = {k: float(np.mean(v)) for k, v in gram.items()
               if k.split("-")[0] != k.split("-")[1]}
        if off:
            print("\n=== reference-direction Gram (cos between the factor-removal "
                  "directions) ===")
            for k, v in off.items():
                print(f"  cos(d_ref_{k.replace('-', ', d_ref_')}) = {v:+.3f}")
            worst = max(off.values())
            print(f"  -> factor separability ceiling: {1 - worst:.3f} "
                  f"(1.0 = orthogonal contrasts, 0.0 = the same direction)")

    print(f"\n=== peak layer per axis ===\n{'axis':12s}" +
          "".join(f"{f:>10s}" for f in factors))
    for name, pl in layer.items():
        row = "".join(f"{int(np.argmax(pl[f])):>10d}" if f in pl else f"{'-':>10s}"
                      for f in factors)
        print(f"{name:12s}{row}")

    print(f"\n=== Spearman rank correlation over the 28 layers ===")
    names = list(layer)
    out = {}
    for f in factors:
        print(f"\n  factor {f}")
        print("      " + "".join(f"{n[:9]:>10s}" for n in names))
        for i, a in enumerate(names):
            cells = []
            for b in names:
                if f in layer[a] and f in layer[b]:
                    r = spearman(layer[a][f], layer[b][f])
                    out[f"{f}:{a}~{b}"] = r
                    cells.append(f"{r:>10.2f}")
                else:
                    cells.append(f"{'-':>10s}")
            print(f"  {a[:4]:4s}" + "".join(cells))

    # branch-resolved agreement, where two axes both have branches
    both = [n for n in names if branch[n] and len(branch[n].get(factors[0], {})) > 1]
    if len(both) >= 2:
        print(f"\n=== per-branch rank correlation ({' vs '.join(both[:2])}) ===")
        a, b = both[0], both[1]
        for f in factors:
            cells = []
            for br in BRANCHES:
                va = branch[a].get(f, {}).get(br)
                vb = branch[b].get(f, {}).get(br)
                cells.append(f"{br}={spearman(va, vb):+.2f}" if va is not None
                             and vb is not None else f"{br}=-")
            print(f"  {f}: " + "   ".join(cells))

    # which branch carries each factor, per axis
    print(f"\n=== branch share per axis (fraction of |value| summed over layers) ===")
    for name in names:
        if not branch[name] or len(branch[name].get(factors[0], {})) < 2:
            continue
        print(f"  {name}")
        for f in factors:
            parts = {br: float(np.abs(branch[name][f][br]).sum())
                     for br in BRANCHES if br in branch[name][f]}
            s = sum(parts.values()) + 1e-12
            print(f"    {f}: " + "  ".join(f"{br}={v/s*100:5.1f}%"
                                           for br, v in parts.items()))

    (d / "axes_agreement.json").write_text(json.dumps(
        {"spearman": out,
         "peak_layer": {n: {f: int(np.argmax(layer[n][f])) for f in layer[n]}
                        for n in names},
         "layer_profile": {n: {f: layer[n][f].tolist() for f in layer[n]}
                           for n in names}}, indent=1))
    print(f"\n[save] {d/'axes_agreement.json'}")


if __name__ == "__main__":
    main()
