#!/usr/bin/env python3
"""Turn factor_layer_probe output into the layer map, the figures, and placement.json.

    python analyze_factor_layer.py ./runs/scene-0014_f16 --tag probe

Writes, next to the .pt:
    analysis.json              every curve, peaks, specificity
    placement.json             the LoRA placement decision (see RULE below)
    fig_restore_heatmap.png    layer x branch restore, one panel per factor
    fig_layer_profile.png      layer profile per factor  (plan section 49 map)
    fig_step_profile.png       timestep profile per factor
    fig_specificity.png        factor x factor cross-restore

RULE (fixed before looking at the data, so that section 44's random-placement
baseline stays a fair comparison):

    keep taps with mean restore > 0 and specificity > --tau,
    take the top --topk per factor, ranked by mean restore.

    specificity(f) = |restore_f| / (sum_{g != f} |restore_g| + eps)      section 28
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

EPS = 1e-8


def load(run_dir, tag):
    fp = Path(run_dir) / f"factor_layer_{tag}.pt"
    if not fp.exists():
        raise SystemExit(f"not found: {fp}")
    return torch.load(fp, weights_only=False), fp


def aggregate(rec):
    """tag -> factor -> mean over probed steps."""
    factors = rec["meta"]["factors"]
    agg = {}
    for tag, per_factor in rec["results"].items():
        agg[tag] = {}
        for f in factors:
            rows = per_factor.get(f, [])
            if not rows:
                continue
            restore = np.array([r["restore"] for r in rows])
            mag = np.array([r["mag"] for r in rows])
            align = np.array([r["align"] for r in rows])
            steps = [r["step"] for r in rows]
            cross = {g: float(np.mean([abs(r["cross"].get(g, 0.0)) for r in rows]))
                     for g in factors if g != f}
            spec = abs(restore.mean()) / (sum(cross.values()) + EPS)
            agg[tag][f] = {
                "restore_mean": float(restore.mean()),
                "restore_per_step": restore.tolist(),
                "mag_mean": float(mag.mean()),
                "align_mean": float(align.mean()),
                "steps": steps,
                "cross_mean_abs": cross,
                "specificity": float(spec),
            }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--tag", default="probe")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args()

    rec, fp = load(args.run_dir, args.tag)
    out = fp.parent
    meta, cfgs = rec["meta"], rec["configs"]
    factors = meta["factors"]
    branch_sets = meta["branch_sets"]
    agg = aggregate(rec)

    # ---- controls ----------------------------------------------------- #
    own = rec["controls"]["own"]
    vn = rec["controls"]["v_norm"]
    null_rel = max(o / (v + EPS) for o, v in zip(own, vn)) if own else float("nan")
    print(f"[control] own-delta re-injection, worst |d|/|v| = {null_rel:.2e}  (0 = clean)")
    l0 = [t for t, c in cfgs.items()
          if c["layers"] == [0] and c["branches"] == ["attn1"]]
    if l0:
        v = agg[l0[0]]
        print("[control] L0/attn1 restore (structural zero): "
              + "  ".join(f"{f}={v[f]['restore_mean']:+.4f}" for f in factors if f in v))

    # ---- layer profile ------------------------------------------------ #
    n_layers = meta["n_layers"]
    profile = {f: np.zeros(n_layers) for f in factors}     # summed over branches
    per_branch = {f: {b: np.zeros(n_layers) for b in branch_sets} for f in factors}
    for tag, c in cfgs.items():
        for f in factors:
            if f not in agg.get(tag, {}):
                continue
            r = agg[tag][f]["restore_mean"]
            bname = tag.split("@")[0]
            for layer in c["layers"]:
                # a group's score is a property of the group; spread it so the
                # per-layer curve integrates to the group total
                profile[f][layer] += r / len(c["layers"])
                per_branch[f][bname][layer] += r / len(c["layers"])

    # additivity: patching every tap reproduces the reference stream exactly, so
    # the single-tap restores should sum to ~1. A sum far from 1 means the taps
    # interact (happens when --group patches many layers at once) and the scores
    # can no longer be read as fractions.
    print("\n=== additivity check (sum of restore over all taps; ~1.0 = readable as fractions) ===")
    for f in factors:
        tot = sum(agg[t][f]["restore_mean"] for t in agg if f in agg[t])
        flag = "ok" if 0.7 < tot < 1.4 else "INTERACTING -- reduce --group"
        print(f"  {f}: {tot:+.3f}   {flag}")

    print("\n=== layer profile (restore, summed over branches) ===")
    for f in factors:
        pk = int(np.argmax(profile[f]))
        tot = float(profile[f].sum())
        print(f"  {f}: peak layer {pk:2d}   total {tot:+.3f}   "
              f"top5 layers {np.argsort(-profile[f])[:5].tolist()}")

    print("\n=== branch share (sum of restore over layers) ===")
    for f in factors:
        parts = {b: float(per_branch[f][b].sum()) for b in branch_sets}
        s = sum(abs(v) for v in parts.values()) + EPS
        print(f"  {f}: " + "  ".join(f"{b}={parts[b]:+.3f}({abs(parts[b])/s*100:4.1f}%)"
                                     for b in branch_sets))

    # ---- placement decision ------------------------------------------- #
    placement = {}
    for f in factors:
        cand = []
        for tag in cfgs:
            v = agg.get(tag, {}).get(f)
            if v is None or v["restore_mean"] <= 0 or v["specificity"] <= args.tau:
                continue
            cand.append({"tag": tag, "layers": cfgs[tag]["layers"],
                         "branches": cfgs[tag]["branches"],
                         "restore": v["restore_mean"],
                         "specificity": v["specificity"]})
        cand.sort(key=lambda d: -d["restore"])
        placement[f] = cand[: args.topk]

    print(f"\n=== placement (rule: restore>0, specificity>{args.tau}, top{args.topk}) ===")
    for f in factors:
        if not placement[f]:
            print(f"  {f}: (none passed -- factor not localised at this resolution)")
        for c in placement[f]:
            print(f"  {f}: {c['tag']:16s} layers={c['layers']}  "
                  f"restore={c['restore']:+.4f}  spec={c['specificity']:.2f}")

    analysis = {
        "meta": meta, "configs": cfgs, "agg": agg,
        "layer_profile": {f: profile[f].tolist() for f in factors},
        "branch_layer_profile": {f: {b: per_branch[f][b].tolist() for b in branch_sets}
                                 for f in factors},
        "controls": {"own_rel_max": null_rel,
                     "ref_norm": rec["controls"]["ref_norm"]},
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))
    (out / "placement.json").write_text(json.dumps(
        {"rule": {"topk": args.topk, "tau": args.tau,
                  "metric": "mean restore, specificity filtered"},
         "sample": meta["sample_id"], "placement": placement}, indent=1))
    print(f"\n[save] {out/'analysis.json'}\n[save] {out/'placement.json'}")

    if not args.no_fig:
        make_figures(out, meta, cfgs, agg, profile, per_branch, factors, branch_sets)


def make_figures(out, meta, cfgs, agg, profile, per_branch, factors, branch_sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = meta["n_layers"]
    colors = {"A": "#d95f02", "G": "#1b9e77", "D": "#7570b3"}

    # layer x branch heatmap, one panel per factor
    fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 3.2), squeeze=False)
    for ax, f in zip(axes[0], factors):
        M = np.stack([per_branch[f][b] for b in branch_sets])
        v = np.abs(M).max() or 1.0
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        ax.set_yticks(range(len(branch_sets)), branch_sets)
        ax.set_xlabel("layer")
        ax.set_title(f"+{f} restore")
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(f"{meta['sample_id']} — where each factor's effect flows")
    fig.tight_layout()
    fig.savefig(out / "fig_restore_heatmap.png", dpi=140)
    plt.close(fig)

    # layer profile (plan section 49)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for f in factors:
        ax.plot(range(n_layers), profile[f], marker="o", ms=3,
                color=colors.get(f), label=f"+{f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("restore (fraction of factor effect)")
    ax.set_title(f"{meta['sample_id']} — factor x layer")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig_layer_profile.png", dpi=140)
    plt.close(fig)

    # timestep profile: restore of the best tap per factor across probed steps
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for f in factors:
        best = max((t for t in agg if f in agg[t]),
                   key=lambda t: agg[t][f]["restore_mean"], default=None)
        if best is None:
            continue
        v = agg[best][f]
        ax.plot(v["steps"], v["restore_per_step"], marker="o", ms=3,
                color=colors.get(f), label=f"+{f} @ {best}")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("restore")
    ax.set_title("best tap per factor, over timesteps")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_step_profile.png", dpi=140)
    plt.close(fig)

    # specificity: mean |cross restore| of each factor's best tap
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    M = np.zeros((len(factors), len(factors)))
    for i, f in enumerate(factors):
        best = max((t for t in agg if f in agg[t]),
                   key=lambda t: agg[t][f]["restore_mean"], default=None)
        if best is None:
            continue
        M[i, i] = agg[best][f]["restore_mean"]
        for j, g in enumerate(factors):
            if g != f:
                M[i, j] = agg[best][f]["cross_mean_abs"].get(g, 0.0)
    v = np.abs(M).max() or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(len(factors)), factors)
    ax.set_yticks(range(len(factors)), [f"{f} tap" for f in factors])
    for i in range(len(factors)):
        for j in range(len(factors)):
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("specificity of the best tap")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(out / "fig_specificity.png", dpi=140)
    plt.close(fig)
    print("[save] fig_restore_heatmap.png  fig_layer_profile.png  "
          "fig_step_profile.png  fig_specificity.png")


if __name__ == "__main__":
    main()
