#!/usr/bin/env python3
"""Axis C read-out: where did training actually move the weights?

Reads the LoRA adapters from `lora_train.py` and reports, per (layer, module):

  1. rel        ||dW||_F / ||W||_F                relative weight change
  2. spectrum   singular values of dW             -> stable rank, effective rank
  3. cos        <dW_f, dW_g> / (||dW_f|| ||dW_g||)  factor-pair alignment
                = parameter-space interference, measured after training
  4. seed       corr(layer profile @seed0, @seed1)  is the localisation stable?

`||dW||` alone is a weak signal -- Adam normalises step size, so it partly
reflects the learning rate rather than importance. Read it together with the
spectrum (is the update one direction or many?) and the seed correlation (is it
reproducible at all?). Where the seed correlation is low, nothing else here means
anything.

Everything is exact and cheap because dW = B A has rank r:
    singular values : QR of B, then svd of (R_B A)      -> r x in
    inner products  : tr((B_f^T B_g)(A_g A_f^T))        -> r x r

Finally, the layer profiles are scored against the predictions the cheap axes
made before any training happened (patching + gradient, see README):
    A -> L9-L11,  G -> L20-L24,  D -> L11-L16

    python weight_delta_analysis.py --lora /mnt/ssd1/mingyu_cvpr2027/lora \
        --out ./runs/lora_analysis
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

KEY = re.compile(r"^(.*)\.lora_(A|B)\.default\.weight$")
BLOCK = re.compile(r"transformer_blocks\.(\d+)\.(attn1|attn2|ff|norm1|norm2|norm3)\.(.+)$")
PREDICTION = {"A": (9, 11), "G": (20, 24), "D": (11, 16)}
GROUPS = ("attn1", "attn2", "ff", "norm1", "norm2", "norm3")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="directory with lora_<factor>_s<seed>.pt")
    ap.add_argument("--out", default="./runs/lora_analysis")
    ap.add_argument("--snapshot", default=None,
                    help="model snapshot dir for base ||W|| (default: from HF_HOME)")
    ap.add_argument("--n_layers", type=int, default=28)
    return ap.parse_args()


def find_snapshot(explicit):
    if explicit:
        return Path(explicit)
    import os
    hub = Path(os.environ["HF_HOME"]) / "hub"
    cands = list((hub / "models--nvidia--Cosmos-Predict2-2B-Video2World"
                  / "snapshots").glob("*/transformer"))
    if not cands:
        raise SystemExit("could not find the transformer snapshot; pass --snapshot")
    return cands[0]


def base_norms(snapshot: Path):
    """||W||_F for every transformer weight, read straight from safetensors."""
    from safetensors import safe_open
    out = {}
    for fp in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(fp), framework="pt") as f:
            for k in f.keys():
                if k.endswith(".weight") and BLOCK.search(k):
                    out[k[: -len(".weight")]] = f.get_tensor(k).float().norm().item()
    return out


def load_adapter(fp: Path):
    """-> {module: (A [r,in], B [out,r])}, scaling, config"""
    rec = torch.load(fp, weights_only=False)
    sd = rec["state_dict"]
    parts = defaultdict(dict)
    for k, v in sd.items():
        m = KEY.match(k)
        if m:
            parts[m.group(1)][m.group(2)] = v.float()
    mods = {k: (v["A"], v["B"]) for k, v in parts.items() if "A" in v and "B" in v}
    cfg = rec["config"]
    scaling = cfg["lora_alpha"] / cfg["rank"] if "lora_alpha" in cfg else 1.0
    return mods, scaling, rec


def spectrum(A, B, scaling):
    """Singular values of dW = scaling * B @ A, exactly, at r x in cost."""
    Q, R = torch.linalg.qr(B)                       # B = Q R,  R is [r, r]
    s = torch.linalg.svdvals(R @ A) * scaling
    return s


def main():
    args = parse_args()
    lora_dir, out = Path(args.lora), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(lora_dir.glob("lora_*_s*.pt"))
    if not files:
        raise SystemExit(f"no adapters in {lora_dir}")
    runs = {}
    for fp in files:
        tag = fp.stem[len("lora_"):]
        factor, seed = tag.rsplit("_s", 1)
        runs.setdefault(factor, {})[int(seed)] = fp
    print(f"[found] " + "  ".join(f"{f}:{sorted(s)}" for f, s in runs.items()))

    wn = base_norms(find_snapshot(args.snapshot))
    print(f"[base] ||W|| for {len(wn)} block weights")

    nl = args.n_layers
    # factor -> seed -> group -> per-layer arrays
    rel = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(nl))))
    stable = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    eff = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    keep = {}       # (factor, seed) -> {module: (A, B, scaling)}
    hist = {}

    for factor, seeds in runs.items():
        for seed, fp in sorted(seeds.items()):
            mods, scaling, rec = load_adapter(fp)
            hist[(factor, seed)] = rec.get("history", [])
            keep[(factor, seed)] = {k: (a, b, scaling) for k, (a, b) in mods.items()}
            for name, (A, B) in mods.items():
                m = BLOCK.search(name)
                if not m:
                    continue
                l, grp = int(m.group(1)), m.group(2)
                s = spectrum(A, B, scaling)
                fro = float(torch.sqrt((s ** 2).sum()))
                w = wn.get(name)
                if w:
                    rel[factor][seed][grp][l] += fro / w
                p = (s ** 2 / (s ** 2).sum().clamp_min(1e-20))
                stable[factor][seed][grp].append(float((fro / s.max().clamp_min(1e-20)) ** 2))
                eff[factor][seed][grp].append(
                    float(torch.exp(-(p * p.clamp_min(1e-20).log()).sum())))

    factors = sorted(runs)
    seeds_all = sorted({s for f in runs for s in runs[f]})

    # ---- 0. did the loss move at all? ------------------------------------- #
    print("\n=== training ===")
    print("  The raw first/last-20 comparison is confounded: the loss depends strongly")
    print("  on which sigma was drawn (0.11 at sigma 0.9 vs 0.58 at sigma 0.13). Read")
    print("  the sigma-controlled column: mean over log-sigma quartiles of the relative")
    print("  change from the first half of training to the second (negative = learned).")
    print(f"  {'run':9s}{'steps':>7s}{'first20':>10s}{'last20':>9s}{'sigma-ctrl':>13s}")
    for (f, s), h in sorted(hist.items()):
        if not h:
            continue
        a = np.mean([x["loss"] for x in h[:20]])
        b = np.mean([x["loss"] for x in h[-20:]])
        ls = np.log([x["sigma"] for x in h])
        loss = np.array([x["loss"] for x in h])
        half = len(h) // 2
        edges = np.quantile(ls, [0, .25, .5, .75, 1.0])
        deltas = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (ls >= lo) & (ls <= hi)
            i1, i2 = m.copy(), m.copy()
            i1[half:] = False
            i2[:half] = False
            if i1.sum() >= 3 and i2.sum() >= 3:
                deltas.append((loss[i2].mean() - loss[i1].mean()) / max(loss[i1].mean(), 1e-9))
        d = float(np.mean(deltas)) * 100 if deltas else float("nan")
        print(f"  {f+' s'+str(s):9s}{len(h):>7d}{a:>10.4f}{b:>9.4f}{d:>12.1f}%")

    # ---- 1. seed stability: gate on this ---------------------------------- #
    print("\n=== seed stability ===")
    print("  Which layers move is 93% decided by the optimisation path, not the task:")
    print("  two runs of the SAME factor at different seeds have cos(dW, dW) = 0.07,")
    print("  while a different factor at the same seed keeps 0.74. So the raw profile")
    print("  cannot be seed-stable by construction -- the quantity that has to replicate")
    print("  is the base-subtracted differential dW_f - dW_base at matched seed.")
    stab = {}
    if len(seeds_all) >= 2 and "base" in factors:
        s0, s1 = seeds_all[0], seeds_all[1]
        print(f"\n  {'':6s}" + "".join(f"{g:>10s}" for g in GROUPS)
              + "     <- corr(differential @s{}, @s{})".format(s0, s1))
        for f in [x for x in factors if x != "base"]:
            if not (s0 in rel[f] and s1 in rel[f]
                    and s0 in rel["base"] and s1 in rel["base"]):
                continue
            row = ""
            for grp in GROUPS:
                a = rel[f][s0][grp] - rel["base"][s0][grp]
                b = rel[f][s1][grp] - rel["base"][s1][grp]
                c = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
                stab[f"{f}:{grp}"] = c
                row += f"{c:>+10.2f}"
            print(f"  {f:6s}{row}")
        # the same statistic on the RAW profile, as the null it has to beat
        print(f"\n  {'':6s}" + "".join(f"{g:>10s}" for g in GROUPS)
              + "     <- same, on the RAW profile (the null)")
        for f in [x for x in factors if x != "base"]:
            if not (s0 in rel[f] and s1 in rel[f]):
                continue
            row = ""
            for grp in GROUPS:
                a, b = rel[f][s0][grp], rel[f][s1][grp]
                c = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
                stab[f"raw:{f}:{grp}"] = c
                row += f"{c:>+10.2f}"
            print(f"  {f:6s}{row}")
        print("\n  The differential is real only where it correlates clearly better than")
        print("  the raw profile does. Where it does not, that group is optimiser noise.")
    else:
        print("  need >=2 seeds and the base control -- cannot check yet")

    # mean over seeds for everything that follows
    mrel = {f: {grp: np.mean([rel[f][s][grp] for s in rel[f]], axis=0)
                for grp in GROUPS} for f in factors}

    # ---- 2. where did the weights move? ----------------------------------- #
    print("\n=== ||dW||/||W|| : peak layer per group ===")
    print(f"  {'factor':8s}" + "".join(f"{g:>10s}" for g in GROUPS))
    for f in factors:
        print(f"  {f:8s}" + "".join(f"{int(np.argmax(mrel[f][g])):>10d}" for g in GROUPS))

    print("\n=== group share of the total relative change ===")
    for f in factors:
        tot = {g: float(mrel[f][g].sum()) for g in GROUPS}
        s = sum(tot.values()) + 1e-12
        print(f"  {f:8s} " + "  ".join(f"{g}={tot[g]/s*100:5.1f}%" for g in GROUPS))

    # ---- 2b. paired factor-attributable update -------------------------- #
    # rel_f - rel_base compares magnitudes; what we want is the magnitude of the
    # difference, ||dW_f - dW_base|| / ||dW_base||, i.e. how much of this layer's
    # movement is attributable to the caption. Same seed, so the shared
    # optimisation path cancels exactly.
    paired = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: np.zeros(nl))))
    if "base" in factors:
        for f in [x for x in factors if x != "base"]:
            for seed in sorted(keep_seeds := [s for s in runs[f] if s in runs["base"]]):
                Kf, Kb = keep[(f, seed)], keep[("base", seed)]
                for name, (Af, Bf, sf) in Kf.items():
                    ob = Kb.get(name)
                    m = BLOCK.search(name)
                    if ob is None or not m:
                        continue
                    Ab, Bb, sb = ob
                    dWf = (Bf @ Af) * sf
                    dWb = (Bb @ Ab) * sb
                    nb = float(dWb.norm())
                    if nb > 0:
                        paired[f][seed][m.group(2)][int(m.group(1))] += \
                            float((dWf - dWb).norm()) / nb
        print("\n=== paired factor-attributable update  ||dW_f - dW_base|| / ||dW_base|| ===")
        print(f"  {'factor':8s}" + "".join(f"{g:>10s}" for g in GROUPS) + "   peak layer")
        for f in paired:
            mp = {g: np.mean([paired[f][s][g] for s in paired[f]], axis=0) for g in GROUPS}
            print(f"  {f:8s}" + "".join(f"{int(np.argmax(mp[g])):>10d}" for g in GROUPS))
        print("\n  attn2 only, factor differential vs the other factors, and the")
        print("  pre-registered prediction from the activation + gradient axes:")
        mpa = {f: np.mean([paired[f][s]["attn2"] for s in paired[f]], axis=0)
               for f in paired}
        for f in mpa:
            others = [g for g in mpa if g != f]
            n = {g: mpa[g] / (np.abs(mpa[g]).sum() + 1e-12) for g in mpa}
            adv = n[f] - np.maximum.reduce([n[g] for g in others]) if others else n[f]
            top = np.argsort(-adv)[:5]
            lo, hi = PREDICTION.get(f, (None, None))
            hit = "" if lo is None else \
                f"   predicted L{lo}-L{hi}: {sum(lo <= t <= hi for t in top)}/5"
            print(f"    {f}: " + "  ".join(f"L{t}({adv[t]:+.4f})" for t in top) + hit)
        # seed replication of exactly this statistic
        if len(seeds_all) >= 2:
            print("\n  seed replication of the paired statistic (attn2):")
            for f in paired:
                ss = sorted(paired[f])
                if len(ss) >= 2:
                    a, b = paired[f][ss[0]]["attn2"], paired[f][ss[1]]["attn2"]
                    c = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else float("nan")
                    print(f"    {f}: corr(s{ss[0]}, s{ss[1]}) = {c:+.3f}")

    # ---- 3. factor-differential, the only lens that discriminated before --- #
    print("\n=== factor differential vs the other experts (base subtracted) ===")
    diff = {}
    real = [f for f in factors if f != "base"]
    if len(real) < 2:
        print(f"  (need >=2 non-base experts, have {real}) -- skipped")
    for grp in ("attn2", "norm2", "ff", "attn1") if len(real) >= 2 else ():
        if grp not in GROUPS:
            continue
        norm = {}
        for f in real:
            v = mrel[f][grp].copy()
            if "base" in factors:                  # remove the "any fine-tuning" part
                v = v - mrel["base"][grp]
            norm[f] = v / (np.abs(v).sum() + 1e-12)
        print(f"  [{grp}]")
        for f in real:
            others = [g for g in real if g != f]
            adv = norm[f] - np.maximum.reduce([norm[g] for g in others])
            diff[f"{f}:{grp}"] = adv.tolist()
            top = np.argsort(-adv)[:5]
            lo, hi = PREDICTION.get(f, (None, None))
            inpred = "" if lo is None else \
                f"   predicted L{lo}-L{hi}: {sum(lo <= t <= hi for t in top)}/5 hit"
            print(f"    {f}: " + "  ".join(f"L{t}({adv[t]:+.3f})" for t in top) + inpred)

    # ---- 4. interference, measured after training -------------------------- #
    print("\n=== cos(dW_f, dW_g) per group, after training ===")
    inter = {}
    for grp in GROUPS:
        cells = []
        for i, f in enumerate(real):
            for g in real[i + 1:]:
                num = den_f = den_g = 0.0
                for name, (Af, Bf, sf) in keep.get((f, seeds_all[0]), {}).items():
                    m = BLOCK.search(name)
                    if not m or m.group(2) != grp:
                        continue
                    other = keep.get((g, seeds_all[0]), {}).get(name)
                    if other is None:
                        continue
                    Ag, Bg, sg = other
                    num += float(torch.trace((Bf.T @ Bg) @ (Ag @ Af.T))) * sf * sg
                    den_f += float((spectrum(Af, Bf, sf) ** 2).sum())
                    den_g += float((spectrum(Ag, Bg, sg) ** 2).sum())
                c = num / (np.sqrt(den_f * den_g) + 1e-20)
                inter[f"{grp}:{f}-{g}"] = c
                cells.append(f"{f}-{g}={c:+.3f}")
        if cells:
            print(f"  {grp:6s} " + "  ".join(cells))

    # ---- 5. is the update one direction or many? -------------------------- #
    print("\n=== update spectrum (rank 16 max) ===")
    print(f"  {'factor':8s}" + "".join(f"{g:>12s}" for g in ("attn2", "ff", "norm2")))
    for f in factors:
        row = ""
        for grp in ("attn2", "ff", "norm2"):
            e = np.mean([np.mean(eff[f][s][grp]) for s in eff[f] if eff[f][s][grp]])
            row += f"{e:>12.2f}"
        print(f"  {f:8s}{row}   (effective rank, entropy-based)")

    res = {"seed_stability": stab, "peak_layer":
           {f: {g: int(np.argmax(mrel[f][g])) for g in GROUPS} for f in factors},
           "rel_layer_profile": {f: {g: mrel[f][g].tolist() for g in GROUPS}
                                 for f in factors},
           "factor_differential": diff, "interference": inter,
           "prediction": {k: list(v) for k, v in PREDICTION.items()}}
    (out / "weight_delta.json").write_text(json.dumps(res, indent=1))
    print(f"\n[save] {out/'weight_delta.json'}")


if __name__ == "__main__":
    main()
