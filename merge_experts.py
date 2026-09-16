#!/usr/bin/env python3
"""Merge LoRA experts that live on disjoint layers into one adapter.

Two LoRA adapters that target different `transformer_blocks.<l>` never touch the
same weight, so applying both is the same function as applying a single adapter
whose state dict is their union -- exactly, not approximately. No weighting, no
SVD, no interference term. That is the practical payoff of placing experts on
disjoint layers, and it is what `add_weighted_adapter` has to approximate when
the supports overlap.

The script refuses to merge overlapping supports rather than silently averaging
them, because for an overlapping pair the union is *not* equivalent and the
result would be a different model than "both experts active".

    source env.sh
    $PY merge_experts.py runs_train/mg_A runs_train/mg_D --out runs_train/mg_AD
"""
import argparse
import json
import re
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file


def layers_of(cfg):
    out = set()
    for m in cfg["target_modules"]:
        g = re.search(r"transformer_blocks\.(\d+)\.", m)
        if g:
            out.add(int(g.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs, each with final/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_overlap", action="store_true",
                    help="merge anyway; the union is then NOT equivalent to "
                         "running both adapters and the result is a new model")
    args = ap.parse_args()

    finals = [Path(r) / "final" for r in args.runs]
    cfgs = [json.loads((f / "adapter_config.json").read_text()) for f in finals]
    sds = [load_file(f / "adapter_model.safetensors") for f in finals]

    for i, (r, c) in enumerate(zip(args.runs, cfgs)):
        print(f"  [{Path(r).name}] r={c['r']} alpha={c['lora_alpha']} "
              f"layers={sorted(layers_of(c))}  {len(sds[i])} tensors")

    # every adapter must share the scaling, or a union changes each one's effect
    assert len({c["r"] for c in cfgs}) == 1 and len({c["lora_alpha"] for c in cfgs}) == 1, \
        "r / alpha differ between adapters; a union would rescale them"

    sets = [layers_of(c) for c in cfgs]
    overlap = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap |= sets[i] & sets[j]
    if overlap:
        msg = (f"supports overlap on layers {sorted(overlap)}: the union is not "
               f"equivalent to running both adapters")
        if not args.allow_overlap:
            raise SystemExit("[refused] " + msg)
        print("[warn] " + msg)

    merged, keys = {}, set()
    for sd in sds:
        for k, v in sd.items():
            assert k not in keys, f"duplicate tensor {k}"
            keys.add(k)
            merged[k] = v

    cfg = dict(cfgs[0])
    mods = []
    for c in cfgs:
        mods += list(c["target_modules"])
    cfg["target_modules"] = sorted(set(mods))

    out = Path(args.out) / "final"
    out.mkdir(parents=True, exist_ok=True)
    save_file(merged, out / "adapter_model.safetensors")
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=1))
    # compare_lora / eval_fidelity read val_ids from the run's config.json
    src = json.loads((Path(args.runs[0]) / "config.json").read_text())
    src["merged_from"] = [str(r) for r in args.runs]
    src["layers"] = sorted(set().union(*sets))
    src["n_trainable"] = sum(json.loads((Path(r) / "config.json").read_text())
                             ["n_trainable"] for r in args.runs)
    (Path(args.out) / "config.json").write_text(json.dumps(src, indent=1))
    print(f"[merged] {len(merged)} tensors, layers {sorted(set().union(*sets))}, "
          f"{len(cfg['target_modules'])} target modules -> {out}")


if __name__ == "__main__":
    main()
