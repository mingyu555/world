"""Schema + analyzer smoke test: fabricate a probe record, run the analyzer on it.

Checks that what factor_layer_probe.py saves is exactly what analyze_factor_layer.py
reads, and that the placement rule and figures come out. Uses a planted ground
truth (A -> ff of layer 12, G -> attn2 of layer 21, D -> attn1 of layer 24) so the
printed layer map can be eyeballed for correctness.

    python test_analyze_smoke.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

BRANCHES = ["attn1", "attn2", "ff"]
FACTORS = ["A", "G", "D"]
N_LAYERS, GROUP, STEPS = 28, 4, [0, 7, 14, 21, 28]
PLANT = {"A": ("ff", 12), "G": ("attn2", 21), "D": ("attn1", 24)}


def fabricate(out: Path):
    groups = [list(range(i, min(i + GROUP, N_LAYERS))) for i in range(0, N_LAYERS, GROUP)]
    configs, results = {}, {}
    for b in BRANCHES:
        for L in groups:
            tag = f"{b}@L{L[0]}-{L[-1]}" if len(L) > 1 else f"{b}@L{L[0]}"
            configs[tag] = {"layers": L, "branches": [b]}
            results[tag] = {}
            for f in FACTORS:
                pb, pl = PLANT[f]
                hit = (b == pb) and (pl in L)
                rows = []
                for i, s in enumerate(STEPS):
                    r = (0.45 if hit else 0.02) + 0.01 * i
                    rows.append({
                        "restore": r, "mag": r + 0.05, "align": 0.9 if hit else 0.2,
                        "cross": {g: (0.03 if hit else 0.02) for g in FACTORS if g != f},
                        "step": s,
                    })
                results[tag][f] = rows
    # the structural-zero control tap
    configs["attn1@L0"] = {"layers": [0], "branches": ["attn1"]}
    results["attn1@L0"] = {f: [{"restore": 0.0, "mag": 0.0, "align": 0.0,
                                "cross": {g: 0.0 for g in FACTORS if g != f}, "step": s}
                               for s in STEPS] for f in FACTORS}

    rec = {
        "meta": {"sample": "fake", "sample_id": "scene-XXXX_fake", "prompt": "p",
                 "phrases": {f: f for f in FACTORS}, "remove_prompts": {},
                 "factors": FACTORS, "n_layers": N_LAYERS, "group": GROUP,
                 "branch_sets": BRANCHES, "probe_steps": STEPS,
                 "latent_shape": [1, 16, 8, 88, 160], "config": {}},
        "configs": configs,
        "results": results,
        "controls": {"own": [1e-7] * len(STEPS),
                     "ref_norm": {f: [1.0] * len(STEPS) for f in FACTORS},
                     "v_norm": [100.0] * len(STEPS)},
        "write_norm": [],
    }
    torch.save(rec, out / "factor_layer_probe.pt")


def main():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        fabricate(d)
        r = subprocess.run([sys.executable, "analyze_factor_layer.py", str(d)],
                           capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
        pl = json.loads((d / "placement.json").read_text())["placement"]
        ok = True
        for f, (pb, plyr) in PLANT.items():
            top = pl[f][0] if pl[f] else None
            hit = top and top["branches"] == [pb] and plyr in top["layers"]
            print(f"  planted {f} -> {pb}@L{plyr} : recovered "
                  f"{top['tag'] if top else None}  {'OK' if hit else 'MISS'}")
            ok &= bool(hit)
        figs = sorted(p.name for p in d.glob("*.png"))
        print(f"  figures: {figs}")
        ok &= len(figs) == 4
        print("\nALL PASS" if ok else "\nFAILED")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
