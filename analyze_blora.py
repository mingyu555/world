#!/usr/bin/env python3
"""Read out `blora_scan.sh`: does confining LoRA to one four-block window make the
factors separate, and do they conflict there?

Four runs share a window and a seed, so LoRA init, sample order, sigma sequence
and noise draws are identical across them and the only difference is the caption.
That matters: dW's raw direction is ~93% seed-determined (cos 0.074 across seeds
vs 0.741 across factors), so an unpaired dW says almost nothing. The paired
difference does:

    D_f = dW_f - dW_base            the part of the update the factor phrase caused

Three readings, per window and module type:

  size        ||D_f|| / ||dW_base||     how much of the update the phrase drove
  share       ||D_f|| / sum_g ||D_g||   which factor drives this window most
                                        (1/3 = no preference)
  interference cos(D_f, D_g)            whether two factors want the same update.
                                        Near 0 means one window can host separate
                                        experts; near 1 means they fight, which is
                                        the interference the whole project is about.

`attn1` is the built-in null. It takes no text, so its update is driven only by
the loss flowing back from later blocks. If attn1's share pattern matches attn2's,
the "selectivity" is a global gradient effect rather than something the text path
did, and the finding is an artifact.

    source env.sh
    $PY analyze_blora.py runs/blora
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

FACTORS = ("A", "G", "D")
MODULE_ORDER = ("attn2.to_k", "attn2.to_v", "attn2.to_q", "attn2.to_out.0",
                "ff.net.0.proj", "ff.net.2",
                "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0")


def deltas(path):
    """{module path -> dW} for one adapter. peft uses alpha == r here, so the
    scaling factor alpha/r is 1 and dW is exactly B @ A."""
    rec = torch.load(path, weights_only=False)
    sd = rec["state_dict"]
    cfg = rec["config"]
    assert cfg["rank"] == cfg["rank"], cfg
    out = {}
    for k in sd:
        if ".lora_A." not in k:
            continue
        mod = k.split(".lora_A.")[0]
        b = k.replace(".lora_A.", ".lora_B.")
        out[mod] = (sd[b].float() @ sd[k].float())
    return out, rec


def mtype(mod):
    """transformer_blocks.7.attn2.to_v -> attn2.to_v"""
    return mod.split(".", 2)[2]


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/blora")
    files = sorted(root.glob("lora_w*_s*.pt"))
    runs = defaultdict(dict)
    for f in files:
        stem = f.stem[len("lora_"):]                  # w0-3_A_s0
        win, fac, seed = stem.rsplit("_", 2)
        runs[(win, seed)][fac] = f
    order = sorted(runs, key=lambda k: int(k[0].lstrip("w").split("-")[0]))
    print(f"[runs] {len(files)} adapters, {len(order)} (window, seed) groups")
    ready = [k for k in order if set(FACTORS) | {"base"} <= set(runs[k])]
    missing = [(k, sorted((set(FACTORS) | {"base"}) - set(runs[k]))) for k in order
               if k not in ready]
    for k, m in missing:
        print(f"[wait] {k[0]} {k[1]}: missing {m}")
    if not ready:
        return

    rows = []
    for win, seed in ready:
        dw, rec = {}, None
        for fac, f in runs[(win, seed)].items():
            dw[fac], rec = deltas(f)
        loss = {fac: (np.mean([h["loss"] for h in torch.load(
            runs[(win, seed)][fac], weights_only=False)["history"]][-20:]))
            for fac in dw}
        Dm = defaultdict(dict)                       # mtype -> factor -> concat vector
        base_n = defaultdict(float)
        for mod in dw["base"]:
            t = mtype(mod)
            base_n[t] += float(dw["base"][mod].norm()) ** 2
            for fac in FACTORS:
                d = (dw[fac][mod] - dw["base"][mod]).flatten()
                Dm[t].setdefault(fac, []).append(d)
        for t in Dm:
            for fac in FACTORS:
                Dm[t][fac] = torch.cat(Dm[t][fac])
            rows.append({"window": win, "seed": seed, "mtype": t,
                         "base_norm": base_n[t] ** 0.5,
                         "D": {f: Dm[t][f] for f in FACTORS},
                         "loss": loss})

    windows = list(dict.fromkeys(r["window"] for r in rows))
    types = [t for t in MODULE_ORDER if any(r["mtype"] == t for r in rows)]

    def get(win, t):
        for r in rows:
            if r["window"] == win and r["mtype"] == t:
                return r
        return None

    print("\n=== 마지막 20스텝 평균 loss (캡션별) ===")
    print(f"  {'window':8s}" + "".join(f"{k:>9s}" for k in ("base",) + FACTORS))
    for win in windows:
        r = get(win, types[0])
        print(f"  {win:8s}" + "".join(f"{r['loss'][k]:>9.4f}"
                                     for k in ("base",) + FACTORS))

    print("\n=== ||D_f|| / ||dW_base||  (구절이 유발한 업데이트 비중) ===")
    print(f"  {'window':8s}{'mtype':16s}" + "".join(f"{f:>9s}" for f in FACTORS))
    for win in windows:
        for t in types:
            r = get(win, t)
            if r is None:
                continue
            print(f"  {win:8s}{t:16s}" +
                  "".join(f"{float(r['D'][f].norm()) / max(r['base_norm'], 1e-20):>9.4f}"
                          for f in FACTORS))

    print("\n=== share = ||D_f|| / sum_g ||D_g||   (0.333 = 선호 없음) ===")
    print(f"  {'window':8s}{'mtype':16s}" + "".join(f"{f:>9s}" for f in FACTORS)
          + f"{'우세':>8s}{'편차':>8s}")
    for win in windows:
        for t in types:
            r = get(win, t)
            if r is None:
                continue
            n = np.array([float(r["D"][f].norm()) for f in FACTORS])
            sh = n / max(n.sum(), 1e-20)
            i = int(sh.argmax())
            print(f"  {win:8s}{t:16s}" + "".join(f"{x:>9.3f}" for x in sh)
                  + f"{FACTORS[i]:>8s}{sh[i] - 1 / 3:>+8.3f}")

    print("\n=== 간섭 cos(D_f, D_g)  (0 이면 분리 학습 가능, 1 이면 충돌) ===")
    pairs = [("A", "G"), ("A", "D"), ("G", "D")]
    print(f"  {'window':8s}{'mtype':16s}" + "".join(f"{a + '-' + b:>9s}"
                                                    for a, b in pairs))
    for win in windows:
        for t in types:
            r = get(win, t)
            if r is None:
                continue
            cs = []
            for a, b in pairs:
                x, y = r["D"][a], r["D"][b]
                cs.append(float(x.dot(y) / (x.norm() * y.norm()).clamp_min(1e-20)))
            print(f"  {win:8s}{t:16s}" + "".join(f"{c:>9.3f}" for c in cs))

    print("\n=== attn1 null 대조: attn2 와 share 패턴이 같은가 ===")
    print(f"  {'window':8s}{'attn2 우세':>12s}{'attn1 우세':>12s}{'일치':>7s}")
    agree = 0
    for win in windows:
        r2 = get(win, "attn2.to_v") or get(win, "attn2.to_k")
        r1 = get(win, "attn1.to_v") or get(win, "attn1.to_q")
        if r2 is None or r1 is None:
            continue
        w2 = FACTORS[int(np.argmax([float(r2["D"][f].norm()) for f in FACTORS]))]
        w1 = FACTORS[int(np.argmax([float(r1["D"][f].norm()) for f in FACTORS]))]
        agree += w2 == w1
        print(f"  {win:8s}{w2:>12s}{w1:>12s}{'o' if w2 == w1 else 'x':>7s}")
    print(f"  일치 {agree}/{len(windows)} — 모두 일치하면 텍스트 경로가 아니라 "
          f"전역 gradient 효과")

    out = root / "blora_summary.json"
    out.write_text(json.dumps([
        {"window": r["window"], "mtype": r["mtype"], "base_norm": r["base_norm"],
         "loss": r["loss"],
         "Dnorm": {f: float(r["D"][f].norm()) for f in FACTORS},
         "cos": {f"{a}-{b}": float(r["D"][a].dot(r["D"][b]) /
                                   (r["D"][a].norm() * r["D"][b].norm()).clamp_min(1e-20))
                 for a, b in pairs}}
        for r in rows], indent=1))
    print(f"\n[save] {out}")


if __name__ == "__main__":
    main()
