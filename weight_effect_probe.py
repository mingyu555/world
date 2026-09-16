#!/usr/bin/env python3
"""What does the trained weight change actually *do*, per layer and per timestep?

The adapters from `lora_train.py` are one tensor each -- training draws a random
sigma every step, so dW itself carries no timestep axis. But dW's *effect* does:
the same dW acts on different activations at different points in denoising.

    effect(f, l, b, t) = || dW_f x_t || / || W x_t ||

x_t is what actually enters that Linear at step t, and ||W x_t|| is the module's
own output there, so the ratio says how much this trained change would perturb
that module at that timestep. Subtracting the base-caption expert leaves the part
attributable to the factor.

This is the functional statistic the ||dW|| analysis was missing. ||dW|| counts a
change even when it points somewhere the activations never go; this does not.
And unlike ||dW_f - dW_base|| / ||dW_base||, the denominator here is the frozen
model's own output rather than another adapter's norm, which varied 540x across
layers and was what made that statistic an artifact.

One frozen trajectory. Every adapter is evaluated against the same activations,
inside the hook, so nothing large is stored.

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=1 $PY weight_effect_probe.py \\
      --lora /mnt/ssd1/mingyu_cvpr2027/lora --seed 0 --out ./runs/weffect
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

from cosmos_common import NEG, build_pipe, latent_shape

KEY = re.compile(r"^(.*)\.lora_(A|B)\.default\.weight$")
BLOCK = re.compile(r"transformer_blocks\.(\d+)\.(attn1|attn2|ff|norm1|norm2|norm3)\.(.+)$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", default="./runs/weffect")
    ap.add_argument("--seed", type=int, default=0, help="which trained seed to read")
    ap.add_argument("--factors", default="A,G,D,base")
    ap.add_argument("--sample", default=None, help="agd_dataset sample for the prompt/image")
    ap.add_argument("--text_field", default="text_gtD")
    ap.add_argument("--cond", default="AGD", help="which caption to run the trajectory with")
    ap.add_argument("--step_every", type=int, default=4)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--noise_seed", type=int, default=0)
    ap.add_argument("--tag", default="weffect")
    ap.add_argument("--pairwise", action="store_true",
                    help="also measure ||(dW_i - dW_j) x_t|| / ||W x_t|| for every pair. "
                         "dW_i - dW_j = [B_i, -B_j] @ [A_i; A_j] is rank 2r, so this is "
                         "one product per pair inside the hook.")
    return ap.parse_args()


def load_adapter(fp):
    sd = torch.load(fp, weights_only=False)
    parts = defaultdict(dict)
    for k, v in sd["state_dict"].items():
        m = KEY.match(k)
        if m:
            parts[m.group(1)][m.group(2)] = v
    cfg = sd["config"]
    scale = cfg.get("lora_alpha", cfg["rank"]) / cfg["rank"]
    return {k: (v["A"], v["B"], scale) for k, v in parts.items()
            if "A" in v and "B" in v}


def main():
    args = parse_args()
    import os
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = Path(args.sample or (Path(os.environ["AGD"]) / "scene-0626_f24"))
    meta = json.loads((sample / "caption.json").read_text())
    prompt = meta["captions"][args.cond][args.text_field]
    image = Image.open(sample / "cond_image.jpg").convert("RGB")
    factors = [f for f in args.factors.split(",") if f]

    lora_dir = Path(args.lora)
    ad = {}
    for f in factors:
        fp = lora_dir / f"lora_{f}_s{args.seed}.pt"
        if not fp.exists():
            raise SystemExit(f"missing {fp}")
        ad[f] = load_adapter(fp)
    print(f"[lora] seed {args.seed}, {len(factors)} experts, "
          f"{len(ad[factors[0]])} modules each")
    print(f"[traj] {sample.name} / {args.cond}: {prompt}")

    pipe = build_pipe()
    tfm = pipe.transformer
    device = torch.device("cuda")
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    print(f"[model] latent {tuple(shape)}")

    # move the adapters onto the device once, in the transformer's dtype
    dt = next(tfm.parameters()).dtype
    for f in factors:
        ad[f] = {k: (A.to(device, dt), B.to(device, dt), s)
                 for k, (A, B, s) in ad[f].items()}

    probe_steps = list(range(0, args.steps, args.step_every))
    state = {"step": 0, "call": 0, "on": False}
    # (module) -> factor -> per probed step
    eff = defaultdict(lambda: defaultdict(list))
    wnorm = defaultdict(list)
    handles = []

    pairs = [(factors[i], factors[j]) for i in range(len(factors))
             for j in range(i + 1, len(factors))] if args.pairwise else []
    if pairs:
        print(f"[pairs] {len(pairs)} pairwise differences")

    def make_hook(name):
        mods = {f: ad[f][name] for f in factors if name in ad[f]}
        cat = {}
        for f, g in pairs:
            if f in mods and g in mods:
                Af, Bf, sf = mods[f]
                Ag, Bg, sg = mods[g]
                cat[(f, g)] = (torch.cat([Af, Ag], 0),
                               torch.cat([Bf * sf, -Bg * sg], 1))

        def hook(module, inp, output):
            if not state["on"] or state["call"] != 0:
                return
            x = inp[0].detach()
            y = output.detach() if torch.is_tensor(output) else output[0].detach()
            wn = float(y.float().norm())
            wnorm[name].append(wn)
            for f, (A, B, sc) in mods.items():
                # dW x = scale * B (A x), rank 16 so this is cheap
                d = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, A), B) * sc
                eff[name][f].append(float(d.float().norm()) / max(wn, 1e-12))
                del d
            for (f, g), (Ac, Bc) in cat.items():
                d = torch.nn.functional.linear(
                    torch.nn.functional.linear(x, Ac), Bc)
                eff[name][f"{f}-{g}"].append(float(d.float().norm()) / max(wn, 1e-12))
                del d
        return hook

    named = dict(tfm.named_modules())
    for name in ad[factors[0]]:
        m = named.get(name)
        if m is None:
            print(f"[warn] module not found: {name}")
            continue
        handles.append(m.register_forward_hook(make_hook(name)))
    print(f"[hooks] {len(handles)} modules instrumented")

    g = torch.Generator(device="cpu").manual_seed(args.noise_seed)
    z0 = torch.randn(shape, generator=g, dtype=torch.float32)

    orig = tfm.forward

    def fwd(*a, **kw):
        state["on"] = state["call"] == 0 and state["step"] in probe_steps
        t0 = time.time()
        v = orig(*a, **kw)
        if state["on"]:
            print(f"  [weffect] step {state['step']:3d} in {time.time() - t0:.1f}s",
                  flush=True)
        state["on"] = False
        state["call"] += 1
        return v

    tfm.forward = fwd

    def cb(p, i, t, kwargs):
        state["step"] += 1
        state["call"] = 0
        return kwargs

    t0 = time.time()
    with torch.no_grad():
        pipe(image=image, prompt=prompt, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps, generator=torch.Generator(device="cpu").manual_seed(args.noise_seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    tfm.forward = orig
    for h in handles:
        h.remove()

    rec = {
        "meta": {"lora": str(lora_dir), "seed": args.seed, "factors": factors,
                 "sample": str(sample), "cond": args.cond, "prompt": prompt,
                 "probe_steps": probe_steps, "n_layers": len(tfm.transformer_blocks),
                 "pairs": [f"{f}-{g}" for f, g in pairs],
                 "config": {k: v for k, v in vars(args).items()}},
        "effect": {k: dict(v) for k, v in eff.items()},
        "wnorm": dict(wnorm),
    }
    fp = out / f"weffect_s{args.seed}_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
