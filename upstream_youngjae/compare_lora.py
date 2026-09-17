#!/usr/bin/env python3
"""Does the expert LoRA actually control its factor?  Generate and measure.

Held-out denoising loss cannot answer this -- it separated the layer sets by
0.0003-0.0018 with mixed sign.  What can is generating video and reading the
factor off the pixels.

For one factor, three checkpoints are compared: the pretrained backbone, the
LoRA placed where the analysis said, and the LoRA placed in the control layers.
For every val clip each checkpoint generates two videos from the *same* initial
noise: one with the factor's true clause, one with that clause replaced by its
opposite.  The measured gap between the two is the amount of control:

    control(clip) = metric(true) - metric(counterfactual)

and the direction is known in advance (day is brighter than night, moving has
more image motion than stationary), so

    CSR = fraction of clips where the gap has the requested sign

with the size of the gap as the graded version.  A checkpoint that ignores the
clause scores 0.5 and a gap near zero.

Only factors with a validated judge are testable: A (luminance, night/day
AUC 0.999) and D (sparse-LK speed proxy, r = 0.62 with ego speed).  Geometry has
no working pixel readout yet, so `--factor G` refuses rather than reporting a
number nobody should trust.

  python compare_lora.py --factor D --runs runs_wm/D_analysis runs_wm/D_control \\
      --clips 8 --out ./compare/D
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_same_noise import NEG, build_pipe, latent_shape
from metrics_agd import appearance, dynamics, geometry

# clause swapped in for the counterfactual, and which way the metric must move
COUNTERFACTUAL = {
    "A": {"night": "it is a bright sunny day with clear daylight",
          "day": "it is night, lit only by streetlights"},
    "D": {"moving": "the ego vehicle remains stationary",
          "stationary": "the ego vehicle drives forward quickly down the road"},
}
METRIC = {"A": "luma", "D": "flow_mean"}


def variant_for(factor, rec):
    """(true clause, counterfactual clause, expected sign of true - cf)."""
    ph = rec["phrases"]
    if factor == "A":
        cf = (COUNTERFACTUAL["A"]["night"] if rec["is_night"]
              else COUNTERFACTUAL["A"]["day"])
        # true - cf on luminance: negative if the clip is night, positive if day
        return ph["A"], cf, (-1.0 if rec["is_night"] else +1.0)
    moving = "stationary" not in rec["gt"]["events"]
    cf = (COUNTERFACTUAL["D"]["moving"] if moving
          else COUNTERFACTUAL["D"]["stationary"])
    return ph["D_gt"], cf, (+1.0 if moving else -1.0)


def compose(rec, factor, clause):
    """The AGD caption with one factor's clause replaced."""
    base = rec["captions"]["base"]["text"].rstrip(" .")
    ph = {"A": rec["phrases"]["A"], "G": rec["phrases"].get("G", ""),
          "D": rec["phrases"]["D_gt"]}
    ph[factor] = clause
    return base + "".join(", " + ph[f] for f in ("A", "G", "D") if ph[f]) + "."


def measure(frames, factor):
    m = {}
    m.update(appearance(frames))
    d = dynamics(frames)
    d.pop("profile", None)
    m.update(d)
    m.update(geometry(frames))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True, choices=["A", "G", "D"])
    ap.add_argument("--runs", nargs="*", default=[], help="LoRA dirs to compare with base")
    ap.add_argument("--dataset", default="../wm_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_video", action="store_true", default=True)
    ap.add_argument("--no_guardrail", action="store_true", default=True)
    args = ap.parse_args()

    if args.factor == "G":
        raise SystemExit(
            "G has no validated pixel judge yet (lane centroid vs map lane index: "
            "r = -0.06). Add a lane detector before comparing G checkpoints.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Path(args.dataset)

    # use the checkpoints' own held-out clips, so nothing measured was trained on
    val = None
    for r in args.runs:
        cfg = json.loads((Path(r) / "config.json").read_text())
        val = set(cfg["val_ids"]) if val is None else val & set(cfg["val_ids"])
    ids = sorted(val)[: args.clips] if val else \
        [json.loads(l)["clip_id"] for l in open(ds / "clips.jsonl")][: args.clips]
    print(f"[compare] factor {args.factor}, {len(ids)} held-out clips, "
          f"checkpoints: base + {[Path(r).name for r in args.runs]}")

    pipe = build_pipe(torch.bfloat16, False, False, args.no_guardrail)
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    from diffusers.utils import export_to_video

    def gen(prompt, image, z):
        with torch.no_grad():
            r = pipe(image=image, prompt=prompt, negative_prompt=NEG,
                     height=args.height, width=args.width, num_frames=args.frames,
                     num_inference_steps=args.steps, guidance_scale=args.guidance,
                     fps=args.fps, generator=torch.Generator().manual_seed(args.seed),
                     latents=z.clone().to("cuda", dtype=torch.bfloat16))
        return r.frames[0]

    import numpy as np
    ckpts = [("base", None)] + [(Path(r).name, r) for r in args.runs]
    rows = []
    for ci, (name, path) in enumerate(ckpts):
        if path:
            from peft import PeftModel
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer,
                                                         str(Path(path) / "final"))
            pipe.transformer.eval()
        for i, sid in enumerate(ids):
            rec = json.loads((ds / sid / "caption.json").read_text())
            img = Image.open(ds / sid / "cond_image.jpg").convert("RGB")
            zp = out / f"z_{sid}.pt"
            if zp.exists():
                z = torch.load(zp)
            else:
                z = torch.randn(shape, generator=torch.Generator().manual_seed(
                    args.seed + i))
                torch.save(z, zp)
            true_c, cf_c, sign = variant_for(args.factor, rec)
            row = {"ckpt": name, "clip": sid, "sign": sign,
                   "true_clause": true_c, "cf_clause": cf_c}
            for tag, clause in (("true", true_c), ("cf", cf_c)):
                fp = out / f"{name}__{sid}__{tag}.mp4"
                if fp.exists():
                    from metrics_agd import read_video
                    frames = [np.asarray(f) for f in read_video(fp)]
                else:
                    frames = [np.asarray(f) for f in gen(compose(rec, args.factor, clause),
                                                         img, z)]
                    if args.save_video:
                        export_to_video([Image.fromarray(f) for f in frames],
                                        str(fp), fps=args.fps)
                for k, v in measure(frames, args.factor).items():
                    row[f"{tag}_{k}"] = v
            key = METRIC[args.factor]
            row["gap"] = row[f"true_{key}"] - row[f"cf_{key}"]
            row["correct"] = bool(row["gap"] * sign > 0)
            rows.append(row)
            print(f"  [{name}] {sid}  {key}: true {row[f'true_{key}']:.4f} "
                  f"cf {row[f'cf_{key}']:.4f}  gap {row['gap']:+.4f} "
                  f"{'OK' if row['correct'] else 'wrong'}", flush=True)
            (out / "rows.json").write_text(json.dumps(rows, indent=1))
        if path:
            pipe.transformer = pipe.transformer.unload()

    print(f"\n{'checkpoint':16s} {'CSR':>6s} {'mean |gap|':>11s} {'signed gap':>11s}")
    for name, _ in ckpts:
        rs = [r for r in rows if r["ckpt"] == name]
        if not rs:
            continue
        csr = float(np.mean([r["correct"] for r in rs]))
        sg = float(np.mean([r["gap"] * r["sign"] for r in rs]))
        print(f"{name:16s} {csr:6.2f} {np.mean([abs(r['gap']) for r in rs]):11.4f} "
              f"{sg:+11.4f}")
    (out / "rows.json").write_text(json.dumps(rows, indent=1))
    print(f"\n[written] {out}/rows.json and {len(list(out.glob('*.mp4')))} videos")


if __name__ == "__main__":
    main()
