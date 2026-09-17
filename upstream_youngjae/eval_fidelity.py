#!/usr/bin/env python3
"""Judge checkpoints against the real clip, not against a bare sign test.

The earlier comparison only asked whether a generation moved the right way when
a clause was flipped. That has no scale: it cannot say whether a checkpoint is
close to reality or merely on the correct side of it, and it hid the fact that
the base model's readings came from corrupted video.

Here the real clip is the anchor. For every held-out clip each checkpoint
generates from the true AGD caption, and the same judge is run on the real
frames and on every generation, so each number is read against two references:

  ceiling   what the judge scores on the real video -- the best any generator
            could do with this judge (D: r = 0.62 with ego speed; A: night/day
            AUC 0.999)
  per-clip  |metric(generation) - metric(real clip)|, the direct fidelity gap

and across clips the same association with ground truth that was validated on
real video is recomputed on each checkpoint's outputs:

  D  r(flow_mean, GT mean speed) and r(flow_mean, real-clip flow_mean)
  A  AUC of luminance separating night from day, and r with the real clip

A checkpoint that reproduces the factor keeps the association; one that ignores
the caption drops to r ~ 0 and AUC ~ 0.5 while its fidelity gap grows.

  python eval_fidelity.py --runs runs_wm/D_analysis runs_wm/D_control \\
      --clips 24 --out ./fidelity
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_same_noise import NEG, build_pipe, latent_shape
from metrics_agd import appearance, dynamics, geometry, read_frames, read_video

KEYS = ("luma", "saturation", "contrast", "flow_mean", "lane_centroid")


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def measure(frames):
    m = {}
    m.update(appearance(frames))
    d = dynamics(frames)
    d.pop("profile", None)
    m.update(d)
    m.update(geometry(frames))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--dataset", default="../wm_dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=24)
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--text_field", default="text_gtD")
    ap.add_argument("--no_guardrail", action="store_true", default=True)
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--skip_base", action="store_true",
                    help="when several checkpoint groups run in parallel, only one "
                         "of them needs to generate the shared base videos")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="extra out dirs whose rows.json is folded in for the report")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Path(args.dataset)
    rows_p = out / "rows.json"
    rows = json.loads(rows_p.read_text()) if rows_p.exists() else []
    done = {(r["ckpt"], r["clip"]) for r in rows}

    val = None
    for r in args.runs:
        cfg = json.loads((Path(r) / "config.json").read_text())
        val = set(cfg["val_ids"]) if val is None else val & set(cfg["val_ids"])
    ids = sorted(val)[: args.clips] if val else \
        [json.loads(l)["clip_id"] for l in open(ds / "clips.jsonl")][: args.clips]

    meta = {sid: json.loads((ds / sid / "caption.json").read_text()) for sid in ids}

    # ---- the anchor: the same judge on the real frames ----------------------
    for sid in ids:
        if ("real", sid) in done:
            continue
        rec = meta[sid]
        frames = read_frames(rec["source"]["frames"], args.frames)
        rows.append({"ckpt": "real", "clip": sid, **measure(frames)})
    rows_p.write_text(json.dumps(rows, indent=1))

    if not args.report_only and args.runs is not None:
        pipe = build_pipe(torch.bfloat16, False, False, args.no_guardrail)
        shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
        from diffusers.utils import export_to_video

        ckpts = ([] if args.skip_base else [("base", None)]) + \
                [(Path(r).name, r) for r in args.runs]
        for name, path in ckpts:
            if all((name, sid) in done for sid in ids):
                print(f"[skip] {name}")
                continue
            if path:
                from peft import PeftModel
                pipe.transformer = PeftModel.from_pretrained(
                    pipe.transformer, str(Path(path) / "final"))
                pipe.transformer.eval()
            for i, sid in enumerate(ids):
                if (name, sid) in done:
                    continue
                rec = meta[sid]
                fp = out / f"{name}__{sid}.mp4"
                if fp.exists():
                    frames = [np.asarray(f) for f in read_video(fp)]
                else:
                    img = Image.open(ds / sid / "cond_image.jpg").convert("RGB")
                    zp = out / f"z_{sid}.pt"
                    if zp.exists():
                        z = torch.load(zp)
                    else:
                        z = torch.randn(shape, generator=torch.Generator()
                                        .manual_seed(args.seed + i))
                        torch.save(z, zp)
                    with torch.no_grad():
                        r = pipe(image=img, prompt=rec["captions"]["AGD"][args.text_field],
                                 negative_prompt=NEG, height=args.height, width=args.width,
                                 num_frames=args.frames, num_inference_steps=args.steps,
                                 guidance_scale=args.guidance, fps=args.fps,
                                 generator=torch.Generator().manual_seed(args.seed),
                                 latents=z.clone().to("cuda", dtype=torch.bfloat16))
                    frames = [np.asarray(f) for f in r.frames[0]]
                    export_to_video([Image.fromarray(f) for f in frames],
                                    str(fp), fps=args.fps)
                rows.append({"ckpt": name, "clip": sid, **measure(frames)})
                rows_p.write_text(json.dumps(rows, indent=1))
                print(f"  [{name}] {sid}  luma {rows[-1]['luma']:.3f} "
                      f"flow {rows[-1]['flow_mean']:+.3f}", flush=True)
            if path:
                pipe.transformer = pipe.transformer.unload()

    # ---- statistics ---------------------------------------------------------
    for extra in args.merge:
        ep = Path(extra) / "rows.json"
        if ep.exists():
            have = {(r["ckpt"], r["clip"]) for r in rows}
            rows += [r for r in json.loads(ep.read_text())
                     if (r["ckpt"], r["clip"]) not in have]
    by = {}
    for r in rows:
        by.setdefault(r["ckpt"], {})[r["clip"]] = r
    real = by.get("real", {})
    common = sorted(set.intersection(*[set(v) for v in by.values()])) if by else []
    if len(common) < 4:
        print(f"[stats] only {len(common)} clips complete across checkpoints")
        return

    speed = np.array([meta[c]["gt"]["ego_speed_mean"] if "ego_speed_mean" in meta[c]["gt"]
                      else np.mean(meta[c]["gt"]["speed_profile"]) for c in common])
    night = np.array([meta[c]["is_night"] for c in common])
    rain = np.array([meta[c]["is_rain"] for c in common])

    print(f"\n=== {len(common)} held-out clips ===")
    print(f"{'checkpoint':14s} {'D: r(flow,GTspeed)':>19s} {'r(flow,real)':>13s} "
          f"{'|flow-real|':>12s} {'A: night/day AUC':>17s} {'r(luma,real)':>13s} "
          f"{'|luma-real|':>12s}")
    for name in ["real"] + [k for k in by if k != "real"]:
        v = by[name]
        flow = np.array([v[c]["flow_mean"] for c in common])
        luma = np.array([v[c]["luma"] for c in common])
        rflow = np.array([real[c]["flow_mean"] for c in common])
        rluma = np.array([real[c]["luma"] for c in common])
        r_sp = np.corrcoef(flow, speed)[0, 1] if flow.std() > 1e-9 else np.nan
        r_fl = np.corrcoef(flow, rflow)[0, 1] if flow.std() > 1e-9 and name != "real" else np.nan
        r_lu = np.corrcoef(luma, rluma)[0, 1] if luma.std() > 1e-9 and name != "real" else np.nan
        a_nd = auc(luma[~night], luma[night]) if night.any() and (~night).any() else np.nan
        print(f"{name:14s} {r_sp:19.3f} {r_fl:13.3f} "
              f"{np.mean(np.abs(flow - rflow)):12.3f} {a_nd:17.3f} {r_lu:13.3f} "
              f"{np.mean(np.abs(luma - rluma)):12.4f}")

    stats = {"clips": common, "n": len(common),
             "note": "real = judge on the original frames, the ceiling for each column"}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"\n[written] {out}/rows.json  {out}/stats.json")


if __name__ == "__main__":
    main()
