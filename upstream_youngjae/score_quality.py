#!/usr/bin/env python3
"""Score every generated clip in a fidelity run for quality, against the real one.

Reads the videos `eval_fidelity.py` already wrote, so no generation is repeated.
Produces one table per checkpoint with the real clips as the reference row:

  fvd_s3d        distribution distance to the real clips (indicative at small N)
  ssim / psnr    frame-by-frame agreement with the real clip
  frame_delta    consecutive-frame change; corruption inflates it
  delta_spike    largest jump over the median jump; a collapsing clip spikes
  warp_residual  what is left after compensating the dominant motion

  python score_quality.py --dirs ./fidelity ./fidelity_A --dataset ../wm_dataset
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_agd import read_frames, read_video
from quality_metrics import fvd, temporal_stability, vs_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--dataset", default="../wm_dataset")
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--out", default="")
    ap.add_argument("--no_fvd", action="store_true")
    args = ap.parse_args()

    ds = Path(args.dataset)
    vids = defaultdict(dict)              # ckpt -> clip -> path
    for d in args.dirs:
        for p in sorted(Path(d).glob("*.mp4")):
            stem = p.stem
            if "__" not in stem:
                continue
            parts = stem.split("__")
            ck, clip = parts[0], parts[1]
            if len(parts) > 2 and parts[2] != "true":
                continue                  # counterfactual renders are not scored here
            vids[ck][clip] = p
    if not vids:
        raise SystemExit("no generated videos found in " + ", ".join(args.dirs))

    clips = sorted(set.intersection(*[set(v) for v in vids.values()]))
    print(f"[quality] {len(vids)} checkpoints x {len(clips)} clips in common")
    if not clips:
        for ck, v in vids.items():
            print(f"   {ck}: {len(v)} clips")
        raise SystemExit("no clip is present for every checkpoint yet")

    real = {c: read_frames(json.loads((ds / c / "caption.json").read_text())
                           ["source"]["frames"], args.frames) for c in clips}

    rows, per_ckpt = [], {}
    for ck in sorted(vids):
        gens, qs = [], []
        for c in clips:
            g = [np.asarray(f) for f in read_video(vids[ck][c])]
            gens.append(g)
            q = {"ckpt": ck, "clip": c}
            q.update(temporal_stability(g))
            q.update(vs_real(g, real[c]))
            rows.append(q)
            qs.append(q)
        per_ckpt[ck] = {k: float(np.mean([q[k] for q in qs]))
                        for k in qs[0] if k not in ("ckpt", "clip")}
        if not args.no_fvd:
            per_ckpt[ck].update(fvd([real[c] for c in clips], gens))
        print(f"  scored {ck}", flush=True)

    ref = {k: float(np.mean([v for v in
                             [temporal_stability(real[c])[k] for c in clips]]))
           for k in ("frame_delta", "warp_residual", "delta_spike")}

    print(f"\n=== {len(clips)} held-out clips ===")
    hdr = f"{'checkpoint':16s} {'fvd_s3d':>9s} {'ssim':>7s} {'psnr':>7s} " \
          f"{'ssim_early':>11s} {'frame_delta':>12s} {'delta_spike':>12s} {'warp_res':>9s}"
    print(hdr)
    print(f"{'real (ref)':16s} {'-':>9s} {'-':>7s} {'-':>7s} {'-':>11s} "
          f"{ref['frame_delta']:12.4f} {ref['delta_spike']:12.2f} {ref['warp_residual']:9.4f}")
    for ck, v in per_ckpt.items():
        print(f"{ck:16s} {v.get('fvd_s3d', float('nan')):9.1f} {v['ssim']:7.3f} "
              f"{v['psnr']:7.2f} {v['ssim_early']:11.3f} {v['frame_delta']:12.4f} "
              f"{v['delta_spike']:12.2f} {v['warp_residual']:9.4f}")
    if not args.no_fvd and len(clips) < 50:
        print(f"\nnote: fvd_s3d over {len(clips)} clips is indicative only; the "
              f"Frechet estimate needs hundreds of samples to settle. Rank "
              f"checkpoints by ssim / delta_spike at this N.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"per_clip": rows, "per_checkpoint": per_ckpt, "real_reference": ref,
             "n_clips": len(clips)}, indent=1))
        print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
