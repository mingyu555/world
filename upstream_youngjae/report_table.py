#!/usr/bin/env python3
"""One table: every checkpoint, factor control and video quality side by side.

Merges the per-factor fidelity runs (which each generated a different pair of
LoRA checkpoints on a different GPU) into a single comparison, with the real
clips as the first row so every column has a scale.

Columns

  factor control -- the same judges that were validated on real video, rerun on
  each checkpoint's generations:
     r(flow, speed)   association between the flow speed proxy and the ego speed
                      ground truth; on real video this is +0.62, which is the
                      ceiling
     night/day AUC    luminance separating night clips from day clips; 0.999 on
                      real video, 0.5 means the checkpoint ignores it
     |flow-real|,     absolute gap to the real clip's own reading
     |luma-real|
  Geometry has no column: its pixel judge failed validation (lane centroid vs
  map lane index r = -0.06), so G checkpoints are compared on quality alone.

  quality -- see quality_metrics.py
     fvd_s3d, ssim, psnr, delta_spike

  python report_table.py --dirs ./fidelity ./fidelity_A ./fidelity_G \\
      --dataset ../wm_dataset --out ../checkpoint_table.json
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

ORDER = ["real", "base", "A_analysis", "A_control", "G_analysis", "G_control",
         "D_analysis", "D_control"]


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--dataset", default="../wm_dataset")
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--out", default="")
    ap.add_argument("--no_quality", action="store_true")
    ap.add_argument("--min_clips", type=int, default=6)
    args = ap.parse_args()

    ds = Path(args.dataset)

    # ---- factor rows, merged across the per-factor runs ---------------------
    rows = []
    seen = set()
    for d in args.dirs:
        p = Path(d) / "rows.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            k = (r["ckpt"], r["clip"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
    by = defaultdict(dict)
    for r in rows:
        by[r["ckpt"]][r["clip"]] = r
    if "real" not in by:
        raise SystemExit("no 'real' rows -- run eval_fidelity.py first")

    # a checkpoint is included once it has enough clips; the common set is the
    # intersection over included checkpoints so every column compares like with like
    keep = {k: v for k, v in by.items() if len(v) >= args.min_clips}
    clips = sorted(set.intersection(*[set(v) for v in keep.values()]))
    print(f"[table] {len(keep)} checkpoints, {len(clips)} clips in common")
    for k, v in by.items():
        mark = "" if k in keep else "  (dropped, too few clips)"
        print(f"   {k:14s} {len(v):3d} clips{mark}")
    if len(clips) < args.min_clips:
        raise SystemExit("not enough clips shared across checkpoints yet")

    meta = {c: json.loads((ds / c / "caption.json").read_text()) for c in clips}
    speed = np.array([np.mean(meta[c]["gt"]["speed_profile"]) for c in clips])
    night = np.array([meta[c]["is_night"] for c in clips])

    # ---- quality, from the videos those runs wrote ---------------------------
    qual = {}
    if not args.no_quality:
        vids = defaultdict(dict)
        for d in args.dirs:
            for p in sorted(Path(d).glob("*.mp4")):
                parts = p.stem.split("__")
                if len(parts) < 2 or (len(parts) > 2 and parts[2] != "true"):
                    continue
                vids[parts[0]][parts[1]] = p
        real_frames = {c: read_frames(meta[c]["source"]["frames"], args.frames)
                       for c in clips}
        ref = [temporal_stability(real_frames[c]) for c in clips]
        qual["real"] = {k: float(np.mean([r[k] for r in ref]))
                        for k in ("frame_delta", "warp_residual", "delta_spike")}
        for ck in keep:
            if ck == "real" or ck not in vids:
                continue
            have = [c for c in clips if c in vids[ck]]
            if len(have) < args.min_clips:
                continue
            gens = [[np.asarray(f) for f in read_video(vids[ck][c])] for c in have]
            q = defaultdict(list)
            for g, c in zip(gens, have):
                for k, v in temporal_stability(g).items():
                    q[k].append(v)
                for k, v in vs_real(g, real_frames[c]).items():
                    q[k].append(v)
            qual[ck] = {k: float(np.mean(v)) for k, v in q.items()}
            qual[ck].update(fvd([real_frames[c] for c in have], gens))
            print(f"  quality scored {ck} ({len(have)} clips)", flush=True)

    # ---- print --------------------------------------------------------------
    def get(ck, key):
        return np.array([keep[ck][c][key] for c in clips])

    real_flow, real_luma = get("real", "flow_mean"), get("real", "luma")
    names = [n for n in ORDER if n in keep] + \
            [n for n in keep if n not in ORDER]

    print(f"\n=== {len(clips)} held-out clips ===\n")
    print(f"{'checkpoint':14s} | {'r(flow,speed)':>13s} {'|flow-real|':>11s} "
          f"{'night/day AUC':>13s} {'|luma-real|':>11s} | "
          f"{'fvd_s3d':>8s} {'ssim':>6s} {'psnr':>6s} {'spike':>6s}")
    print("-" * 108)
    out_rows = {}
    for n in names:
        flow, luma = get(n, "flow_mean"), get(n, "luma")
        r_sp = np.corrcoef(flow, speed)[0, 1] if flow.std() > 1e-9 else np.nan
        a_nd = auc(luma[~night], luma[night]) if night.any() and (~night).any() else np.nan
        gf = np.nan if n == "real" else float(np.mean(np.abs(flow - real_flow)))
        gl = np.nan if n == "real" else float(np.mean(np.abs(luma - real_luma)))
        q = qual.get(n, {})
        f = lambda v, w, p: (f"{v:{w}.{p}f}" if v == v else " " * (w - 1) + "-")
        print(f"{n:14s} | {f(r_sp,13,3)} {f(gf,11,3)} {f(a_nd,13,3)} {f(gl,11,4)} | "
              f"{f(q.get('fvd_s3d', np.nan),8,1)} {f(q.get('ssim', np.nan),6,3)} "
              f"{f(q.get('psnr', np.nan),6,2)} {f(q.get('delta_spike', np.nan),6,2)}")
        out_rows[n] = {"r_flow_speed": r_sp, "flow_gap": gf,
                       "night_day_auc": a_nd, "luma_gap": gl, **q}

    print(f"\nceiling on real video: r(flow,speed) +0.62, night/day AUC 0.999, "
          f"delta_spike {qual.get('real', {}).get('delta_spike', float('nan')):.2f}")
    print("Geometry has no factor column: its pixel judge failed validation "
          "(r = -0.06 with the map lane index), so G rows are quality only.")
    if len(clips) < 50 and not args.no_quality:
        print(f"fvd_s3d over {len(clips)} clips is indicative only.")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"n_clips": len(clips), "clips": clips, "rows": out_rows}, indent=1))
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
