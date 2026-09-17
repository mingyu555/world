#!/usr/bin/env python3
"""Do the per-factor readouts actually recover the ground truth on real video?

Run before any of them is used to judge a generated clip.  For each real
wm_dataset clip it measures the readout and checks it against the label that
came from ego_pose or the map:

  D  flow-magnitude profile vs the ego speed profile -- per-clip Pearson r, and
     across clips the correlation of mean flow with mean speed, plus whether
     the flow trend agrees with accelerating / decelerating.
  A  luminance vs the night flag, blue-warm balance vs the rain flag (AUC).
  G  lane centroid vs which lane the ego is in (map ground truth).

  python validate_metrics.py --root ../wm_dataset --limit 60
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_agd import appearance, dynamics, geometry, read_frames


def auc(pos, neg):
    """Probability a positive scores above a negative (Mann-Whitney)."""
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    return float((r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../wm_dataset")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--frames", type=int, default=29)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    caps = sorted(Path(args.root).glob("*/caption.json"))[: args.limit]
    rows = []
    for i, p in enumerate(caps):
        rec = json.loads(p.read_text())
        frames = read_frames(rec["source"]["frames"], args.frames)
        if len(frames) < 8:
            continue
        m = {}
        m.update(appearance(frames))
        m.update(dynamics(frames))
        m.update(geometry(frames))
        gt = rec["gt"]
        sp = np.array(gt["speed_profile"][: args.frames], dtype=float)
        prof = np.array(m.pop("profile"))
        # align the flow profile (one value per frame pair) with the speed profile
        r = float("nan")
        if len(prof) > 3 and len(sp) > 3:
            k = min(len(prof), len(sp) - 1)
            a, b = prof[:k], sp[1:k + 1]
            if a.std() > 1e-6 and b.std() > 1e-6:
                r = float(np.corrcoef(a, b)[0, 1])
        g = rec.get("gt_geometry") or {}
        rows.append({
            "id": rec["sample_id"], **m, "speed_r": r,
            "speed_mean": float(sp.mean()), "speed_delta": float(sp[-1] - sp[0]),
            "is_night": bool(rec["is_night"]), "is_rain": bool(rec["is_rain"]),
            "events": gt["events"],
            "lane_count": g.get("lane_count"), "ego_from_right": g.get("ego_from_right"),
        })
        if i % 20 == 0:
            print(f"  [{i+1}/{len(caps)}] {rec['sample_id']}", flush=True)

    R = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
         for k in ("luma", "contrast", "saturation", "blue_warm", "flow_mean",
                   "flow_slope", "lane_centroid", "lane_spread", "speed_r",
                   "speed_mean", "speed_delta")}
    night = np.array([r["is_night"] for r in rows])
    rain = np.array([r["is_rain"] for r in rows])
    print(f"\n=== {len(rows)} clips ===")

    print("\nD  flow vs ego speed")
    ok = np.isfinite(R["speed_r"])
    print(f"   per-clip profile r: median {np.nanmedian(R['speed_r']):+.3f}, "
          f"{np.mean(R['speed_r'][ok] > 0.3):.0%} of clips above 0.3")
    print(f"   across clips, mean flow vs mean speed: "
          f"r = {np.corrcoef(R['flow_mean'], R['speed_mean'])[0,1]:+.3f}")
    acc = np.array(["accelerating" in r["events"] for r in rows])
    dec = np.array(["decelerating" in r["events"] for r in rows])
    if acc.any() and dec.any():
        print(f"   flow slope, accelerating {R['flow_slope'][acc].mean():+.4f} "
              f"vs decelerating {R['flow_slope'][dec].mean():+.4f}  "
              f"(AUC {auc(R['flow_slope'][acc], R['flow_slope'][dec]):.3f})")

    print("\nA  photometry vs labels")
    print(f"   luma: night {R['luma'][night].mean():.3f} vs day "
          f"{R['luma'][~night].mean():.3f}  (AUC {auc(R['luma'][~night], R['luma'][night]):.3f})")
    print(f"   blue_warm: rain {R['blue_warm'][rain].mean():+.3f} vs dry "
          f"{R['blue_warm'][~rain].mean():+.3f}  "
          f"(AUC {auc(R['blue_warm'][rain], R['blue_warm'][~rain]):.3f})")
    print(f"   saturation: rain {R['saturation'][rain].mean():.3f} vs dry "
          f"{R['saturation'][~rain].mean():.3f}  "
          f"(AUC {auc(R['saturation'][~rain], R['saturation'][rain]):.3f})")

    print("\nG  lane centroid vs map lane index")
    lane = np.array([r["ego_from_right"] if r["ego_from_right"] else np.nan
                     for r in rows], dtype=float)
    m = np.isfinite(lane)
    if m.sum() > 5 and np.nanstd(lane[m]) > 0:
        print(f"   r(lane_centroid, ego_from_right) = "
              f"{np.corrcoef(R['lane_centroid'][m], lane[m])[0,1]:+.3f}  (n={int(m.sum())})")
        for k in sorted(set(lane[m].astype(int))):
            s = R["lane_centroid"][m][lane[m] == k]
            print(f"     lane {k} from right: centroid {s.mean():+.4f} (n={len(s)})")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
