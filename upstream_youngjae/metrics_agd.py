"""Per-factor readouts measured straight off a video.

These are the judges the placement claim needs: held-out denoising loss cannot
tell the layer sets apart (0.0003-0.0018 apart, mixed sign), because
reconstruction error is overall fit, not factor control.  Each factor gets a
scalar that can be read from pixels and checked against ground truth:

  A  mean luminance and blue-vs-warm balance -- night/day and rain are
     photometric, so the frames carry the answer directly.
  D  median vertical displacement of sparse tracked corners in the road region,
     which tracks ego speed.
  G  lateral centroid of lane-marking energy in the lower image.

Every one of them was validated on the real clips first
(`validate_metrics.py`) -- a readout that cannot recover the ground truth on
real video has no business judging generated video -- and the results decide
what each may be used for:

  A  WORKS.  luminance separates night from day with AUC 0.999 (122 clips);
     saturation separates rain from dry with AUC 0.923.  blue-vs-warm is
     useless (0.52) and is kept only as a diagnostic.
  D  WORKS AT CLIP LEVEL ONLY.  `lk_dy` correlates with mean ego speed at
     r = +0.62 across clips, so speed-level contrasts (stationary vs moving)
     are measurable.  The within-clip profile does not track the speed profile
     (median r = +0.09) and accelerating vs decelerating is at chance
     (AUC 0.50) with or without smoothing, so acceleration must NOT be tested
     with it.  Dense Farneback was worse still: bottom-band dy correlated
     -0.31 with speed, wide-ROI magnitude +0.03.
  G  DOES NOT WORK.  the lane centroid is uncorrelated with the map's lane
     index (r = -0.06, and the per-lane means are not monotonic).  Geometry
     needs a real lane detector; until then G has no pixel-level judge.
"""
import numpy as np

ROI_TOP = 0.55          # road occupies roughly the lower half of a front camera
ROI_BOT = 0.95
ROI_L, ROI_R = 0.25, 0.75


def _roi(h, w):
    return (slice(int(ROI_TOP * h), int(ROI_BOT * h)),
            slice(int(ROI_L * w), int(ROI_R * w)))


def appearance(frames):
    """Photometric summary: luminance, colour balance, contrast."""
    import cv2
    lum, blue, sat, con = [], [], [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        lum.append(g.mean() / 255.0)
        con.append(g.std() / 255.0)
        sat.append(hsv[..., 1].mean() / 255.0)
        b, r = f[..., 2].mean(), f[..., 0].mean()
        blue.append((b - r) / 255.0)
    return {"luma": float(np.mean(lum)), "contrast": float(np.mean(con)),
            "saturation": float(np.mean(sat)), "blue_warm": float(np.mean(blue))}


def flow_profile(frames, step=1):
    """Median vertical displacement of tracked corners, per frame pair.

    Sparse Lucas-Kanade on Shi-Tomasi corners restricted to the road region.
    For a forward camera the image motion of ground features is downward and
    grows with ego speed, so the median dy is a speed proxy; the median rejects
    the few features on independently moving vehicles.  Sparse tracking beat
    every dense variant tried (r = +0.62 with mean ego speed vs +0.03 for dense
    magnitude over the same ROI) because asphalt gives dense flow almost no
    texture to lock onto.
    """
    import cv2
    prev, out = None, []
    for i in range(0, len(frames), step):
        f = frames[i]
        h, w = f.shape[:2]
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            mask = np.zeros_like(prev)
            mask[int(0.60 * h):int(0.97 * h), int(0.15 * w):int(0.85 * w)] = 255
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 7, mask=mask)
            if p0 is not None and len(p0) > 15:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None,
                                                     winSize=(21, 21), maxLevel=3)
                ok = st.reshape(-1) == 1
                d = (p1 - p0).reshape(-1, 2)[ok]
                out.append(float(np.median(d[:, 1])) if len(d) else np.nan)
            else:
                out.append(np.nan)
        prev = g
    a = np.array(out, dtype=float)
    if len(a) and np.isnan(a).any():
        idx = np.arange(len(a))
        ok = ~np.isnan(a)
        a = np.interp(idx, idx[ok], a[ok]) if ok.sum() > 1 else np.zeros_like(a)
    return a


def dynamics(frames, step=1):
    """Clip-level speed proxy.

    Only `flow_mean` is trustworthy: the slope is reported for completeness but
    does not discriminate accelerating from decelerating (AUC 0.50 on real
    clips), so it must not be used to test an acceleration claim.
    """
    p = flow_profile(frames, step=step)
    if len(p) < 4:
        return {"flow_mean": 0.0, "flow_slope": 0.0, "profile": []}
    t = np.arange(len(p), dtype=float)
    return {"flow_mean": float(np.mean(p)),
            "flow_slope": float(np.polyfit(t, p, 1)[0]),
            "profile": p.tolist()}


def geometry(frames, n=8):
    """Lateral centroid of lane-marking energy, and how spread it is.

    Bright, locally-contrasty pixels in the lower image are dominated by lane
    paint; where their mass sits horizontally shifts with the ego's lane.
    Reported in units of image width from centre (negative = left).
    """
    import cv2
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    cents, spreads = [], []
    for i in idx:
        f = frames[i]
        h, w = f.shape[:2]
        ys, xs = _roi(h, w)
        g = cv2.cvtColor(f[ys, xs], cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        thr = np.percentile(g, 92)
        mask = (g >= thr).astype(np.float32)
        col = mask.mean(0)
        if col.sum() < 1e-6:
            continue
        x = np.arange(len(col), dtype=float)
        c = float((col * x).sum() / col.sum())
        v = float(np.sqrt((col * (x - c) ** 2).sum() / col.sum()))
        cents.append((c / len(col)) - 0.5)
        spreads.append(v / len(col))
    if not cents:
        return {"lane_centroid": 0.0, "lane_spread": 0.0}
    return {"lane_centroid": float(np.mean(cents)),
            "lane_spread": float(np.mean(spreads))}


def all_metrics(frames, flow_step=1):
    m = {}
    m.update(appearance(frames))
    m.update(dynamics(frames, step=flow_step))
    m.update(geometry(frames))
    return m


def read_video(path, max_frames=0):
    import cv2
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        if max_frames and len(out) >= max_frames:
            break
    cap.release()
    return out


def read_frames(paths, max_frames=0):
    import cv2
    out = []
    for p in paths[: max_frames or None]:
        im = cv2.imread(str(p))
        if im is not None:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out
