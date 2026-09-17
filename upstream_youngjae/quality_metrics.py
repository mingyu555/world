"""Video quality, alongside the per-factor readouts.

The factor judges say whether the caption was obeyed; they say nothing about
whether the video holds together.  That gap is not hypothetical -- the base
checkpoint produced smeared, artefacted 29-frame clips whose optical flow was
reading corruption rather than motion, and the factor metrics happily scored
them.

Three levels, because they fail at different sample sizes:

FVD          distribution distance between real and generated clips in the
             feature space of a Kinetics-pretrained 3D CNN.  The canonical
             number uses I3D; torchvision ships S3D with Kinetics-400 weights,
             which is the same family and gives a comparable ranking, so it is
             reported as `fvd_s3d` and must not be quoted as a canonical FVD.
             It also needs hundreds of clips to be stable -- at N < 50 treat it
             as indicative only, which is why the two below exist.

per-clip vs real   SSIM and PSNR against the real clip frame by frame.  The
             generation starts from the same conditioning frame, so early
             frames should match closely and the score degrades as the two
             trajectories diverge -- reported over all frames and over the
             first third separately.

temporal     mean absolute difference between consecutive frames, and the
             residual after compensating the dominant motion.  Corruption shows
             up here immediately: a coherent drive has smooth frame-to-frame
             change, a collapsing one spikes.  Works on a single clip.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# single-clip quality, no reference needed
# --------------------------------------------------------------------------- #
def temporal_stability(frames, stride=1):
    """How smoothly the video changes, and how much of that is not motion."""
    import cv2
    g = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
         for f in frames[::stride]]
    if len(g) < 3:
        return {"frame_delta": 0.0, "warp_residual": 0.0, "delta_spike": 0.0}
    deltas, resid = [], []
    for a, b in zip(g, g[1:]):
        deltas.append(float(np.abs(b - a).mean()))
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = a.shape
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        warped = cv2.remap(a, xx + fl[..., 0], yy + fl[..., 1],
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        resid.append(float(np.abs(b - warped).mean()))
    d = np.array(deltas)
    return {"frame_delta": float(d.mean()),
            "warp_residual": float(np.mean(resid)),
            # a collapsing clip has a few very large jumps, not a uniform drift
            "delta_spike": float(d.max() / max(np.median(d), 1e-6))}


# --------------------------------------------------------------------------- #
# paired with the real clip
# --------------------------------------------------------------------------- #
def vs_real(gen, real):
    """SSIM / PSNR against the real clip, overall and over the first third."""
    from skimage.metrics import structural_similarity as ssim
    import cv2
    n = min(len(gen), len(real))
    if n < 2:
        return {}
    s, p = [], []
    for i in range(n):
        a = cv2.cvtColor(gen[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(real[i], cv2.COLOR_RGB2GRAY)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        s.append(float(ssim(a, b, data_range=255)))
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        p.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
    k = max(1, n // 3)
    return {"ssim": float(np.mean(s)), "psnr": float(np.mean(p)),
            "ssim_early": float(np.mean(s[:k])), "psnr_early": float(np.mean(p[:k]))}


# --------------------------------------------------------------------------- #
# distribution level
# --------------------------------------------------------------------------- #
_MODEL = {}


def _s3d(device="cuda"):
    if "m" not in _MODEL:
        import torch
        from torchvision.models.video import s3d, S3D_Weights
        m = s3d(weights=S3D_Weights.KINETICS400_V1).eval().to(device)
        m.classifier = torch.nn.Identity()      # 1024-d pooled features
        _MODEL["m"] = m
    return _MODEL["m"]


def video_features(clips, device="cuda", size=224, n_frames=16, batch=4):
    """Kinetics S3D features for a list of clips (each a list of RGB frames)."""
    import torch
    import cv2
    m = _s3d(device)
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    feats = []
    for i in range(0, len(clips), batch):
        chunk = []
        for c in clips[i:i + batch]:
            idx = np.linspace(0, len(c) - 1, n_frames).astype(int)
            arr = np.stack([cv2.resize(c[j], (size, size)) for j in idx])
            chunk.append(arr)
        x = torch.from_numpy(np.stack(chunk)).to(device).float().div_(255.0)
        x = x.permute(0, 4, 1, 2, 3)            # B,C,T,H,W
        x = (x - mean) / std
        with torch.no_grad():
            f = m(x)
        feats.append(f.flatten(1).float().cpu().numpy())
    return np.concatenate(feats, 0)


def frechet(a, b, eps=1e-6):
    """Fréchet distance between two Gaussian-fitted feature sets.

    With fewer clips than feature dimensions the covariances are rank
    deficient and the matrix square root is ill conditioned, so a small ridge
    is added -- the usual FID practice.  It does not rescue the estimate at
    tiny N; it only keeps it finite.
    """
    from scipy import linalg
    mu1, mu2 = a.mean(0), b.mean(0)
    s1 = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    s2 = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1.dot(s2))
    if isinstance(covmean, tuple):          # older scipy returned (root, errest)
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def fvd(real_clips, gen_clips, device="cuda"):
    """Fréchet distance in S3D feature space.  See the module docstring: this is
    FVD-shaped, not the canonical I3D number, and needs many clips to settle."""
    fr = video_features(real_clips, device)
    fg = video_features(gen_clips, device)
    return {"fvd_s3d": frechet(fr, fg), "n_real": len(real_clips), "n_gen": len(gen_clips)}
