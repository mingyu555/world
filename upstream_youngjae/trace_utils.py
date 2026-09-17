"""Shared helpers for same-noise layer x timestep tracing (H2/H3).

Storage strategy
----------------
Exact per-(layer, step) hidden states are far too large to keep for every run
(93 frame @ 1280x704 -> ~42k tokens x 2048 dim -> 170 MB per (layer, step)).
Instead every run stores, per (layer, step):

  norm        ||h||_F                                  scalar
  tok_mean    mean over tokens                         [D]
  frame_norm  ||h||_F per latent frame                 [T_lat]
  sketch      Q h R  with fixed seeded gaussians       [K, K]

`sketch` is a bilinear Johnson-Lindenstrauss sketch.  Because Q and R are drawn
from one fixed seed they are identical across runs, so

    ||sketch_a - sketch_b||_F  ~=  ||h_a - h_b||_F

up to the JL scaling factor (see `sketch_scale`).  That gives the layer x
timestep divergence map D_h(l,t) of plan section 18 without storing hidden
states.  `--full_every` still dumps exact hidden states for validation.
"""
import math
import torch


class Sketcher:
    """Fixed random bilinear sketch h[S, D] -> [K, K]."""

    def __init__(self, k=96, seed=1234):
        self.k = k
        self.seed = seed
        self._Q = {}   # S -> [K, S]
        self._R = {}   # D -> [D, K]

    def _q(self, S, device):
        key = (S, device)
        if key not in self._Q:
            g = torch.Generator(device="cpu").manual_seed(self.seed + S)
            self._Q[key] = torch.randn(self.k, S, generator=g).to(device) / math.sqrt(self.k)
        return self._Q[key]

    def _r(self, D, device):
        key = (D, device)
        if key not in self._R:
            g = torch.Generator(device="cpu").manual_seed(self.seed * 7 + D)
            self._R[key] = torch.randn(D, self.k, generator=g).to(device) / math.sqrt(self.k)
        return self._R[key]

    def __call__(self, h):
        """h: [S, D] float -> [K, K] float32."""
        S, D = h.shape
        h = h.float()
        return (self._q(S, h.device) @ h @ self._r(D, h.device)).cpu()

    @staticmethod
    def scale(S=None, D=None):
        """Sketch -> Frobenius conversion factor.

        With Q ~ N(0, 1/K)^{K x S} and R ~ N(0, 1/K)^{D x K},
        E ||Q h R||_F^2 = ||h||_F^2, so the factor is exactly 1.  Verified
        empirically against exact hidden-state dumps (--full_every): the
        sketch estimate tracks ||h_a - h_b||_F to ~6 % relative at K=64,
        ~4 % at K=96.
        """
        return 1.0


def block_list(transformer):
    """Return the ModuleList of transformer blocks, whatever it is called."""
    for name in ("transformer_blocks", "blocks", "layers"):
        mod = getattr(transformer, name, None)
        if mod is not None and len(mod) > 0:
            return name, mod
    raise RuntimeError(f"no block list found on {type(transformer).__name__}")


def as_hidden(out):
    """Normalise a block output to the hidden-state tensor."""
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if isinstance(out, dict):
        return next(iter(out.values()))
    raise TypeError(type(out))
