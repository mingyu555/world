"""Branch-resolved residual tap for CosmosTransformerBlock.

A Cosmos block is three pure-residual writes:

    h = h + gate1 * attn1(norm1(h))          self attention   (spatio-temporal)
    h = h + gate2 * attn2(norm2(h), text)    cross attention  (the only text path)
    h = h + gate3 * ff(norm3(h))             feed forward

`BlockPatcher` replaces every block's `forward` with an identical copy that exposes
each of those three deltas at a tap point, so a delta can be

  * recorded   (mode="capture")  -> store[(layer, branch)]
  * substituted(mode="inject")   -> the block adds someone else's delta instead

Substituting the delta computed under a *different prompt*, while leaving the rest
of the stream untouched, is the local causal question of plan section 27: does the
factor's effect on the model output flow through this (layer, branch)?

The replacement forward must be numerically identical to the original when no tap
is active; `test_block_patch.py` checks that against the real diffusers class.

Only the plain text-conditioned configuration is supported: no controlnet
(`controlnet_residual`), no zero-conv (`before_proj`/`after_proj`), no image
context. Those are asserted, not silently ignored.
"""

import types

import torch

BRANCHES = ("attn1", "attn2", "ff")


class BlockPatcher:
    def __init__(self, transformer, cache_device="cpu"):
        self.blocks = transformer.transformer_blocks
        self.n_layers = len(self.blocks)
        self.cache_device = cache_device

        self.mode = None          # None | "capture" | "inject"
        self.enabled = False      # only tap while the probe says so
        self.store = {}           # (layer, branch) -> delta on cache_device
        self.inject = {}          # (layer, branch) -> delta to substitute
        self.h_norm = {}          # layer -> ||h|| after the block (capture mode)
        self.capture_keys = None  # None = every tap; else only these (layer, branch)
        self.reference = {}       # (layer, branch) -> delta to compare against
        self.graft = {}           # layer -> h to force as that block's input
        self.text = {}            # layer -> encoder_hidden_states for that block only
        self.h_in = {}            # capture mode: layer -> h at block entry
        self.diff = {}            # compare mode: ||delta - reference||
        self.mag = {}             # compare mode: ||delta||

        self._orig = []

    # ------------------------------------------------------------------ #
    def attach(self):
        if self._orig:
            return
        for i, blk in enumerate(self.blocks):
            assert getattr(blk, "before_proj", None) is None, "controlnet block unsupported"
            assert getattr(blk, "after_proj", None) is None, "controlnet block unsupported"
            self._orig.append(blk.forward)
            blk.forward = types.MethodType(_make_forward(i, self), blk)

    def detach(self):
        for blk, fwd in zip(self.blocks, self._orig):
            blk.forward = fwd
        self._orig = []

    # ------------------------------------------------------------------ #
    def tap(self, layer, branch, delta):
        """Called by the instrumented forward for every branch of every block."""
        if not self.enabled:
            return delta
        key = (layer, branch)
        if self.mode == "compare":
            # Accumulate ||delta - reference|| and ||delta|| without storing the
            # tensor. Lets many prompts be compared against one captured
            # reference at no transfer cost.
            ref = self.reference.get(key)
            if ref is not None:
                d = delta.detach()
                r = ref.to(d.device, dtype=d.dtype, non_blocking=True)
                self.diff[key] = float((d - r).float().norm())
                self.mag[key] = float(d.float().norm())
            return delta
        if self.mode == "capture":
            if self.capture_keys is None or key in self.capture_keys:
                self.store[key] = delta.detach().to(self.cache_device, copy=True)
            return delta
        if self.mode == "inject":
            sub = self.inject.get(key)
            if sub is not None:
                return sub.to(delta.device, dtype=delta.dtype, non_blocking=True)
        return delta

    def note_h(self, layer, h):
        if self.enabled and self.mode == "capture":
            self.h_norm[layer] = h.detach().float().norm().item()

    def entry(self, layer, h):
        """Called at block entry. In capture mode records the input; if a graft is
        set, hands the block the reference stream's input instead of its own.

        Grafting makes the measurement local: every block then sees the *same*
        input under every prompt, so a difference in what it writes can only come
        from the text. Without it, block l's input has already diverged from the
        reference by layers 0..l-1, and the write difference mixes "this block
        processed the text differently" with "this block was handed a different
        input".
        """
        if not self.enabled:
            return h
        if self.mode == "capture":
            self.h_in[layer] = h.detach().to(self.cache_device, copy=True)
            self.h_in_norm = getattr(self, "h_in_norm", {})
            self.h_in_norm[layer] = h.detach().float().norm().item()
        g = self.graft.get(layer)
        if g is not None:
            return g.to(h.device, dtype=h.dtype, non_blocking=True)
        return h

    # ------------------------------------------------------------------ #
    def capture(self):
        """Context: record every branch delta of the next forward."""
        return _Mode(self, "capture")

    def comparison(self, reference):
        """Context: measure each branch delta against `reference`, storing scalars."""
        return _Mode(self, "compare", reference=reference)

    def text_override(self, mapping):
        """Context: give the listed blocks a different encoder_hidden_states."""
        return _Mode(self, None, text=mapping)

    def injection(self, subs):
        """Context: substitute the given {(layer, branch): delta} in the next forward."""
        return _Mode(self, "inject", subs)

    def clear(self):
        self.store = {}
        self.h_norm = {}
        self.h_in = {}


class _Mode:
    def __init__(self, patcher, mode, subs=None, reference=None, text=None):
        self.p, self.mode, self.subs = patcher, mode, subs or {}
        self.reference = reference or {}
        self.text = text or {}

    def __enter__(self):
        if self.mode == "capture":
            self.p.clear()
        if self.mode == "compare":
            self.p.reference = self.reference
            self.p.diff, self.p.mag = {}, {}
        self.p.inject = self.subs
        self.p.text = self.text
        self.p.mode = self.mode
        self.p.enabled = True
        return self.p

    def __exit__(self, *exc):
        self.p.enabled = False
        self.p.mode = None
        self.p.inject = {}
        self.p.text = {}
        return False


# ---------------------------------------------------------------------- #
def _make_forward(layer_idx, patcher):
    """A copy of CosmosTransformerBlock.forward with tap points.

    Signature and argument order follow diffusers
    `CosmosTransformer3DModel.forward`, which calls each block positionally as
        block(hidden_states, encoder_hidden_states, embedded_timestep, temb,
              image_rotary_emb, extra_pos_emb, attention_mask, controlnet_residual)
    """

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        embedded_timestep=None,
        temb=None,
        image_rotary_emb=None,
        extra_pos_emb=None,
        attention_mask=None,
        controlnet_residual=None,
        latents=None,
        block_idx=None,
    ):
        assert controlnet_residual is None, "controlnet residual unsupported"

        hidden_states = patcher.entry(layer_idx, hidden_states)

        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. self attention
        norm_hidden_states, gate = self.norm1(hidden_states, embedded_timestep, temb)
        delta = gate * self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + patcher.tap(layer_idx, "attn1", delta)

        # 2. cross attention (the text path)
        # A per-layer text override makes the causal question directly askable:
        # give one block the factor prompt and every other block the base prompt,
        # and whatever moves in the output is what that block's reading of the
        # text contributes. attn2 takes the text as an argument, so this costs
        # nothing structurally.
        txt = patcher.text.get(layer_idx, encoder_hidden_states) \
            if patcher.enabled else encoder_hidden_states
        norm_hidden_states, gate = self.norm2(hidden_states, embedded_timestep, temb)
        delta = gate * self.attn2(
            norm_hidden_states,
            encoder_hidden_states=txt,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + patcher.tap(layer_idx, "attn2", delta)

        # 3. feed forward
        norm_hidden_states, gate = self.norm3(hidden_states, embedded_timestep, temb)
        delta = gate * self.ff(norm_hidden_states)
        hidden_states = hidden_states + patcher.tap(layer_idx, "ff", delta)

        patcher.note_h(layer_idx, hidden_states)
        return hidden_states

    return forward


# ---------------------------------------------------------------------- #
def restore_scores(d_patch, d_ref):
    """How much of the reference difference does this patch reproduce?

    d_patch = v_patched - v_full ,  d_ref = v_remove(g) - v_full   (flat float)

        restore = <d_patch, d_ref> / ||d_ref||^2      1.0 = fully explains g
        mag     = ||d_patch|| / ||d_ref||
        align   = cos(d_patch, d_ref)                 restore = mag * align

    `restore` is a projection, so contributions of disjoint (layer, branch) tap
    points are directly comparable and roughly additive -- unlike the cosine
    alone, which saturates (see cosmos_exp/README.md on the generate-and-compare
    failure).
    """
    ref_sq = d_ref.dot(d_ref).clamp_min(1e-12)
    restore = (d_patch.dot(d_ref) / ref_sq).item()
    n_p = d_patch.norm()
    n_r = d_ref.norm().clamp_min(1e-12)
    mag = (n_p / n_r).item()
    align = torch.nn.functional.cosine_similarity(d_patch, d_ref, dim=0).item()
    return {"restore": restore, "mag": mag, "align": align}
