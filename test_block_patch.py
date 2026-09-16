"""Offline correctness test for BlockPatcher -- no model weights needed.

Instantiates the real diffusers `CosmosTransformerBlock` at toy dimensions on CPU
and checks the four properties the measurement depends on:

  1. with no tap active, the instrumented forward is bitwise identical to the
     original block forward;
  2. capture -> inject of a stream's own deltas is a no-op;
  3. injecting another stream's delta at one (layer, branch) changes the output;
  4. injecting another stream's deltas at *every* (layer, branch) reproduces that
     stream exactly -- because the block is pure residual, so restore == 1.0.

Property 4 is the one that makes `restore` interpretable as "fraction of the
factor's effect that flows through this tap point".

    python test_block_patch.py
"""

import sys

import torch
from diffusers.models.transformers.transformer_cosmos import CosmosTransformerBlock

from block_patch import BRANCHES, BlockPatcher, restore_scores

torch.manual_seed(0)

HEADS, HEAD_DIM, TEXT_DIM, ADALN = 2, 8, 16, 8
HIDDEN = HEADS * HEAD_DIM
S_IMG, S_TXT, N_LAYERS = 12, 7, 3


class Stack(torch.nn.Module):
    """Minimal stand-in for CosmosTransformer3DModel's block list."""

    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=HEADS,
                    attention_head_dim=HEAD_DIM,
                    cross_attention_dim=TEXT_DIM,
                    adaln_lora_dim=ADALN,
                )
                for _ in range(N_LAYERS)
            ]
        )

    def run(self, h, text, emb_t, temb):
        for blk in self.transformer_blocks:
            h = blk(h, text, emb_t, temb, None, None, None, None)
        return h


def main():
    stack = Stack().eval()
    h0 = torch.randn(1, S_IMG, HIDDEN)
    emb_t = torch.randn(1, HIDDEN)
    temb = torch.randn(1, 3 * HIDDEN)
    text_a = torch.randn(1, S_TXT, TEXT_DIM)          # "AGD"
    text_b = torch.randn(1, S_TXT, TEXT_DIM)          # "AGD minus one factor"

    with torch.no_grad():
        ref_a = stack.run(h0, text_a, emb_t, temb)
        ref_b = stack.run(h0, text_b, emb_t, temb)

    patcher = BlockPatcher(stack, cache_device="cpu")
    patcher.attach()
    ok = True

    # 1. pass-through identity ------------------------------------------ #
    with torch.no_grad():
        out = stack.run(h0, text_a, emb_t, temb)
    same = torch.equal(out, ref_a)
    print(f"[1] pass-through identical to original forward : {same}")
    ok &= same

    # 2. capture then re-inject the same stream -------------------------- #
    with torch.no_grad():
        with patcher.capture():
            _ = stack.run(h0, text_a, emb_t, temb)
        own = dict(patcher.store)
        with patcher.injection(own):
            out = stack.run(h0, text_a, emb_t, temb)
    same = torch.equal(out, ref_a)
    print(f"[2] self-injection is a no-op                   : {same}  "
          f"(captured {len(own)} taps = {N_LAYERS} layers x {len(BRANCHES)} branches)")
    ok &= same and len(own) == N_LAYERS * len(BRANCHES)

    # capture the *other* stream's deltas
    with torch.no_grad():
        with patcher.capture():
            _ = stack.run(h0, text_b, emb_t, temb)
        other = dict(patcher.store)

    # 3. single-tap injection moves the output --------------------------- #
    d_ref = (ref_b - ref_a).flatten().float()
    print("[3] single-tap injection (restore = fraction of the b-vs-a difference"
          " reproduced):")
    total = 0.0
    for layer in range(N_LAYERS):
        for br in BRANCHES:
            with torch.no_grad(), patcher.injection({(layer, br): other[(layer, br)]}):
                out = stack.run(h0, text_a, emb_t, temb)
            sc = restore_scores((out - ref_a).flatten().float(), d_ref)
            total += sc["restore"]
            print(f"      L{layer} {br:5s}  restore={sc['restore']:+.4f}  "
                  f"mag={sc['mag']:.4f}  align={sc['align']:+.4f}")
    moved = abs(total) > 1e-6
    print(f"      sum of single-tap restores = {total:+.4f}")
    ok &= moved

    # 4. full injection reproduces the other stream exactly --------------- #
    with torch.no_grad(), patcher.injection(other):
        out = stack.run(h0, text_a, emb_t, temb)
    err = (out - ref_b).abs().max().item()
    sc = restore_scores((out - ref_a).flatten().float(), d_ref)
    print(f"[4] full injection == stream b                  : max|err|={err:.2e}  "
          f"restore={sc['restore']:.6f} (expect 1.000000)")
    ok &= err < 1e-6 and abs(sc["restore"] - 1.0) < 1e-4

    patcher.detach()
    with torch.no_grad():
        out = stack.run(h0, text_a, emb_t, temb)
    same = torch.equal(out, ref_a)
    print(f"[5] detach restores the original forward        : {same}")
    ok &= same

    print("\nALL PASS" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
