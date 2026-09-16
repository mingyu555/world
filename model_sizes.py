#!/usr/bin/env python3
"""Exact parameter counts for the three Cosmos-Predict2-2B-Video2World modules.

The repo is gated and `hf_cache` is not readable from this account, so the
configs are *reconstructed* from what is measured and known, then every count is
cross-checked against the checkpoint file sizes on the Hub (which are public
metadata). If bytes/param comes out at 2.0 (bf16) or 4.0 (fp32) the
reconstruction is right.

Measured facts this is built on (cosmos_exp traces of the real model):
    28 transformer blocks, hidden 2048, seq 84,480 for a 24x88x160 latent
    -> patch_size (1,2,2), since 24 * 44 * 80 = 84,480
    latent_shape used `config.in_channels - 1 == 16`  -> in_channels 17
    tokenizer max_sequence_length 512, T5EncoderModel + AutoencoderKLWan

Note the parameter count is essentially independent of the head split: q/k/v/o
are hidden x hidden either way, only the tiny RMSNorm over head_dim differs.

    python model_sizes.py
"""

import torch
from accelerate import init_empty_weights

REPO = "nvidia/Cosmos-Predict2-2B-Video2World"

# --- reconstructed configs ------------------------------------------------ #
DIT = dict(
    in_channels=17,                 # 16 latent channels + condition mask
    out_channels=16,
    num_attention_heads=16,         # 16 x 128 = 2048 (measured hidden)
    attention_head_dim=128,
    num_layers=28,                  # measured
    mlp_ratio=4.0,
    text_embed_dim=1024,            # T5-11B d_model
    adaln_lora_dim=256,
    max_size=(128, 240, 240),
    patch_size=(1, 2, 2),           # measured
    rope_scale=(2.0, 1.0, 1.0),
    concat_padding_mask=True,
    extra_pos_embed_type="learnable",
)

T5 = dict(                          # google-t5/t5-11b, encoder only
    vocab_size=32128, d_model=1024, d_ff=65536, d_kv=128,
    num_heads=128, num_layers=24, feed_forward_proj="relu",
)

VAE = dict(                         # Wan2.1 VAE
    base_dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
    temperal_downsample=[False, True, True],
)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def hub_sizes():
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(REPO, files_metadata=True)
        return {s.rfilename: (s.size or 0) for s in info.siblings}
    except Exception as e:
        print(f"[warn] hub metadata unavailable: {e}")
        return {}


def main():
    from diffusers import AutoencoderKLWan, CosmosTransformer3DModel
    from transformers import T5Config, T5EncoderModel

    with init_empty_weights():
        dit = CosmosTransformer3DModel(**DIT)
        t5 = T5EncoderModel(T5Config(**T5))
        vae = AutoencoderKLWan(**VAE)

    sizes = hub_sizes()

    def disk(prefix):
        return sum(v for k, v in sizes.items() if k.startswith(prefix))

    rows = [
        ("text_encoder  (T5-11B enc)", n_params(t5), disk("text_encoder/")),
        ("transformer   (DiT)", n_params(dit), disk("transformer/")),
        ("vae           (Wan2.1)", n_params(vae), disk("vae/")),
    ]
    print(f"\n=== {REPO} ===")
    print(f"{'module':28s} {'params':>12s} {'ckpt':>10s} {'B/param':>9s}  dtype")
    tot_p = tot_d = 0
    for name, p, d in rows:
        tot_p += p
        tot_d += d
        bpp = d / p if p else 0
        dt = "bf16/fp16" if 1.9 < bpp < 2.1 else "fp32" if 3.9 < bpp < 4.1 else "?"
        print(f"{name:28s} {p/1e9:9.3f} B {d/1e9:7.2f} GB {bpp:8.2f}   {dt}")
    print(f"{'TOTAL':28s} {tot_p/1e9:9.3f} B {tot_d/1e9:7.2f} GB")
    print(f"\n'2B' 는 DiT 만: {n_params(dit)/1e9:.3f} B / "
          f"전체 {tot_p/1e9:.3f} B 중 {n_params(dit)/tot_p*100:.0f}%")

    # ---- DiT breakdown --------------------------------------------------- #
    blk = dit.transformer_blocks[0]
    nl = len(dit.transformer_blocks)
    groups = [
        ("attn1  self-attn (3D)", [blk.attn1]),
        ("attn2  cross-attn (text)", [blk.attn2]),
        ("ff     GELU MLP x4", [blk.ff]),
        ("norm1/2/3 adaLN-LoRA", [blk.norm1, blk.norm2, blk.norm3]),
    ]
    per_blk = sum(sum(n_params(m) for m in ms) for _, ms in groups)
    print(f"\n=== DiT block 1개 (총 {nl} block) ===")
    print(f"{'':28s} {'per block':>11s} {'share':>7s} {'x28':>11s}")
    for name, ms in groups:
        p = sum(n_params(m) for m in ms)
        print(f"  {name:26s} {p/1e6:8.2f} M {p/per_blk*100:6.1f}% {p*nl/1e6:9.1f} M")
    print(f"  {'합계':26s} {per_blk/1e6:8.2f} M {100.0:6.1f}% {per_blk*nl/1e6:9.1f} M")

    shared = [("patch_embed", dit.patch_embed), ("time_embed", dit.time_embed),
              ("learnable_pos_embed", dit.learnable_pos_embed),
              ("norm_out", dit.norm_out), ("proj_out", dit.proj_out)]
    print(f"\n=== block 밖 ===")
    for name, m in shared:
        if m is not None:
            print(f"  {name:26s} {n_params(m)/1e6:8.2f} M")

    # ---- LoRA target inventory ------------------------------------------- #
    print(f"\n=== LoRA 대상 모듈 (block당 / 전체 {nl} block) ===")
    targets = [
        ("attn1.to_q/k/v/out.0", [blk.attn1.to_q, blk.attn1.to_k, blk.attn1.to_v,
                                  blk.attn1.to_out[0]]),
        ("attn2.to_q", [blk.attn2.to_q]),
        ("attn2.to_k/to_v  (T5 -> hidden)", [blk.attn2.to_k, blk.attn2.to_v]),
        ("attn2.to_out.0", [blk.attn2.to_out[0]]),
        ("ff.net.0.proj / ff.net.2", [blk.ff.net[0].proj, blk.ff.net[2]]),
        ("norm{1,2,3}.linear_{1,2}", [blk.norm1.linear_1, blk.norm1.linear_2,
                                      blk.norm2.linear_1, blk.norm2.linear_2,
                                      blk.norm3.linear_1, blk.norm3.linear_2]),
    ]
    for name, ms in targets:
        p = sum(n_params(m) for m in ms)
        shp = " ".join(f"{tuple(m.weight.shape)}" for m in ms[:2])
        print(f"  {name:32s} {p/1e6:7.2f} M  {p*nl/1e6:8.1f} M   {shp}")

    # rank-16 LoRA cost, for placement budgeting
    print(f"\n=== rank-16 LoRA 파라미터 (A+B), block당 ===")
    for name, ms in targets:
        cost = sum(16 * (m.weight.shape[0] + m.weight.shape[1]) for m in ms)
        print(f"  {name:32s} {cost/1e3:7.1f} K  ({cost*nl/1e6:.2f} M for all {nl})")


if __name__ == "__main__":
    main()
