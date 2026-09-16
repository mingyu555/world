"""Shared Cosmos-Predict2-2B plumbing (self-contained; weights read from HF_HOME)."""

import os

import torch

# Repo id, or a local snapshot directory. A read-only HF cache can make
# huggingface_hub fail when it tries to take a lock, so pointing COSMOS_MODEL at
# the snapshot directory itself is the escape hatch (env.sh resolves it).
MODEL = os.environ.get("COSMOS_MODEL") or "nvidia/Cosmos-Predict2-2B-Video2World"

# Cosmos default negative prompt (used by the reference pipeline and by
# cosmos_exp/run_same_noise.py; kept identical so runs are comparable).
NEG = (
    "The video captures a series of frames showing ugly scenes, static with no motion, "
    "motion blur, over-saturation, shaky footage, low resolution, grainy texture, "
    "pixelated images, poorly lit areas, underexposed and overexposed scenes, poor "
    "color balance, washed out colors, choppy sequences, jerky movements, low frame "
    "rate, artifacting, color banding, unnatural transitions, outdated special "
    "effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, "
    "visual noise, and flickering. Overall, the video is of poor quality."
)

# AGD minus one factor -> the caption combination that remains
REMOVE = {"A": "GD", "G": "AD", "D": "AG"}


def build_pipe(dtype=torch.bfloat16, offload=False, vae_tiling=False, device="cuda"):
    from diffusers import Cosmos2VideoToWorldPipeline

    pipe = Cosmos2VideoToWorldPipeline.from_pretrained(MODEL, torch_dtype=dtype)
    # diffusers >= 0.35: `_execution_device` can raise AttributeError behind the
    # property; pin it so accelerate hooks do not shadow it.
    try:
        _ = pipe._execution_device
    except Exception:
        type(pipe)._execution_device = property(lambda self: torch.device(device))
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if vae_tiling:
        try:
            pipe.vae.enable_tiling()
        except Exception as e:  # pragma: no cover
            print(f"[warn] vae tiling unavailable: {e}")
    return pipe


def latent_shape(pipe, batch, frames, height, width):
    """Match Cosmos2VideoToWorldPipeline.prepare_latents exactly."""
    channels = pipe.transformer.config.in_channels - 1   # last channel = cond mask
    ct = pipe.vae_scale_factor_temporal
    cs = pipe.vae_scale_factor_spatial
    return (batch, channels, (frames - 1) // ct + 1, height // cs, width // cs)


def factor_phrases(meta, text_field):
    """The verbatim A/G/D phrases inside the AGD caption of one agd_dataset sample."""
    ph = meta["phrases"]
    return {
        "A": ph["A"],
        "G": ph["G"],
        "D": ph["D_gt"] if text_field == "text_gtD" else ph["D_qwen"],
    }
