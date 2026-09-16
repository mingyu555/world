#!/usr/bin/env python3
"""Minimal-pair probe: one base prompt, one factor phrase swapped in.

The dataset's captions differ in length (A 14 tokens, G 17, D 14), in syntax, and
even in subject (24 of 100 base captions are not about the ego vehicle). Any
per-layer difference between them mixes the factor with all of that. This probe
removes the confound: a single fixed base prompt, and one short phrase appended
in the same slot.

    base    The vehicle drives along the road.
    +A      The vehicle drives along the road, in the rain.
    +G      The vehicle drives along the road, in the left lane.
    +D      The vehicle drives along the road, braking hard.
    +filler The vehicle drives along the road, in the video.

Four phrases per factor, not one, so the question becomes "which layers respond
to *any* appearance phrase but not to geometry or dynamics phrases" rather than
"which layers respond to this particular word". The **filler** group carries no
factor content and is syntactically identical to the A group ("in the ..."), so
it measures what a same-shaped, same-length, meaningless addition does -- the
prompt-length control of plan section 10, which has never been run.

Measurement, at the reference latent z_t of the *base* trajectory (teacher
forced, so nothing accumulates):

    delta_p(l, b)         what block l branch b writes under prompt p
    effect(p, l, b)  = || delta_p(l,b) - delta_base(l,b) || / || h_l ||
    v_p              = f(z_t, p)            the model output, for the Gram matrix

One capture of the base stream per step, then one forward per prompt compared
against it -- the comparison happens inside the hook, so no tensor is copied.

Usage
-----
  source env.sh
  CUDA_VISIBLE_DEVICES=1 $PY minimal_pair_probe.py --out ./runs/minpair
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from block_patch import BRANCHES, BlockPatcher
from cosmos_common import NEG, build_pipe, latent_shape

BASE = "The vehicle drives along the road."

# Every phrase adds exactly 4 T5 tokens, verified with the tokenizer. The first
# run used 6-token geometry phrases ("in the left lane"), and 7 of the 9 cells
# where geometry led turned out to track token count (r = 0.51-0.86) rather than
# geometry -- so the counts have to match before the test means anything.
PHRASES = {
    "A": ["in the rain", "in the fog", "in the dark", "in the snow"],
    "G": ["on the left", "on the right", "at the junction", "toward the exit"],
    "D": ["braking hard", "speeding up", "slowing down", "turning sharply"],
    "filler": ["in the video", "in the clip", "in the frame", "in the shot"],
}


def build_prompts(base=BASE, phrases=PHRASES):
    out = [("base", "base", base)]
    for grp, ws in phrases.items():
        for w in ws:
            out.append((grp, w, base[:-1] + ", " + w + "."))
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./runs/minpair")
    ap.add_argument("--image", default=None,
                    help="conditioning image (default: a clean agd_dataset sample)")
    ap.add_argument("--step_every", type=int, default=4)
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_device", default="cuda")
    ap.add_argument("--tag", default="minpair")
    ap.add_argument("--base", default=BASE, help="base prompt (must end with '.')")
    ap.add_argument("--phrases", default=None, help="json file overriding PHRASES")
    ap.add_argument("--graft", action="store_true",
                    help="force every block's input to the reference stream's, so the "
                         "measured write difference is local to that block")
    return ap.parse_args()


def main():
    args = parse_args()
    import os
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Path(args.image or (Path(os.environ["AGD"]) / "scene-0626_f24" / "cond_image.jpg"))
    image = Image.open(img).convert("RGB")
    phrases = json.loads(Path(args.phrases).read_text()) if args.phrases else PHRASES
    base = args.base
    prompts = build_prompts(base, phrases)

    pipe = build_pipe()
    n_layers = len(pipe.transformer.transformer_blocks)
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    device = torch.device("cuda")
    print(f"[model] {n_layers} blocks   latent {tuple(shape)}   image {img.name}")

    # ---- token accounting: how many T5 tokens does each phrase add? -------- #
    tok = pipe.tokenizer
    n_base = int(tok(base, return_tensors="pt")["attention_mask"].sum())
    print(f"\n[tokens] base '{base}' = {n_base}")
    added = {}
    for grp, word, p in prompts[1:]:
        n = int(tok(p, return_tensors="pt")["attention_mask"].sum())
        added[word] = n - n_base
        print(f"  +{grp:6s} {word:20s} -> {n:3d} tokens  (+{n - n_base})")
    by_grp = {g: [added[w] for w in phrases[g]] for g in phrases}
    print("  그룹별 추가 토큰: " + "  ".join(f"{g}={v}" for g, v in by_grp.items()))

    with torch.no_grad():
        emb = {word: pipe._get_t5_prompt_embeds(p, device=device)
               for _, word, p in prompts}

    zpath = out / "z_T_shared.pt"
    if zpath.exists():
        z0 = torch.load(zpath)
        assert tuple(z0.shape) == tuple(shape)
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        z0 = torch.randn(shape, generator=g, dtype=torch.float32)
        torch.save(z0, zpath)

    patcher = BlockPatcher(pipe.transformer, cache_device=args.cache_device)
    patcher.attach()
    probe_steps = list(range(0, args.steps, args.step_every))
    variants = [(g, w) for g, w, _ in prompts if g != "base"]
    print(f"\n[plan] {len(variants)} variants x {len(probe_steps)} steps "
          f"= {len(variants) * len(probe_steps)} forwards (+1 capture per step)")

    orig = pipe.transformer.forward
    state = {"step": 0, "call": 0, "busy": False}
    eff = {}       # word -> {(l,b): [per step]}
    vdiff = {}     # word -> [flat output difference per step]
    hn = []
    controls = []

    def flat(r):
        return (r[0] if isinstance(r, (tuple, list)) else r.sample).float().flatten()

    def probe(kw):
        t0 = time.time()
        kwb = dict(kw); kwb["encoder_hidden_states"] = emb["base"]
        patcher.graft = {}
        with patcher.capture():
            v_base = flat(orig(**kwb))
        ref = dict(patcher.store)
        h = dict(getattr(patcher, "h_in_norm", patcher.h_norm))
        hn.append(h)
        if args.graft:
            patcher.graft = dict(patcher.h_in)
            # control: the base prompt, grafted onto its own stream, must reproduce
            # itself exactly -- every write difference has to be 0
            with patcher.comparison(ref):
                _ = orig(**kwb)
            worst = max(patcher.diff.values()) if patcher.diff else 0.0
            controls.append(worst)
        for grp, word in variants:
            kwv = dict(kw); kwv["encoder_hidden_states"] = emb[word]
            with patcher.comparison(ref):
                v = flat(orig(**kwv))
            d = patcher.diff
            e = eff.setdefault(word, {})
            for (l, b), val in d.items():
                e.setdefault((l, b), []).append(val / max(h.get(l, 1.0), 1e-8))
            vdiff.setdefault(word, []).append((v - v_base).cpu())
        patcher.graft = {}
        patcher.clear()
        ctl = f"  graft-null={controls[-1]:.2e}" if controls else ""
        print(f"  [minpair] step {state['step']:3d}  {len(variants)} variants "
              f"in {time.time() - t0:.0f}s{ctl}", flush=True)

    def fwd(*a, **kw):
        v = orig(*a, **kw)
        if state["busy"]:
            return v
        if state["call"] == 0 and state["step"] in probe_steps:
            state["busy"] = True
            try:
                with torch.no_grad():
                    probe(kw)
            finally:
                state["busy"] = False
        state["call"] += 1
        return v

    pipe.transformer.forward = fwd

    def cb(p, i, t, kwargs):
        state["step"] += 1
        state["call"] = 0
        return kwargs

    t0 = time.time()
    with torch.no_grad():
        pipe(image=image, prompt=base, negative_prompt=NEG,
             height=args.height, width=args.width, num_frames=args.frames,
             num_inference_steps=args.steps, guidance_scale=args.guidance,
             fps=args.fps, generator=torch.Generator(device="cpu").manual_seed(args.seed),
             latents=z0.clone().to(device, dtype=torch.bfloat16),
             output_type="latent",
             callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    pipe.transformer.forward = orig
    patcher.detach()

    # output-space Gram between every pair of variants, per step
    words = [w for _, w in variants]
    gram = []
    for si in range(len(probe_steps)):
        M = torch.stack([vdiff[w][si] for w in words])
        M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-20)
        gram.append((M @ M.T).numpy().tolist())

    rec = {
        "meta": {"base": base, "phrases": phrases, "added_tokens": added,
                 "n_layers": n_layers, "branches": list(BRANCHES),
                 "probe_steps": probe_steps, "words": words,
                 "groups": [g for g, _ in variants],
                 "latent_shape": list(shape), "image": str(img),
                 "graft": args.graft, "graft_null": controls,
                 "config": {k: v for k, v in vars(args).items()}},
        "effect": {w: {f"{l}|{b}": v for (l, b), v in e.items()} for w, e in eff.items()},
        "vnorm": {w: [float(x.norm()) for x in vdiff[w]] for w in words},
        "gram": gram,
    }
    fp = out / f"minpair_{args.tag}.pt"
    torch.save(rec, fp)
    print(f"\n[save] {fp}   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
