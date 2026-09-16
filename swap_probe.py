#!/usr/bin/env python3
"""B-LoRA's block analysis done properly: swap the factor phrase, do not add it.

`layer_prompt_image_probe.py` compared the base caption against base + a factor
phrase. That is not what B-LoRA did, and the difference matters:

    B-LoRA          "a [bunny] sitting"  ->  "a [tiger] sitting"
                    same length, one content word, drastically different content

    what we had     "The vehicle continues straight."
                 -> "The vehicle continues straight on a clear day with dry
                     roads and bright lighting."
                    +10 words (4 -> 15), and the addition *elaborates* rather
                    than contradicts -- the base never claimed a weather, and
                    the model's default output is already a clear day, so there
                    is nothing for the prompt to change

That explains the earlier result rather than contradicting it: only dynamics
showed an effect (10 of 59 samples) and dynamics is the one factor whose phrase
does contradict the base -- "continues straight" against "speeds up and turns
left". Appearance and geometry never conflicted with anything, so 56 of 59
samples had nothing to localise.

This probe fixes the design. The reference caption is the sample's own
single-factor caption, so it already commits to a value of that factor; the
injected caption is the same sentence with that phrase replaced by another
sample's phrase for the same slot:

    reference  "The truck continues straight on a sunny day with clear skies
                and dry road conditions."
    injected   "The truck continues straight on a clear night with dry road
                surfaces and good lighting."

The partner is chosen from the dataset's own inventory for that slot (24 unique
appearance phrases, 47 geometry, 13 dynamics), as the one with the lowest SigLIP
text similarity to the original among candidates within a word-count tolerance --
so the swap is as semantically opposed as the data allows while staying
length-matched, with no hand-written antonyms.

The B-LoRA statistic is then directly computable, because there are now two
competing captions rather than one caption and its extension:

    d_toward = [sim(V, p_inj) - sim(V, p_ref)] - [sim(V0, p_inj) - sim(V0, p_ref)]

positive means injecting at that window moved the video toward the swapped
phrase. The factor-matched pixel metrics from `layer_prompt_image_probe` are
reported alongside.

    source env.sh
    # ceiling screen: does swapping the phrase change anything at all?
    CUDA_VISIBLE_DEVICES=0 $PY swap_probe.py --factor A --windows "" --out ./runs/swapscreen
    # window grid on the samples where it does
    CUDA_VISIBLE_DEVICES=0 $PY swap_probe.py --factor A --only sceneX,sceneY --out ./runs/swapgrid
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from block_patch import BlockPatcher
from cosmos_common import NEG, build_pipe, factor_phrases, latent_shape
from layer_prompt_image_probe import SigLIP, flow_fields, lum, video_metrics
from lora_train import clean_samples

WINDOWS = ["0-3", "4-7", "8-11", "12-15", "16-19", "20-23", "24-27"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True, choices=["A", "G", "D"])
    ap.add_argument("--out", default="./runs/swap")
    ap.add_argument("--agd", default=None,
                    help="dataset root. agd_dataset (2 fps keyframes) by default; "
                         "point at wm_dataset for the 12 Hz / 29 frame clips the "
                         "experts are actually trained on -- the two differ in "
                         "frame rate and horizon, which is the domain gap that "
                         "made the first placement analysis hard to act on.")
    ap.add_argument("--text_field", default="text_gtD", choices=["text", "text_gtD"])
    ap.add_argument("--only", default=None, help="comma list of sample names")
    ap.add_argument("--samples", type=int, default=0, help="0 = all clean samples")
    ap.add_argument("--sample_offset", type=int, default=0)
    ap.add_argument("--windows", default=",".join(WINDOWS),
                    help="'' for the ceiling screen (all 28 blocks only)")
    ap.add_argument("--pairs", default=None,
                    help="JSON file with {'pairs': [[ref_phrase, inj_phrase], ...]}, "
                         "replacing the dataset inventory. The reference caption "
                         "becomes the sample's base caption plus ref_phrase and the "
                         "injected one base plus inj_phrase, so the swap stays a "
                         "length-matched minimal pair -- but the phrases can be "
                         "written to describe something the conditioning image "
                         "cannot fix. Pairs are assigned to samples round-robin.")
    ap.add_argument("--tol", type=int, default=3,
                    help="allowed word-count difference for the partner phrase")
    ap.add_argument("--frames", type=int, default=13)
    ap.add_argument("--height", type=int, default=704)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_ceiling", action="store_true")
    ap.add_argument("--no_cond", action="store_true",
                    help="drop the image conditioning. The pipeline hard-overwrites "
                         "the first latent frame with the conditioning image at "
                         "every step (lines 704/719), which pins appearance and "
                         "geometry no matter what the caption says -- measured: "
                         "frame 0's luminance change under a day->night swap is "
                         "exactly 0. Zeroing cond_indicator turns the run into "
                         "text-to-video, so the caption is the only thing "
                         "specifying the scene.")
    ap.add_argument("--step_bands", default=None,
                    help="comma list of denoising-step ranges to confine the "
                         "injection to, e.g. '0-6,7-13,14-19'. Everything measured "
                         "so far injected for the whole trajectory, so a window's "
                         "effect is summed over all steps; this asks *when* the "
                         "text acts. AC3D found camera motion is settled in the "
                         "first 40% of the trajectory and Characterizing Motion "
                         "Encoding puts the motion/appearance boundary around "
                         "t=700-950, so appearance and dynamics are expected to "
                         "split here as well as by layer.")
    ap.add_argument("--siglip", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--tag", default="swap")
    return ap.parse_args()


CONNECTORS = ("on a ", "on the ", "on ", "in a ", "in the ", "in ",
              "under a ", "under the ", "under ", "with a ", "with ", "at a ",
              "at the ", "at ", "from a ", "from the ", "from ")


def split_connector(p):
    """('on a ', 'clear day with dry roads') -- the inventory is inconsistent about
    whether the leading preposition belongs to the phrase, so swapping raw phrases
    produced captions like "drives forward rainy night". Keying the inventory on
    the connector-free core and re-attaching the *original's* connector keeps the
    swapped sentence grammatical and makes the word-count match honest."""
    for c in CONNECTORS:
        if p.lower().startswith(c):
            return p[:len(c)], p[len(c):]
    return "", p


def choose_partners(phrases, sig, tol):
    """For each phrase core, the length-matched inventory core least like it.

    Length matching first (a longer replacement would reintroduce the token-count
    confound the previous design suffered from), then minimum SigLIP text cosine
    among what survives. The tolerance widens only if nothing qualifies.
    """
    cores = {}
    for p in set(phrases):
        pre, core = split_connector(p)
        cores.setdefault(core, pre)
    uniq = sorted(cores)
    emb = sig.text(uniq).cpu().numpy()
    idx = {c: i for i, c in enumerate(uniq)}
    wc = {c: len(c.split()) for c in uniq}
    out = {}
    for p in set(phrases):
        pre, core = split_connector(p)
        for t in (tol, tol * 2, tol * 4, 10 ** 6):
            cand = [q for q in uniq if q != core and abs(wc[q] - wc[core]) <= t]
            if cand:
                break
        sims = [float(emb[idx[core]] @ emb[idx[q]]) for q in cand]
        j = int(np.argmin(sims))
        out[p] = {"old": core, "partner": pre + cand[j], "core": cand[j],
                  "cos": sims[j], "words": (wc[core], wc[cand[j]]),
                  "n_cand": len(cand)}
    return out


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.agd or os.environ["AGD"])
    device = torch.device("cuda")
    f = args.factor

    if (root / "clips.jsonl").exists():
        # wm_dataset: the builder already verified that each factor phrase occurs
        # exactly once in every combination caption, and records it as span_ok
        samples = [d for d in sorted(root.iterdir())
                   if (d / "caption.json").exists()
                   and json.loads((d / "caption.json").read_text()).get("span_ok")]
        print(f"[dataset] wm_dataset, {len(samples)} clips with span_ok")
    else:
        samples = clean_samples(root, args.text_field)
    meta = {s.name: json.loads((s / "caption.json").read_text()) for s in samples}
    pairs = ph = None
    if args.pairs:
        pairs = json.loads(Path(args.pairs).read_text())["pairs"]
        for x, y in pairs:
            assert len(x.split()) == len(y.split()), (x, y)
        usable = samples
        print(f"[data] {len(samples)} clean, {len(pairs)} hand-written {f} pairs "
              f"(word-matched), assigned round-robin")
    else:
        ph = {s.name: factor_phrases(meta[s.name], args.text_field)[f]
              for s in samples}
        # the phrase has to be a substring of the caption for the swap to be a swap
        usable = [s for s in samples if ph[s.name] and
                  ph[s.name] in meta[s.name]["captions"][f][args.text_field]]
        print(f"[data] {len(samples)} clean, {len(usable)} with a substitutable "
              f"{f} phrase")

    if args.only:
        want = [w for w in args.only.split(",") if w]
        by = {s.name: s for s in usable}
        miss = [w for w in want if w not in by]
        assert not miss, f"not usable for {f}: {miss}"
        picked = [by[w] for w in want]
    else:
        picked = usable[args.sample_offset:]
        if args.samples:
            picked = picked[:args.samples]
    assert picked

    windows = [w for w in args.windows.split(",") if w]
    base_sites = [(w, list(range(int(w.split("-")[0]), int(w.split("-")[1]) + 1)))
                  for w in windows]
    if not args.no_ceiling:
        base_sites.append(("all", list(range(28))))
    if args.step_bands:
        bands = []
        for b_ in args.step_bands.split(","):
            lo, hi = b_.split("-")
            bands.append((b_, (int(lo), int(hi))))
        sites = [(f"{n}@s{bn}", ls, bd) for n, ls in base_sites for bn, bd in bands]
    else:
        sites = [(n, ls, None) for n, ls in base_sites]
    n_gen = len(picked) * (1 + len(sites))
    print(f"[plan] {len(picked)} samples x (1 reference + {len(sites)} sites) "
          f"= {n_gen} generations")

    pipe = build_pipe()
    tfm = pipe.transformer
    shape = latent_shape(pipe, 1, args.frames, args.height, args.width)
    patcher = BlockPatcher(tfm)
    patcher.attach()
    orig_fwd = tfm.forward
    gate = {"pos": None, "map": {}, "band": None, "step": 0}

    def fwd(*a, **kw):
        e = kw.get("encoder_hidden_states")
        if gate["pos"] is None:
            gate["pos"] = e
        on = bool(gate["map"]) and e is gate["pos"]
        if on and gate["band"] is not None:
            lo, hi = gate["band"]
            on = lo <= gate["step"] <= hi
        patcher.text = gate["map"] if on else {}
        patcher.enabled = on
        try:
            return orig_fwd(*a, **kw)
        finally:
            patcher.enabled = False
            patcher.text = {}

    tfm.forward = fwd

    if args.no_cond:
        _orig_prep = pipe.prepare_latents

        def prep(*a, **kw):
            lat, init, ci, ui, cm, um = _orig_prep(*a, **kw)
            ci = torch.zeros_like(ci)
            cm = torch.zeros_like(cm)
            if ui is not None:
                ui = torch.zeros_like(ui)
                um = torch.zeros_like(um)
            return lat, init, ci, ui, cm, um

        pipe.prepare_latents = prep
        print("[no_cond] image conditioning removed (cond_indicator = 0)")

    sig = SigLIP(args.siglip, device)
    part = pcos = None
    if pairs is None:
        part = choose_partners([ph[s.name] for s in usable], sig, args.tol)
        print(f"[partners] {len(part)} unique {f} phrases; "
              f"cos range {min(v['cos'] for v in part.values()):.3f}..."
              f"{max(v['cos'] for v in part.values()):.3f}")
    else:
        pe = sig.text([x for pr in pairs for x in pr]).cpu().numpy()
        pcos = [float(pe[2 * i] @ pe[2 * i + 1]) for i in range(len(pairs))]
        print("[pairs] SigLIP cos per pair: " + "  ".join(f"{c:.3f}" for c in pcos))

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    z0 = torch.randn(shape, generator=g, dtype=torch.float32)

    def generate(image, cap, override=None, layers=None, band=None):
        gate["map"], gate["pos"] = {}, None
        gate["band"], gate["step"] = band, 0

        def step_cb(pipeline, i, t, kwargs):
            gate["step"] = i + 1
            return kwargs

        if override is not None:
            with torch.no_grad():
                emb = pipe._get_t5_prompt_embeds(override, device=device)
            gate["map"] = {l: emb for l in layers}
        with torch.no_grad():
            res = pipe(image=image, prompt=cap, negative_prompt=NEG,
                       height=args.height, width=args.width,
                       num_frames=args.frames, num_inference_steps=args.steps,
                       guidance_scale=args.guidance, fps=args.fps,
                       generator=torch.Generator(device="cpu").manual_seed(args.seed),
                       latents=z0.clone().to(device, dtype=torch.bfloat16),
                       output_type="np", callback_on_step_end=step_cb,
                       callback_on_step_end_tensor_inputs=["latents"])
        gate["map"], gate["pos"] = {}, None
        gate["band"] = None
        return np.asarray(res.frames[0], dtype=np.float32)

    order = {x.name: i for i, x in enumerate(usable)}
    rows = []
    t_all = time.time()
    for si, s in enumerate(picked):
        m = meta[s.name]
        if pairs is not None:
            # Index the pair by the sample's position in the full clean list, not
            # by its position in this shard or in --only. Otherwise the same
            # sample gets a different pair depending on how the run was sharded,
            # and a ceiling measured in one run cannot normalise a window grid
            # measured in another.
            gi = order[s.name]
            core, new = pairs[gi % len(pairs)]
            base = m["captions"]["base"][args.text_field].rstrip(". ")
            p_ref = f"{base} {core}."
            p_inj = f"{base} {new}."
            pc = pcos[gi % len(pairs)]
            words = (len(core.split()), len(new.split()))
        else:
            p_ref = m["captions"][f][args.text_field]
            old = ph[s.name]
            core = part[old]["old"]
            new = part[old]["core"]
            # replace the connector-free core, keeping the sentence's preposition
            assert core in p_ref, (core, p_ref)
            p_inj = p_ref.replace(core, new)
            pc = part[old]["cos"]
            words = part[old]["words"]
        image = Image.open(s / "cond_image.jpg").convert("RGB")
        print(f"\n[{si+1}/{len(picked)}] {s.name}   cos(ref,inj)={pc:.3f}  "
              f"words {words}")
        print(f"  ref: {p_ref}")
        print(f"  inj: {p_inj}")

        V0 = generate(image, p_ref)
        flow0 = flow_fields(V0)
        tv = sig.text([p_ref, p_inj])
        s0 = (sig.video(V0) @ tv.T)[0].tolist()
        margin0 = s0[1] - s0[0]
        print(f"  reference: lum {lum(V0).mean():.4f}   "
              f"sim(ref)={s0[0]:+.4f} sim(inj)={s0[1]:+.4f} margin={margin0:+.4f}")

        for name, layers, band in sites:
            t0 = time.time()
            V = generate(image, p_ref, p_inj, layers, band)
            mt, _ = video_metrics(V, V0, flow0)
            s1 = (sig.video(V) @ tv.T)[0].tolist()
            mt["d_toward"] = (s1[1] - s1[0]) - margin0
            # Video2World overwrites the first latent frame with the conditioning
            # image at every step, so appearance may be pinned there and free to
            # drift later. Split the luminance change by frame to see which.
            dl = np.abs(lum(V).mean((1, 2)) - lum(V0).mean((1, 2)))
            mt["d_lum_f0"] = float(dl[0])
            mt["d_lum_last"] = float(dl[-1])
            mt["d_lum_slope"] = float(np.polyfit(np.arange(len(dl)), dl, 1)[0])
            mt["sim_ref"] = s1[0]
            mt["sim_inj"] = s1[1]
            mt.update({"sample": s.name, "site": name, "factor": f,
                       "phrase_old": core, "phrase_new": new,
                       "phrase_cos": pc,
                       "phrase_core_old": core, "phrase_core_new": new,
                       "words": list(words),
                       "n_layers": len(layers),
                       "band": list(band) if band else None,
                       "secs": time.time() - t0})
            rows.append(mt)
            print(f"    {name:6s} toward {mt['d_toward']:+.4f}  "
                  f"lum {mt['d_lum']:.4f}  hist {mt['d_hist']:.4f}  "
                  f"struct {mt['d_struct']:.4f}  flow {mt['d_flow']:.4f}  "
                  f"dir {mt['d_flowdir']:5.2f}  ({mt['secs']:.0f}s)", flush=True)
            (out / f"swap_{f}_{args.tag}.json").write_text(json.dumps(
                {"meta": {"factor": f, "config": {k: v for k, v in vars(args).items()},
                          "samples": [p.name for p in picked],
                          "sites": [x[0] for x in sites]}, "rows": rows}))

    tfm.forward = orig_fwd
    patcher.detach()
    print(f"\n[save] {out / f'swap_{f}_{args.tag}.json'}   {len(rows)} cells   "
          f"{time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
