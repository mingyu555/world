#!/usr/bin/env python3
"""Stage 1 -- one LoRA per factor at its analysed layers, or one over everything.

  --factor A|G|D   trains on that factor's caption (base + the factor clause)
                   and is placed on the layers the analysis picked for it
  --factor AGD     trains on the full AGD caption; used with `--layers all` this
                   is the whole-model comparison the per-factor arrangement has
                   to beat

Objective.  The pipeline's preconditioning is rectified flow in disguise:
with t = sigma/(sigma+1) it feeds the transformer `(1-t)*x0 + t*n` and reads out
`x0_hat = (1-t)*x_t - t*F`.  Substituting gives the training target exactly

    F_target = n - x0        (plain velocity, uniform weighting)

so the loss below is standard flow matching and matches inference by
construction.  Latent frame 0 is the conditioning frame: it is replaced by the
clean conditioning latent, timestepped at `sigma_conditioning`, and excluded
from the loss -- the same treatment the video2world pipeline gives it.

Placement.  `--layers` takes the layer set for this factor (from
`f29/stats_xattn.json` / `causal_probe_*.json`).  LoRA is attached only to those
blocks, which is the claim being tested: analysis-guided placement beats
uniform or random placement (plan section 44).

  python train_lora.py --factor G --layers 7,8,10,13,15 --out runs/G
  python train_lora.py --factor D --layers all          --out runs/D --rank 8
"""
import argparse, json, math, random, sys, time
from pathlib import Path

import torch
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_same_noise import MODEL

# caption condition each run trains on, and the one substituted when the factor
# clause is dropped (a fraction of the time, so the adapter sees the factor
# being present and absent rather than only present)
FACTOR_COND = {"A": ("A", "base"), "G": ("G", "base"), "D": ("D", "base"),
               # the whole-caption run: every factor at once, dropped together
               "AGD": ("AGD", "base")}


class ClipSet(Dataset):
    def __init__(self, cache, ids, cond, neg_cond=None, p_uncond=0.1):
        self.c, self.ids, self.cond = Path(cache), ids, cond
        self.neg_cond, self.p_uncond = neg_cond, p_uncond

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        lat = torch.load(self.c / "latents" / f"{sid}.pt", map_location="cpu")
        txt = torch.load(self.c / "text" / f"{sid}.pt", map_location="cpu")
        cond = self.cond
        if self.neg_cond and random.random() < self.p_uncond:
            cond = self.neg_cond      # drop the factor phrase sometimes
        return (lat["z"][0].float(), lat["z_cond"][0].float(),
                txt["emb"][cond][0].float(), sid)


def build_lora(transformer, layers, rank, alpha, targets):
    from peft import LoraConfig, get_peft_model
    mods = [f"transformer_blocks.{l}.{t}" for l in layers for t in targets]
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=mods, init_lora_weights="gaussian")
    model = get_peft_model(transformer, cfg)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[lora] {len(layers)} layers x {len(targets)} modules  r={rank} "
          f"-> {n_tr/1e6:.2f} M trainable / {n_all/1e9:.2f} B")
    return model, n_tr


def sample_t(bs, device, sigma_min, sigma_max, mode="logitnormal"):
    t_lo = sigma_min / (sigma_min + 1)
    t_hi = sigma_max / (sigma_max + 1)
    if mode == "uniform":
        t = torch.rand(bs, device=device)
    else:                                     # logit-normal, SD3 style
        t = torch.sigmoid(torch.randn(bs, device=device))
    return t.clamp(t_lo, t_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", required=True, choices=["A", "G", "D", "AGD"])
    ap.add_argument("--layers", required=True, help="comma list, ranges, or 'all'")
    ap.add_argument("--cache", default="./cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default="attn2.to_q,attn2.to_k,attn2.to_v,attn2.to_out.0,ff.net.0.proj,ff.net.2")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--p_uncond", type=float, default=0.1)
    ap.add_argument("--t_mode", default="logitnormal", choices=["logitnormal", "uniform"])
    ap.add_argument("--val_frac", type=float, default=0.15,
                    help="0 when the official nuScenes split already separates "
                         "validation; a couple of clips are still held back so the "
                         "loss curve has something to report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--grad_ckpt", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    meta = json.loads((cache / "meta.json").read_text())

    from diffusers import CosmosTransformer3DModel
    tr = CosmosTransformer3DModel.from_pretrained(MODEL, subfolder="transformer",
                                                  torch_dtype=torch.bfloat16).to("cuda")
    tr.requires_grad_(False)
    n_layers = len(tr.transformer_blocks)
    layers = (list(range(n_layers)) if args.layers == "all"
              else sorted({int(x) for part in args.layers.split(",")
                           for x in (range(int(part.split('-')[0]), int(part.split('-')[1]) + 1)
                                     if '-' in part else [int(part)])}))
    targets = args.targets.split(",")
    model, n_tr = build_lora(tr, layers, args.rank, args.alpha, targets)
    if args.grad_ckpt:
        tr.enable_gradient_checkpointing()

    ids = meta["samples"]
    rng = random.Random(args.seed); rng.shuffle(ids)
    n_val = max(2, int(len(ids) * args.val_frac))
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    pos, neg = FACTOR_COND[args.factor]
    dl = DataLoader(ClipSet(cache, train_ids, pos, neg, args.p_uncond),
                    batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True)
    vdl = DataLoader(ClipSet(cache, val_ids, pos, None, 0.0), batch_size=args.batch)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.steps,
                                                pct_start=0.05)
    sigma_min, sigma_max = 0.002, 80.0
    t_cond = 1e-4 / (1e-4 + 1)

    (out / "config.json").write_text(json.dumps({
        **vars(args), "layers": layers, "targets": targets, "n_trainable": n_tr,
        "n_train": len(train_ids), "n_val": len(val_ids), "val_ids": val_ids,
        "data_meta": meta.get("note", ""),
    }, indent=1))

    def loss_on(z, zc, emb):
        z, zc, emb = z.cuda().float(), zc.cuda().float(), emb.cuda()
        B, C, T, H, W = z.shape
        t = sample_t(B, z.device, sigma_min, sigma_max, args.t_mode)
        tt = t.view(B, 1, 1, 1, 1)
        n = torch.randn_like(z)
        x_in = (1 - tt) * z + tt * n                       # rectified-flow interpolant
        target = n - z                                     # velocity

        ind = torch.zeros(1, 1, T, 1, 1, device=z.device)
        ind[:, :, :zc.shape[2]] = 1.0                      # frame 0 is the conditioning
        x_in = ind * zc + (1 - ind) * x_in
        ts = ind * t_cond + (1 - ind) * tt.expand(-1, 1, T, 1, 1)
        cmask = ind.expand(B, 1, T, H, W)
        pad = torch.zeros(1, 1, H * 8, W * 8, device=z.device, dtype=torch.bfloat16)

        F = model(hidden_states=x_in.to(torch.bfloat16),
                  timestep=ts.to(torch.bfloat16),
                  encoder_hidden_states=emb.to(torch.bfloat16),
                  fps=meta["backbone_fps"], condition_mask=cmask.to(torch.bfloat16),
                  padding_mask=pad, return_dict=False)[0].float()
        w = (1 - ind)                                      # no loss on the conditioning frame
        return ((F - target) ** 2 * w).sum() / (w.expand_as(F).sum() + 1e-8)

    print(f"[train] factor={args.factor} layers={layers} "
          f"train={len(train_ids)} val={len(val_ids)} steps={args.steps}")
    hist, step, t0 = [], 0, time.time()
    it = iter(dl)
    model.train()
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(args.accum):
            try:
                z, zc, emb, _ = next(it)
            except StopIteration:
                it = iter(dl); z, zc, emb, _ = next(it)
            l = loss_on(z, zc, emb) / args.accum
            l.backward()
            acc += l.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        opt.step(); sched.step(); step += 1
        hist.append({"step": step, "loss": acc, "grad_norm": gn, "lr": sched.get_last_lr()[0]})
        if step % args.log_every == 0:
            r = hist[-args.log_every:]
            print(f"  step {step:5d}/{args.steps}  loss {sum(x['loss'] for x in r)/len(r):.4f}  "
                  f"|g| {gn:.2f}  lr {sched.get_last_lr()[0]:.2e}  "
                  f"{(time.time()-t0)/step:.1f}s/step", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                torch.manual_seed(1234)
                vl = [loss_on(z, zc, e).item() for z, zc, e, _ in vdl]
            model.train()
            print(f"  [val] step {step}  loss {sum(vl)/len(vl):.4f}", flush=True)
            hist.append({"step": step, "val_loss": sum(vl) / len(vl)})
            model.save_pretrained(out / f"ckpt_{step}")
            (out / "history.json").write_text(json.dumps(hist, indent=1))
    model.save_pretrained(out / "final")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"[done] {out}  {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
