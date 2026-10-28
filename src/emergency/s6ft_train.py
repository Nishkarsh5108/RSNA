"""s6ft: warm-start fine-tune of nartaa's s6fall (CC BY-NC 4.0) on the multi-teacher targets (plan section 13).

Model: heliosli's S6 Net (compact_train.py, the class the inference stage loads), convnext_small + spatial attention, loaded
from E_prior_s42_fall.pt (strict). Stem and stages 0-1 frozen; stages 2-3 checkpointed per block (use_reentrant=False, so
blocks fed by the frozen stages still get gradients); state-dict keys unchanged, so the output drops into s6fall's slot.
Data: the compact cache (s6ft_cache.py: 64 windows per study, JPEG q90, 320 px) and targets (s6ft_targets.py); random
288 crop per batch as in nartaa's trainer; gold_58 is not in either file.
Loss: per finding alpha_c (0.5 soft BCE + 0.5 soft pairwise ranking against a queue of recent detached logits),
alpha 1.6 on Lateral OA, PF OA, Synovitis, Lateral Meniscus, Fracture, else 1.0 (--alpha uniform: 1.0 everywhere);
mean over findings weighted by alpha.
Optimiser: AdamW (encoder 3e-5, rest 1e-4, wd 0.05), 5 % warm-up then cosine to 0, clip 5, EMA 0.995 (EMA weights are the
model; nothing is selected on gold). fp16 + GradScaler on GPUs without bf16 (T4), bf16 otherwise.
Single GPU:  python s6ft_train.py --cache <dir> --targets <csv> --init <s6fall.pt> --code <s6 src dir> --out <dir>
Two GPUs:    torchrun --nproc_per_node=2 s6ft_train.py ...  (DDP, one process per GPU)
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
ALPHA = torch.tensor([1.6 if c in ("Lateral OA", "PF OA", "Synovitis", "Lateral Meniscus", "Fracture") else 1.0 for c in LAB])


class CacheDS(torch.utils.data.Dataset):
    def __init__(self, files, tgt, windows, seed):
        self.files, self.tgt, self.windows, self.seed, self.epoch = files, tgt, windows, seed, 0

    def __len__(self):
        return len(self.files)

    def _open(self, f):
        if "::" not in f:
            return np.load(f)
        import io, zipfile
        zp, member = f.split("::", 1)
        zs = self.__dict__.setdefault("_zips", {})
        if zp not in zs:
            zs[zp] = zipfile.ZipFile(zp)
        return np.load(io.BytesIO(zs[zp].read(member)))

    def __getitem__(self, i):
        import cv2
        z = self._open(self.files[i])
        uid, jpg, off, meta = str(z["uid"]), z["jpg"], z["off"], z["meta"]
        k = len(meta)
        rng = np.random.default_rng((self.seed, self.epoch, i))
        sel = np.sort(rng.choice(k, self.windows, replace=False)) if k > self.windows else np.arange(k)
        img = np.empty((len(sel), 3, 320, 320), np.uint8)
        for a, w in enumerate(sel):
            for ch in range(3):
                q = 3 * w + ch
                img[a, ch] = cv2.imdecode(jpg[off[q]:off[q + 1]], cv2.IMREAD_GRAYSCALE)
        return torch.from_numpy(img), torch.from_numpy(meta[sel].astype(np.float32)), torch.from_numpy(self.tgt[uid]), i


def collate(batch):
    W = max(len(b[0]) for b in batch)
    img = torch.zeros((len(batch), W) + tuple(batch[0][0].shape[1:]), dtype=torch.uint8)
    meta = torch.zeros((len(batch), W, batch[0][1].shape[-1]))
    mask = torch.zeros((len(batch), W), dtype=torch.bool)
    for j, (x, m, _, _) in enumerate(batch):
        img[j, :len(x)], meta[j, :len(x)], mask[j, :len(x)] = x, m, True
    return img, meta, mask, torch.stack([b[2] for b in batch]), torch.tensor([b[3] for b in batch])


def add_ckpt(mod):
    f = mod.forward

    def g(*a):
        if torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(f, *a, use_reentrant=False)
        return f(*a)
    mod.forward = g


def pair_loss(z, t, qz, qt):
    """soft pairwise ranking: every batch study vs the batch and the queue; pair weight |t_i - t_j|, per finding."""
    Z, T = torch.cat([z, qz]), torch.cat([t, qt])
    d = z[:, None, :] - Z[None, :, :]                    # [b, b+q, C]
    w = t[:, None, :] - T[None, :, :]
    s, w = torch.sign(w), w.abs()
    loss = (F.softplus(-s * d) * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1e-6)
    return loss                                           # [C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--code", required=True, help="folder with heliosli's compact_train.py (S6 Net)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--queue", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0, help="throughput test: stop after this many optimizer steps")
    ap.add_argument("--max-hours", type=float, default=11.0, help="save the EMA and stop when this much wall time has passed")
    ap.add_argument("--mem-fraction", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--alpha", default="brief", choices=["brief", "uniform"],
                    help="brief: 1.6 on the five findings above (s6ft v1); uniform: 1.0 everywhere (plan 14 C, s6ft_v2)")
    a = ap.parse_args()
    t_start = time.time()
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    if ddp:
        torch.distributed.init_process_group("nccl")
    lr_ = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(lr_)
    dev = torch.device("cuda", lr_)
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, lr_)
    torch.manual_seed(a.seed + rank)
    sys.path.insert(0, a.code)
    import compact_train as CT
    assert CT.LABELS == LAB
    import pandas as pd
    tg = pd.read_csv(a.targets, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    tgt = {u: v.astype(np.float32) for u, v in zip(tg.index, tg.values)}
    files = sorted(glob.glob(os.path.join(a.cache, "**", "*.npz"), recursive=True))
    if not files:                                     # the dataset kept the uploaded folder as a zip archive
        import zipfile
        for zp in sorted(glob.glob(os.path.join(a.cache, "**", "*.zip"), recursive=True)):
            files += [f"{zp}::{m}" for m in zipfile.ZipFile(zp).namelist() if m.endswith(".npz")]
    probe = CacheDS(files, {}, a.windows, a.seed)
    uid_of = {f: str(probe._open(f)["uid"]) for f in files}
    files = [f for f in files if uid_of[f] in tgt]
    if rank == 0:
        print(f"s6ft: {len(files)} cached studies with targets (of {len(tg)} targets)", flush=True)
    ds = CacheDS(files, tgt, a.windows, a.seed)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=a.seed, drop_last=True) if ddp else None
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=sampler is None, sampler=sampler, num_workers=a.workers,
                                     collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0,
                                     prefetch_factor=2 if a.workers else None)
    model = CT.Net("convnext_small", None, freeze_blocks=16, crop=a.crop, state_spec=None, in_chans=3, spatial=True)
    sd = torch.load(a.init, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    for n, p in model.named_parameters():
        if n.startswith(("enc.stem", "enc.stages.0.", "enc.stages.1.")):
            p.requires_grad_(False)
    for s in (2, 3):
        for blk in model.enc.stages[s].blocks:
            add_ckpt(blk)
    model = model.to(dev).to(memory_format=torch.channels_last)
    model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if not n.startswith("enc.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr}, {"params": rest, "lr": a.head_lr}], weight_decay=a.wd)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    ema = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[lr_]) if ddp else model
    steps_per_epoch = len(dl) // a.accum
    total = max(1, int(round(a.epochs * steps_per_epoch))) if not a.max_steps else a.max_steps
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(dev)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    alpha = (ALPHA if a.alpha == "brief" else torch.ones(len(LAB))).to(dev)
    qz = torch.zeros((0, 12), device=dev)
    qt = torch.zeros((0, 12), device=dev)
    gen = torch.Generator().manual_seed(a.seed + 17 * rank)
    if rank == 0:
        print(f"s6ft: {steps_per_epoch} steps/epoch x {a.epochs} = {total} steps, warm-up {warm}, amp {amp_dtype}, "
              f"world {torch.distributed.get_world_size() if ddp else 1}, trainable {sum(p.numel() for p in enc) / 1e6:.1f}M enc + "
              f"{sum(p.numel() for p in rest) / 1e6:.2f}M head", flush=True)
    step, ep, log, stop = 0, 0, [], False
    t0 = time.time()
    while step < total and not stop:
        ds.epoch = ep
        if sampler is not None:
            sampler.set_epoch(ep)
        tot, nb = 0.0, 0
        for it, (img, meta, mask, t, _) in enumerate(dl):
            S = img.shape[-1]
            oy = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
            ox = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
            img = img[..., oy:oy + a.crop, ox:ox + a.crop].to(dev, non_blocking=True)
            meta, mask, t = meta.to(dev), mask.to(dev), t.to(dev)
            with torch.autocast("cuda", dtype=amp_dtype):
                z = net(img, meta, mask)
            z = z.float()
            bce = (F.binary_cross_entropy_with_logits(z, t, reduction="none") * alpha).sum(1).mean() / alpha.sum()
            pl = (pair_loss(z, t, qz, qt) * alpha).sum() / alpha.sum()
            loss = (0.5 * bce + 0.5 * pl) / a.accum
            scaler.scale(loss).backward()
            qz = torch.cat([z.detach(), qz])[:a.queue]
            qt = torch.cat([t, qt])[:a.queue]
            tot += float(loss) * a.accum
            nb += 1
            if (it + 1) % a.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            with torch.no_grad():
                d_ = min(a.ema, (1 + step) / (10 + step))
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        ema[n].mul_(d_).add_(p.detach().float(), alpha=1 - d_)
            if rank == 0 and (step <= 3 or step % 50 == 0):
                el = time.time() - t0
                print(f"ep {ep} step {step}/{total} loss {tot / nb:.4f} | {el / step:.2f} s/step, ETA {(total - step) * el / step / 60:.0f} min "
                      f"| peak {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.2f} GB", flush=True)
            if step >= total:
                break
            if (time.time() - t_start) / 3600 > a.max_hours:
                stop = True
                if rank == 0:
                    print(f"STOP: {a.max_hours} h wall time reached at step {step}", flush=True)
                break
        log.append({"epoch": ep, "loss": round(tot / max(1, nb), 5), "steps": step, "minutes": round((time.time() - t0) / 60, 1)})
        if rank == 0:
            print("EPOCH", json.dumps(log[-1]), flush=True)
        ep += 1
    if rank == 0:
        os.makedirs(a.out, exist_ok=True)
        full = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
        last = dict(full)
        for n in names:
            full[n] = ema[n].cpu()
        torch.save(full, os.path.join(a.out, "s6ft_ema.pt"))
        torch.save(last, os.path.join(a.out, "s6ft_last.pt"))
        json.dump({"args": vars(a), "log": log, "steps": step, "total_planned": total, "stopped_early": stop, "amp": str(amp_dtype),
                   "studies": len(files), "seconds": round(time.time() - t_start, 1)},
                  open(os.path.join(a.out, "s6ft_run.json"), "w"), indent=1)
        print("S6FT_DONE", step, "steps; saved s6ft_ema.pt", flush=True)
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
