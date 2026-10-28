"""s6ft_specialist_train.py: Specialist fine-tuning on pure vision-distilled targets.

Initializes from s6ft_v2 (0.9366 Gold AUC).
Trains on D:/rsna_work/s6ft_cache_all with training_labels_v6_vision_pure.csv.
Supports group-specialist loss weighting:
  - all: balanced multi-task with 0.4 BCE + 0.6 soft pairwise ranking
  - g1_ligaments: focused on ACL, MCL, Medial Meniscus, Lateral Meniscus
  - g2_cartilage: focused on Medial OA, Lateral OA, PF OA, Fracture, Contusion
  - g3_fluid: focused on Effusion, Baker's, Synovitis
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

GROUPS = {
    "g1_ligaments": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "g2_cartilage": ["Medial OA", "Lateral OA", "PF OA", "Fracture", "Contusion"],
    "g3_fluid": ["Effusion", "Baker's", "Synovitis"],
}


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
    Z, T = torch.cat([z, qz]), torch.cat([t, qt])
    d = z[:, None, :] - Z[None, :, :]
    w = t[:, None, :] - T[None, :, :]
    s, w = torch.sign(w), w.abs()
    loss = (F.softplus(-s * d) * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1e-6)
    return loss


def build_loss_weights(mode, device):
    weights = torch.ones(len(LAB), device=device)
    if mode == "all":
        # Slight boost on historically harder classes
        for c in ("Lateral OA", "PF OA", "Synovitis", "Lateral Meniscus", "Fracture"):
            weights[LAB.index(c)] = 1.3
    elif mode in GROUPS:
        focus = GROUPS[mode]
        for i, c in enumerate(LAB):
            weights[i] = 2.5 if c in focus else 0.5
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/rsna_work/s6ft_cache_all")
    ap.add_argument("--targets", default="data/labels_v4/training_labels_v6_vision_pure.csv")
    ap.add_argument("--init", default="D:/rsna_work/s6ft_v2_local/s6ft_ema.pt")
    ap.add_argument("--code", default="D:/rsna_work/public_code/s6_src")
    ap.add_argument("--out", default="D:/rsna_work/s6ft_v3_local")
    ap.add_argument("--mode", default="all", choices=["all", "g1_ligaments", "g2_cartilage", "g3_fluid"])
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--crop", type=int, default=288)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--head-lr", type=float, default=8e-5)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--queue", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261021)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)

    sys.path.insert(0, a.code)
    import compact_train as CT
    assert CT.LABELS == LAB
    import pandas as pd

    tg = pd.read_csv(a.targets, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    tgt = {u: v.astype(np.float32) for u, v in zip(tg.index, tg.values)}

    files = sorted(glob.glob(os.path.join(a.cache, "**", "*.npz"), recursive=True))
    probe = CacheDS(files, {}, a.windows, a.seed)
    uid_of = {f: str(probe._open(f)["uid"]) for f in files}
    files = [f for f in files if uid_of[f] in tgt]
    print(f"[s6ft_specialist] mode={a.mode} | {len(files)} cached studies with targets (of {len(tg)} targets)", flush=True)

    ds = CacheDS(files, tgt, a.windows, a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                                     collate_fn=collate, drop_last=True, pin_memory=True,
                                     persistent_workers=a.workers > 0, prefetch_factor=2 if a.workers else None)

    model = CT.Net("convnext_small", None, freeze_blocks=16, crop=a.crop, state_spec=None, in_chans=3, spatial=True)
    sd = torch.load(a.init, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    print(f"Loaded init weights from {a.init}", flush=True)

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
    ema = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    steps_per_epoch = len(dl) // a.accum
    total = max(1, int(round(a.epochs * steps_per_epoch)))
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
    )

    use_bf16 = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability(dev)[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)

    loss_weights = build_loss_weights(a.mode, dev)
    qz = torch.zeros((0, 12), device=dev)
    qt = torch.zeros((0, 12), device=dev)
    gen = torch.Generator().manual_seed(a.seed + 17)

    print(f"[s6ft_specialist] {steps_per_epoch} steps/epoch x {a.epochs} = {total} steps, warm={warm}, amp={amp_dtype}", flush=True)
    step, ep, log = 0, 0, []
    t0 = time.time()

    while step < total:
        ds.epoch = ep
        tot, nb = 0.0, 0
        for it, (img, meta, mask, t, _) in enumerate(dl):
            S = img.shape[-1]
            oy = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
            ox = int(torch.randint(0, S - a.crop + 1, (1,), generator=gen))
            img = img[..., oy:oy + a.crop, ox:ox + a.crop].to(dev, non_blocking=True)
            meta, mask, t = meta.to(dev), mask.to(dev), t.to(dev)

            with torch.autocast("cuda", dtype=amp_dtype):
                z = model(img, meta, mask)
            z = z.float()

            bce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
            ploss = pair_loss(z, t, qz, qt) if len(qz) > 0 else torch.zeros(12, device=dev)
            # 0.4 BCE + 0.6 Pairwise Ranking Loss, weighted per finding
            loss_per_class = 0.4 * bce.mean(0) + 0.6 * ploss
            loss = (loss_weights * loss_per_class).sum() / loss_weights.sum()

            scaler.scale(loss / a.accum).backward()
            tot += float(loss.item())
            nb += 1

            if (it + 1) % a.accum == 0 or (it + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1

                # Update EMA
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            ema[n].mul_(a.ema).add_(p.detach().float(), alpha=1 - a.ema)

                # Queue update
                with torch.no_grad():
                    qz = torch.cat([qz, z.detach()])[-a.queue:]
                    qt = torch.cat([qt, t.detach()])[-a.queue:]

                if step % 50 == 0 or step == total:
                    elapsed = time.time() - t0
                    s_per_step = elapsed / step
                    eta_m = (total - step) * s_per_step / 60
                    lr_curr = sched.get_last_lr()[0]
                    print(f"[step {step}/{total}] ep {ep} loss {tot/nb:.4f} lr {lr_curr:.2e} | {s_per_step:.2f} s/step | ETA {eta_m:.1f} min", flush=True)

            if step >= total or (time.time() - t_start) > a.max_hours * 3600:
                break
        ep += 1

    # Save final model and EMA
    out_tag = f"_{a.mode}" if a.mode != "all" else ""
    save_path = os.path.join(a.out, f"s6ft_specialist{out_tag}_ema.pt")
    # Build clean state dict matching CT.Net keys
    final_sd = {}
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k in ema:
            final_sd[k] = ema[k].cpu()
        else:
            final_sd[k] = v.cpu()

    torch.save(final_sd, save_path)
    print(f"Successfully saved specialist checkpoint to {save_path} ({os.path.getsize(save_path)} bytes)", flush=True)


if __name__ == "__main__":
    main()
