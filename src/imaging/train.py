"""Train / validate a 2.5D multi-slot MIL knee model (local or Kaggle).

Validation protocol (see reports/image_model_experiments.md):
  * train on non-gold studies of folds != val_fold (optionally + gold with --gold-weight > 0),
  * select the checkpoint by macro AUC on the PSEUDO-labelled val fold (teacher binarised at 0.5),
  * predict the 58 gold studies at the selected checkpoint and SAVE THEM SEALED (no metric is
    computed here); gold metrics are computed once, later, by evaluate_sealed.py.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from data import KneeStudyDataset, gpu_augment, load_labels, pad_collate  # noqa: E402
from model import KneeMIL  # noqa: E402
from study_index import TARGETS  # noqa: E402
from metrics import safe_auc  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--index", default=f"{R}/data/imaging_v5/study_index_train.csv")
    ap.add_argument("--labels", default=f"{R}/data/labels_v3/training_labels_v3.csv")
    ap.add_argument("--val-labels", default=f"{R}/data/labels_v3/training_labels_v3.csv",
                    help="labels for the pseudo-val metric (default: the report teacher, so runs trained on other "
                         "targets, e.g. pseudo-labels, are scored against the same reference)")
    ap.add_argument("--cache", default=f"{R}/output/v2/preprocessed")
    ap.add_argument("--preproc", default="v2", choices=["v2", "v6", "hr"],
                    help="preprocessing that produced --cache (recorded for inference): v2 = 224 px squash, "
                         "v6 = 0.6 mm/px aspect-preserving 256 px crop, hr = 140 mm field / 384 grid, centre 352 stored")
    ap.add_argument("--crop", type=int, default=0, help="square crop of every window (hr: 320); random per series in training")
    ap.add_argument("--K-eval", type=int, default=-1, help="windows per slot for validation/gold; -1 = same as --K, 0 = every slice")
    ap.add_argument("--k-max", type=int, default=64, help="cap on windows per slot when --K-eval 0")
    ap.add_argument("--out-root", default=f"{R}/output/imaging_v5/runs")
    ap.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--slots", default="0,1,2")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--agg", default="attn12")
    ap.add_argument("--no-pos", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-bb", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.5, help="warm-up length in epochs")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--label-transform", default="soft", choices=["soft", "hard"])
    ap.add_argument("--gold-weight", type=float, default=0.0, help="0 = gold never trained on")
    ap.add_argument("--pos-weight-clip", type=float, default=0.0, help="0 = plain BCE")
    ap.add_argument("--conf-weight", type=float, default=0.0,
                    help="0 = off; else each label's BCE is weighted by |2y-1|^gamma (down-weights uncertain teacher labels)")
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--train-all-folds", action="store_true", help="final model: train on every non-gold fold")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--random-series", action="store_true")
    ap.add_argument("--no-orient", action="store_true", help="ablation: disable laterality normalisation")
    ap.add_argument("--sag-pos-fix", action="store_true",
                    help="correct slice-position feature for right-knee sagittal series (see data.KneeStudyDataset)")
    ap.add_argument("--lo", type=float, default=0.04)
    ap.add_argument("--hi", type=float, default=0.96)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16"],
                    help="autocast dtype; auto = bf16 on Ampere or newer, else fp16 with loss scaling")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-limit-min", type=float, default=0, help="stop cleanly (resumable) after N minutes")
    return ap.parse_args(argv)


def macro_auc(y_bin, p):
    per = {t: safe_auc(y_bin[:, j], p[:, j]) for j, t in enumerate(TARGETS)}
    vals = [v for v in per.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


@torch.no_grad()
def predict(model, loader, dev, amp_dt=torch.bfloat16):
    model.eval()
    out = np.zeros((len(loader.dataset), len(TARGETS)), np.float32)
    for b in loader:
        x = b["x"].to(dev, non_blocking=True).float().div_(255)
        wm = b["wmask"].to(dev) if "wmask" in b else None
        with torch.autocast("cuda", dtype=amp_dt):
            lg = model(x, b["mask"].to(dev), b["pos"].to(dev), wm)
        out[b["i"].numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
    return out


def keep_awake():
    """Prevent Windows from sleeping while this process runs (no system settings are changed)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main(args):
    keep_awake()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run_dir = os.path.join(args.out_root, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    dev = "cuda"
    # bf16 on Ampere or newer (the local RTX 4050); fp16 + loss scaling on older GPUs (e.g. Kaggle T4, no native bf16)
    amp_dt = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        args.amp, torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dt == torch.float16)
    idx = pd.read_csv(args.index)
    slots = [int(s) for s in args.slots.split(",")]
    gold = idx[idx.is_gold].reset_index(drop=True)
    if args.train_all_folds:
        tr = idx[idx.fold >= 0]
        va = idx[idx.fold == args.val_fold]        # still reported, but it is in training (not a holdout)
    else:
        tr = idx[(idx.fold >= 0) & (idx.fold != args.val_fold)]
        va = idx[idx.fold == args.val_fold]
    if args.gold_weight > 0:
        tr = pd.concat([tr, gold])
    if args.max_train:
        tr = tr.sample(min(args.max_train, len(tr)), random_state=args.seed)
    tr, va = tr.reset_index(drop=True), va.reset_index(drop=True)
    y_tr, _ = load_labels(args.labels, tr.StudyInstanceUID, args.label_transform)
    w_tr = np.where(tr.is_gold, args.gold_weight, 1.0).astype(np.float32)
    y_va_soft, _ = load_labels(args.val_labels, va.StudyInstanceUID, "soft")
    y_va_bin = (y_va_soft >= 0.5).astype(np.float32)
    spacing_map = {}                    # only needed for --random-series (non-default candidates)
    hdr_path = f"{R}/reports/eda/dicom/series_headers.csv"
    if args.random_series and os.path.exists(hdr_path):
        hdr = pd.read_csv(hdr_path, dtype=str, usecols=["SeriesInstanceUID_dir", "SpacingBetweenSlices", "SliceThickness"])
        sp = pd.to_numeric(hdr.SpacingBetweenSlices, errors="coerce").fillna(pd.to_numeric(hdr.SliceThickness, errors="coerce"))
        spacing_map = dict(zip(hdr.SeriesInstanceUID_dir, sp.fillna(3.5)))
    orient = not args.no_orient
    k_eval = args.K if args.K_eval < 0 else args.K_eval
    ds_tr = KneeStudyDataset(tr, args.cache, slots, args.K, train=True, labels=y_tr, weights=w_tr, lo=args.lo, hi=args.hi,
                             random_series=args.random_series, spacing_map=spacing_map, seed=args.seed, orient=orient,
                             sag_pos_fix=args.sag_pos_fix, crop=args.crop)
    ds_va = KneeStudyDataset(va, args.cache, slots, k_eval, train=False, lo=args.lo, hi=args.hi, orient=orient,
                             sag_pos_fix=args.sag_pos_fix, crop=args.crop, k_max=args.k_max)
    ds_go = KneeStudyDataset(gold, args.cache, slots, k_eval, train=False, lo=args.lo, hi=args.hi, orient=orient,
                             sag_pos_fix=args.sag_pos_fix, crop=args.crop, k_max=args.k_max)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
              prefetch_factor=4 if args.workers > 0 else None)
    ev_bs = 4 if (args.crop == 0 and k_eval > 0) else 2
    ev_coll = pad_collate if k_eval == 0 else None
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=args.bs, shuffle=True, drop_last=True, **kw)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=ev_bs, shuffle=False, collate_fn=ev_coll, **kw)
    dl_go = torch.utils.data.DataLoader(ds_go, batch_size=ev_bs, shuffle=False, num_workers=0, collate_fn=ev_coll)

    model = KneeMIL(args.backbone, True, n_slots=6, agg=args.agg, use_pos=not args.no_pos,
                    drop_path=args.drop_path, grad_ckpt=args.grad_ckpt).to(dev)
    enc_params = list(model.enc.parameters())
    enc_ids = {id(p) for p in enc_params}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_bb}, {"params": head_params, "lr": args.lr_head}],
                            weight_decay=args.wd)
    steps_per_epoch = len(dl_tr) // args.accum
    total, warm = steps_per_epoch * args.epochs, int(steps_per_epoch * args.warmup)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) *
                                             0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=args.ema) if args.ema > 0 else None
    pw = None
    if args.pos_weight_clip > 0:
        prev = y_tr.mean(0).clip(1e-3, 1)
        pw = torch.tensor(np.clip((1 - prev) / prev, 1.0, args.pos_weight_clip), device=dev)

    start_epoch, best, hist, step = 0, -1.0, [], 0
    ck_last = os.path.join(run_dir, "ckpt_last.pt")
    if args.resume and os.path.exists(ck_last):
        st = torch.load(ck_last, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if ema is not None and st.get("ema") is not None:
            ema.module.load_state_dict(st["ema"])
        start_epoch, best, hist, step = st["epoch"] + 1, st["best"], st["hist"], st["step"]
        if st.get("scaler") is not None:
            scaler.load_state_dict(st["scaler"])
        print(f"resumed at epoch {start_epoch}", flush=True)

    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for ep in range(start_epoch, args.epochs):
        model.train()
        t0, run_loss, n_seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(dl_tr):
            x = b["x"].to(dev, non_blocking=True).float().div_(255)
            mask, pos = b["mask"].to(dev), b["pos"].to(dev)
            y, w = b["y"].to(dev), b["w"].to(dev)
            if not args.no_aug:
                x = gpu_augment(x, mask)
            with torch.autocast("cuda", dtype=amp_dt):
                lg = model(x, mask, pos)
            loss_el = F.binary_cross_entropy_with_logits(lg.float(), y, pos_weight=pw, reduction="none")
            if args.conf_weight > 0:
                loss_el = loss_el * (2 * y - 1).abs().pow(args.conf_weight)
            loss_el = loss_el.mean(1)
            loss = (loss_el * w).sum() / w.sum().clamp(min=1e-6) / args.accum
            scaler.scale(loss).backward()          # plain backward when the scaler is disabled (bf16)
            run_loss += loss.item() * args.accum * len(y); n_seen += len(y)
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if ema is not None:
                    ema.update(model, step=step)
            if it % 200 == 0:
                print(f"ep {ep} it {it}/{len(dl_tr)} loss {run_loss/max(n_seen,1):.4f} "
                      f"{n_seen/(time.time()-t0):.2f} studies/s mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        t_train = time.time() - t0
        eval_model = ema.module if ema is not None else model
        t1 = time.time()
        p_va = predict(eval_model, dl_va, dev, amp_dt)
        t_val = time.time() - t1
        va_auc, va_per = macro_auc(y_va_bin, p_va)
        va_bce = float(F.binary_cross_entropy(torch.tensor(p_va).clamp(1e-6, 1 - 1e-6), torch.tensor(y_va_soft)).item())
        row = {"epoch": ep, "train_loss": run_loss / max(n_seen, 1), "val_bce_soft": va_bce, "val_macro_auc": va_auc,
               "train_min": t_train / 60, "val_min": t_val / 60, "train_studies_per_s": n_seen / t_train,
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9, "lr_bb": opt.param_groups[0]["lr"],
               **{f"val_auc_{t}": v for t, v in va_per.items()}}
        hist.append(row)
        pd.DataFrame(hist).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        print(f"== epoch {ep}: train_loss {row['train_loss']:.4f} val_bce {va_bce:.4f} val_macroAUC {va_auc:.4f} "
              f"({t_train/60:.1f}+{t_val/60:.1f} min)", flush=True)
        # --train-all-folds: the val fold is part of training, so selection is meaningless -> keep the LAST epoch
        if va_auc > best or args.train_all_folds:
            best = max(best, va_auc)
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "epoch": ep, "val_macro_auc": va_auc},
                       os.path.join(run_dir, "ckpt_best.pt"))
            pd.DataFrame(p_va, columns=TARGETS).assign(StudyInstanceUID=va.StudyInstanceUID.values).to_csv(
                os.path.join(run_dir, "val_preds_best.csv"), index=False)
            p_go = predict(eval_model, dl_go, dev, amp_dt)       # SEALED: saved, not scored
            np.savez(os.path.join(run_dir, "gold_preds_SEALED.npz"), uids=gold.StudyInstanceUID.values, p=p_go, epoch=ep)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None, "epoch": ep, "best": best,
                    "hist": hist, "step": step, "scaler": scaler.state_dict() if scaler.is_enabled() else None}, ck_last)
        if args.time_limit_min and (time.time() - t_start) / 60 > args.time_limit_min and ep < args.epochs - 1:
            print("time limit reached; stopping (resume with --resume)", flush=True)
            return
    h = pd.DataFrame(hist)
    bi = int(h.val_macro_auc.idxmax())
    summ = {"run_id": args.run_id, "best_epoch": int(h.epoch[bi]), "best_val_macro_auc": float(h.val_macro_auc[bi]),
            "last_val_macro_auc": float(h.val_macro_auc.iloc[-1]), "epochs": args.epochs,
            "total_train_min": float(h.train_min.sum()), "total_val_min": float(h.val_min.sum()),
            "peak_mem_gb": float(h.peak_mem_gb.max()), "n_train": len(tr), "n_val": len(va),
            "val_is_holdout": not args.train_all_folds, "saved_epoch": int(h.epoch.iloc[-1]) if args.train_all_folds else int(h.epoch[bi]),
            "per_class_val_auc_best": {t: float(h[f"val_auc_{t}"][bi]) for t in TARGETS}}
    json.dump(summ, open(os.path.join(run_dir, "summary.json"), "w"), indent=1)
    print("DONE", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main(get_args())
