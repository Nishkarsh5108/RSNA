"""
Training script for DINOv2 + Region-Guided Attention (RGA) model.

Usage on Kaggle
---------------
1. Add the preprocessed volumes dataset and the labels dataset as inputs.
2. Copy this entire `model/` directory to `/kaggle/working/model/` or add
   it as a utility-script dataset.
3. Run this script or paste it into a notebook cell:

       %run model/train.py           # from a notebook
       python model/train.py         # from terminal

The script auto-detects the Kaggle environment and adjusts paths accordingly.
"""
import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score

# Allow imports when running from project root or from model/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from model.config import Config
from model.dataset import KneeMRIDataset
from model.knee_model import KneeModel


# =====================================================================
# Seed everything
# =====================================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================================
# Macro-averaged AUC
# =====================================================================
def compute_macro_auc(y_true, y_pred, target_cols):
    """Compute per-target AUC and macro average (skip constant targets)."""
    aucs = {}
    for i, col in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        if len(np.unique(yt > 0.5)) < 2:
            continue  # skip if only one class present
        aucs[col] = roc_auc_score((yt > 0.5).astype(int), yp)
    macro = np.mean(list(aucs.values())) if aucs else 0.0
    return macro, aucs


# =====================================================================
# Training loop for one epoch
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, config, device):
    model.train()
    # Keep backbone frozen even in train mode
    model.backbone.eval()

    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        slices    = batch["slices"].to(device)
        plane_ids = batch["plane_ids"].to(device)
        seq_ids   = batch["seq_ids"].to(device)
        masks     = batch["masks"].to(device)
        labels    = batch["labels"].to(device)

        with autocast(dtype=torch.float16):
            logits = model(slices, plane_ids, seq_ids, masks)
            loss = criterion(logits, labels)
            loss = loss / config.GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRAD_ACCUM_STEPS

    return total_loss / max(len(loader), 1)


# =====================================================================
# Validation loop
# =====================================================================
@torch.no_grad()
def validate(model, loader, criterion, config, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        slices    = batch["slices"].to(device)
        plane_ids = batch["plane_ids"].to(device)
        seq_ids   = batch["seq_ids"].to(device)
        masks     = batch["masks"].to(device)
        labels    = batch["labels"].to(device)

        with autocast(dtype=torch.float16):
            logits = model(slices, plane_ids, seq_ids, masks)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    macro_auc, per_target = compute_macro_auc(labels, preds, config.TARGET_COLS)

    return avg_loss, macro_auc, per_target


# =====================================================================
# Main
# =====================================================================
def main():
    cfg = Config
    seed_everything(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("  DINOv2 + Region-Guided Attention — Knee MRI Classifier")
    print("=" * 72)
    print(f"  Environment : {'Kaggle' if cfg.IS_KAGGLE else 'Local'}")
    print(f"  Device      : {device}")
    print(f"  Volumes dir : {cfg.volumes_dir()}")
    print(f"  Labels file : {cfg.labels_path()}")
    print(f"  Val fold    : {cfg.TRAIN_FOLD}")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load metadata
    # ------------------------------------------------------------------
    labels_df = pd.read_csv(cfg.labels_path())
    series_df = pd.read_csv(cfg.series_path())

    # Fill any remaining NaN targets with 0
    labels_df[cfg.TARGET_COLS] = labels_df[cfg.TARGET_COLS].fillna(0.0)

    train_df = labels_df[labels_df["fold"] != cfg.TRAIN_FOLD].reset_index(drop=True)
    val_df   = labels_df[labels_df["fold"] == cfg.TRAIN_FOLD].reset_index(drop=True)
    print(f"\n  Train studies : {len(train_df)}")
    print(f"  Val   studies : {len(val_df)}")

    # ------------------------------------------------------------------
    # 2. Datasets & loaders
    # ------------------------------------------------------------------
    vol_dir = cfg.volumes_dir()

    train_ds = KneeMRIDataset(train_df, series_df, vol_dir, cfg, is_train=True)
    val_ds   = KneeMRIDataset(val_df,   series_df, vol_dir, cfg, is_train=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 3. Model, loss, optimiser, scheduler
    # ------------------------------------------------------------------
    model = KneeModel(cfg).to(device)
    trainable = model.count_trainable_params()
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")
    print(f"  Frozen params    : {total - trainable:,}")

    # Multi-GPU support (DataParallel for 2x T4)
    if torch.cuda.device_count() > 1:
        print(f"  Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-6,
    )
    scaler = GradScaler()

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    best_auc = 0.0
    out_dir = cfg.output_dir()
    ckpt_path = os.path.join(out_dir, "best_model.pt")

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Macro AUC':>10} | {'LR':>10} | {'Time':>6}")
    print("-" * 72)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg, device)
        val_loss, macro_auc, per_target = validate(model, val_loader, criterion, cfg, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        marker = ""
        if macro_auc > best_auc:
            best_auc = macro_auc
            marker = " ★"
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, ckpt_path)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {val_loss:>10.4f} | {macro_auc:>10.4f} | {lr:>10.6f} | {elapsed:>5.0f}s{marker}")

    # ------------------------------------------------------------------
    # 5. Final results
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"  Best Macro AUC : {best_auc:.4f}")
    print(f"  Checkpoint     : {ckpt_path}")
    print("=" * 72)

    # Print per-target AUC of the best epoch (reload and re-eval)
    best_state = torch.load(ckpt_path, map_location=device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(best_state)
    _, _, per_target = validate(model, val_loader, criterion, cfg, device)

    print("\n  Per-Target AUC (Best Epoch):")
    for col, auc in sorted(per_target.items(), key=lambda x: -x[1]):
        print(f"    {col:<22} {auc:.4f}")


if __name__ == "__main__":
    main()
