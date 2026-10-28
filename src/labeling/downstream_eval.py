"""Downstream test of label sources: does a label set train a better image model?

Protocol (pre-registered, see reports/baseline_comparison.md):
  * Image features: the frozen EfficientNet-B0 v2 cache (one fluid-sensitive-first
    series per plane, 12 slices, 1280-d per slice) that produced the 0.775 public LB.
  * Model: the v2 head (shared slice attention-pool per plane -> concat -> MLP).
  * Train ONLY on the 4,349 non-gold studies with the label set under test.
    Gold studies are never used for training, early stopping, checkpoint or
    threshold selection. A fixed number of epochs is used for every label set.
  * Evaluate once on all 58 gold (image-read) studies; repeat over seeds; compare
    label sets with a paired bootstrap over gold studies on the seed-averaged
    predictions.
  * Label targets can be soft (BCE accepts probabilities) and may contain NaN,
    which is MASKED out of the loss (used to test "not mentioned != negative").
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS, binary_report, macro_auc_ci, paired_macro_auc_diff, safe_auc  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"


class AttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)


class Head(nn.Module):
    def __init__(self, dim=1280, n=12):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(nn.Linear(dim * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):  # x: (B, 3, T, D)
        return self.head(torch.cat([self.pool(x[:, p]) for p in range(3)], dim=1))


def load_features(uids, cache_dir):
    X = np.zeros((len(uids), 3, 12, 1280), dtype=np.float16)
    for i, u in enumerate(uids):
        z = np.load(os.path.join(cache_dir, f"{u}.npz"))
        X[i, 0], X[i, 1], X[i, 2] = z["sagittal"], z["coronal"], z["axial"]
    return X


def train_and_predict(Xtr, Ytr, Xte, seed, epochs=12, lr=1e-3, wd=1e-4, bs=64, device="cpu",
                      pos_weight_clip=(1.0, 10.0)):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Head().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.from_numpy(Xtr)
    Y = torch.from_numpy(np.nan_to_num(Ytr, nan=0.0).astype(np.float32))
    M = torch.from_numpy((~np.isnan(Ytr)).astype(np.float32))
    prev = np.nansum(Ytr, axis=0) / np.maximum((~np.isnan(Ytr)).sum(axis=0), 1)
    pw = torch.tensor(np.clip((1 - prev) / np.clip(prev, 1e-3, None), *pos_weight_clip), dtype=torch.float32, device=device)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n - bs + 1, bs):  # drop_last like v2
            idx = perm[s:s + bs]
            xb = Xtr_t[idx].to(device).float()
            yb, mb = Y[idx].to(device), M[idx].to(device)
            logits = model(xb)
            loss_el = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pw, reduction="none") * mb
            loss = loss_el.sum() / mb.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        out = []
        for s in range(0, len(Xte), 256):
            out.append(torch.sigmoid(model(torch.from_numpy(Xte[s:s + 256]).to(device).float())).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-sets", required=True, help="JSON file: {name: csv_path_with_StudyInstanceUID_and_12_targets}")
    ap.add_argument("--cache", default=f"{ROOT}/output/v2/feature_cache")
    ap.add_argument("--out", default=f"{ROOT}/reports/eda/downstream")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--reference", default="", help="label-set name used as the paired-comparison reference")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv")
    gold = train[train[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")[TARGETS].astype(float)
    gold_uids = list(gold.index)
    nongold = [u for u in train.StudyInstanceUID if u not in set(gold_uids)]
    t0 = time.time()
    Xtr_all = load_features(nongold, args.cache)
    Xte = load_features(gold_uids, args.cache)
    print(f"features loaded in {time.time()-t0:.0f}s: train {Xtr_all.shape} gold {Xte.shape}", flush=True)

    sets = json.load(open(args.label_sets))
    results, preds = [], {}
    for name, path in sets.items():
        lab = pd.read_csv(path).set_index("StudyInstanceUID")[TARGETS].astype(float).reindex(nongold)
        keep = lab.notna().any(axis=1).values  # studies with at least one usable target
        Ytr = lab.values[keep]
        P = []
        for sd in range(args.seeds):
            P.append(train_and_predict(Xtr_all[keep], Ytr, Xte, seed=sd, epochs=args.epochs, device=args.device))
        P = np.stack(P)  # (seeds, 58, 12)
        preds[name] = P
        np.save(os.path.join(args.out, f"gold_preds_{name}.npy"), P)
        per_seed = [np.nanmean([safe_auc(gold.values[:, j], P[s, :, j]) for j in range(12)]) for s in range(args.seeds)]
        Pm = P.mean(0)
        mp, lo, hi = macro_auc_ci(gold.values, Pm, n_boot=2000, seed=3)
        row = {"label_set": name, "n_train_studies": int(keep.sum()), "masked_cells_frac": float(np.isnan(Ytr).mean()),
               "macro_auc_seedavg": mp, "ci_lo": lo, "ci_hi": hi,
               "macro_auc_per_seed_mean": float(np.mean(per_seed)), "macro_auc_per_seed_sd": float(np.std(per_seed))}
        for j, t in enumerate(TARGETS):
            row[f"auc_{t}"] = safe_auc(gold.values[:, j], Pm[:, j])
        results.append(row)
        print(f"{name:32s} macro AUC {mp:.4f} [{lo:.3f},{hi:.3f}] per-seed {np.mean(per_seed):.4f}±{np.std(per_seed):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(args.out, "downstream_results.csv"), index=False)
    if args.reference and args.reference in preds:
        ref = preds[args.reference].mean(0)
        comp = []
        for name, P in preds.items():
            if name == args.reference:
                continue
            d, lo, hi, p_le0 = paired_macro_auc_diff(gold.values, P.mean(0), ref, n_boot=2000, seed=5)
            comp.append({"label_set": name, "vs": args.reference, "delta_macro_auc": d, "ci_lo": lo, "ci_hi": hi,
                         "bootstrap_frac_delta_le_0": p_le0})
        pd.DataFrame(comp).to_csv(os.path.join(args.out, f"paired_vs_{args.reference}.csv"), index=False)
        print(pd.DataFrame(comp).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
