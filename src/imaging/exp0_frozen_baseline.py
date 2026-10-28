"""Experiment 0: the v2 frozen-feature model (EfficientNet-B0 ImageNet features, 12 slices,
one series per plane, attention pool + MLP head) under the SAME split / protocol as the
fine-tuned runs: train on non-gold folds != 0, select epoch on pseudo-val fold 0 (teacher
binarised at 0.5), save gold predictions sealed.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from downstream_eval import Head, load_features  # noqa: E402
from study_index import TARGETS  # noqa: E402
from train import macro_auc  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
torch.set_num_threads(2)
idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train.csv")
lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
tr, va, go = idx[(idx.fold >= 1)], idx[idx.fold == 0], idx[idx.is_gold]
cache = f"{R}/output/v2/feature_cache"
Xtr, Xva, Xgo = (load_features(list(d.StudyInstanceUID), cache) for d in (tr, va, go))
ytr = lab.loc[tr.StudyInstanceUID, TARGETS].values.astype(np.float32)
yva = (lab.loc[va.StudyInstanceUID, TARGETS].values >= 0.5).astype(np.float32)
out = f"{R}/output/imaging_v5/runs/E0_frozen_effb0_v2head"
os.makedirs(out, exist_ok=True)
rows, best = [], (-1, None, None)
for seed in range(3):
    torch.manual_seed(seed)
    m = Head()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, Y = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    prev = ytr.mean(0)
    pw = torch.tensor(np.clip((1 - prev) / np.clip(prev, 1e-3, None), 1, 10), dtype=torch.float32)
    for ep in range(20):
        m.train()
        perm = torch.randperm(len(Xt))
        for s in range(0, len(Xt) - 63, 64):
            b = perm[s:s + 64]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xt[b].float()), Y[b], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pva = torch.sigmoid(m(torch.from_numpy(Xva).float())).numpy()
            pgo = torch.sigmoid(m(torch.from_numpy(Xgo).float())).numpy()
        auc, per = macro_auc(yva, pva)
        rows.append({"seed": seed, "epoch": ep, "val_macro_auc": auc, **{f"val_auc_{t}": v for t, v in per.items()}})
        if auc > best[0]:
            best = (auc, pgo, (seed, ep))
            pd.DataFrame(pva, columns=TARGETS).assign(StudyInstanceUID=va.StudyInstanceUID.values).to_csv(
                f"{out}/val_preds_best.csv", index=False)
    print(f"seed {seed}: best val so far {best[0]:.4f} at {best[2]}", flush=True)
h = pd.DataFrame(rows)
h.to_csv(f"{out}/history.csv", index=False)
bi = h.val_macro_auc.idxmax()
np.savez(f"{out}/gold_preds_SEALED.npz", uids=go.StudyInstanceUID.values, p=best[1], epoch=int(h.epoch[bi]))
json.dump({"run_id": "E0_frozen_effb0_v2head", "best_val_macro_auc": float(h.val_macro_auc[bi]), "best_epoch": int(h.epoch[bi]),
           "best_seed_epoch": [int(h.seed[bi]), int(h.epoch[bi])],
           "per_class_val_auc_best": {t: float(h[f"val_auc_{t}"][bi]) for t in TARGETS}},
          open(f"{out}/summary.json", "w"), indent=1)
print("best pseudo-val macro AUC", round(float(h.val_macro_auc[bi]), 4))
