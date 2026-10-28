"""'Human labels only' baseline: train on the 58 gold studies alone, leave-one-out.

Frozen v2 features are mean-pooled per plane (3 x 1280), reduced with PCA fitted
inside each LOO fold (no leakage), then a per-target L2 logistic regression.
The regularisation strength is fixed a priori (C=0.1) -- not tuned on gold.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS, macro_auc_ci, safe_auc  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv")
gold = train[train[TARGETS].notna().all(axis=1)].set_index("StudyInstanceUID")[TARGETS].astype(int)
X = []
for u in gold.index:
    z = np.load(f"{ROOT}/output/v2/feature_cache/{u}.npz")
    X.append(np.concatenate([z["sagittal"].mean(0), z["coronal"].mean(0), z["axial"].mean(0)]))
X = np.array(X)
Y = gold.values
rows = []
for n_comp in (8, 16):
    P = np.zeros(Y.shape, dtype=float)
    for i in range(len(X)):
        tr = np.arange(len(X)) != i
        sc = StandardScaler().fit(X[tr])
        pca = PCA(n_components=n_comp, random_state=0).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[i:i + 1]))
        for j in range(12):
            if len(np.unique(Y[tr, j])) < 2:
                P[i, j] = Y[tr, j].mean()
                continue
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(Ztr, Y[tr, j])
            P[i, j] = clf.predict_proba(Zte)[0, 1]
    mp, lo, hi = macro_auc_ci(Y, P, n_boot=2000, seed=21)
    row = {"model": f"gold-only LOO, PCA{n_comp}+LR(C=0.1)", "macro_auc": mp, "lo": lo, "hi": hi}
    row.update({f"auc_{t}": safe_auc(Y[:, j], P[:, j]) for j, t in enumerate(TARGETS)})
    rows.append(row)
    np.save(f"{ROOT}/reports/eda/downstream/gold_only_loo_pca{n_comp}.npy", P)
out = pd.DataFrame(rows)
os.makedirs(f"{ROOT}/reports/eda/downstream", exist_ok=True)
out.to_csv(f"{ROOT}/reports/eda/downstream/gold_only_baseline.csv", index=False)
print(out.round(3).T.to_string())
