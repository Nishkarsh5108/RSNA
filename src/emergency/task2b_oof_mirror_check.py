"""Plan section 12, upgrade B's extra condition: S6-only mirror-TTA check on oof_300.

Each oof_300 study is scored by the one S6 fold model that never trained on it (fold = int(sha1(uid)[:8], 16) % 4):
  plain  = z_plain[fold]                          (what the Kaggle stage computes)
  mirror = z_mirror[fold] = 0.5 * (plain + P(mirrored pass)), MCL kept from the plain pass (R6's v4c rule)
Labels: labels_v3 (data/labels_v3/training_labels_v3.csv), confident cells only: <= 0.2 negative, >= 0.8 positive, the
rest dropped per finding. Macro AUC = mean over the findings that have both classes.
Rule (fixed 2026-10-20 00:46 IST, before any prediction existed): delta = macro(mirror) - macro(plain) >= 0.
Paired bootstrap (10,000, seed 0) CI reported, descriptive only. s6fall and Raptor are all-data models, so oof_300 is
in-sample for them and they are not scored here.
Usage: python -I src/emergency/task2b_oof_mirror_check.py
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, M, S, idx=None):
    """Y [n,12] binary, M [n,12] confident-cell mask, S [n,12] scores; idx: bootstrap row indices."""
    if idx is not None:
        Y, M, S = Y[idx], M[idx], S[idx]
    return np.nanmean([auc(Y[M[:, j], j], S[M[:, j], j]) for j in range(12)])


def main():
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    st = pd.read_csv(f"{T1}/task1_studies.csv", dtype={"StudyInstanceUID": str})
    st = st[st["set"] == "oof_300"]
    ids, zp, zm = [], [], []
    for u, f in zip(st.StudyInstanceUID, st.s6_fold):
        p = f"{T1}/s6/{hashlib.sha256(u.encode()).hexdigest()[:16]}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p, allow_pickle=True)
        assert str(z["uid"]) == u and int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 == f
        assert np.isfinite(z["z_mirror"][f]).all(), u
        ids.append(u); zp.append(z["z_plain"][f]); zm.append(z["z_mirror"][f])
    zp, zm = np.stack(zp), np.stack(zm)
    soft = lab.loc[ids, LAB].values.astype(float)
    M = (soft <= 0.2) | (soft >= 0.8)
    Y = (soft >= 0.8).astype(int)
    print(f"oof_300 studies with S6 outputs: {len(ids)} of {len(st)} | confident cells per finding: "
          + ", ".join(f"{c} {int(M[:, j].sum())}/{int(Y[M[:, j], j].sum())}+" for j, c in enumerate(LAB)))
    base, mir = macro(Y, M, zp), macro(Y, M, zm)
    rng = np.random.default_rng(0)
    boot = np.array([macro(Y, M, zm, b) - macro(Y, M, zp, b) for b in (rng.integers(0, len(ids), len(ids)) for _ in range(10000))])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    per = {c: round(auc(Y[M[:, j], j], zm[M[:, j], j]) - auc(Y[M[:, j], j], zp[M[:, j], j]), 4) for j, c in enumerate(LAB)}
    res = {"n": len(ids), "plain_macro": round(base, 4), "mirror_macro": round(mir, 4), "delta": round(mir - base, 4),
           "ci95": [round(lo, 4), round(hi, 4)], "per_finding_delta": per, "condition_met": bool(mir - base >= 0),
           "corr_plain_mirror": round(float(np.corrcoef(zp.ravel(), zm.ravel())[0, 1]), 4)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{T1}/task2b_oof_mirror_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
