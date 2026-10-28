"""Plan section 14 C (mission 2026-10-21, Track 2B): soft targets for s6ft_v2. Nothing here reads gold labels.

  P_report = 0.5 labels_v3 + 0.5 nartaa's JEV blend (one source alone where the other lacks the study): as s6ft_targets.py
  P_vis    = S6's held-out-fold prediction on the cache (s6_oof_cache.py), quantile-mapped per finding onto P_report over the
             pool (the k-th smallest P_vis gets the k-th smallest P_report value), so the marginal label distribution is
             exactly P_report's and only the ordering comes from the images
  target   = 0.70 P_report + 0.30 P_vis, every cell (no rules, no confident-positive pull-down)
Usage: python -I src/emergency/s6ft_targets_v5.py -> data/labels_v4/training_labels_v5_clean.csv + _summary.json
"""
import json
import os

import numpy as np
import pandas as pd

R = "D:/Coding/kaggle/RSNA"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
JEV = "D:/rsna_work/public_weights/rsna-knee-s6fall-weights/training_code/labels/labels_v1_blendjev.parquet"
VIS = f"{R}/output/emergency_research/track2/s6_oof_cache.csv"
W_REP, W_VIS = 0.70, 0.30


def main():
    gold = set(pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID)
    v3 = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    jev = pd.read_parquet(JEV).set_index("StudyInstanceUID")[LAB]
    vis = pd.read_csv(VIS, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    pool = sorted(((set(v3.index) | set(jev.index)) & set(vis.index)) - gold)
    assert not (set(pool) & gold) and not (set(vis.index) & gold)
    a = v3.reindex(pool).values.astype(float)
    b = jev.reindex(pool).values.astype(float)
    rep = np.where(np.isnan(a), b, np.where(np.isnan(b), a, 0.5 * a + 0.5 * b))
    keep = np.isfinite(rep).all(axis=1)
    pool = [u for u, k in zip(pool, keep) if k]
    rep = rep[keep]
    pv = vis.loc[pool].values.astype(float)
    assert np.isfinite(pv).all()
    qm = np.empty_like(pv)
    summary = {"pool": len(pool), "weights": {"report": W_REP, "vis_quantile_mapped": W_VIS}, "per_finding": {}}
    for j, c in enumerate(LAB):
        order = np.argsort(pv[:, j], kind="mergesort")
        qm[order, j] = np.sort(rep[:, j])
        t = W_REP * rep[:, j] + W_VIS * qm[:, j]
        summary["per_finding"][c] = {
            "mean_report": round(float(rep[:, j].mean()), 4), "mean_target": round(float(t.mean()), 4),
            "spearman_vis_report": round(float(pd.Series(pv[:, j]).corr(pd.Series(rep[:, j]), method="spearman")), 4),
            "mean_abs_change": round(float(np.abs(t - rep[:, j]).mean()), 4)}
    tgt = np.clip(W_REP * rep + W_VIS * qm, 0, 1)
    os.makedirs(f"{R}/data/labels_v4", exist_ok=True)
    out = pd.DataFrame(tgt, columns=LAB)
    out.insert(0, "StudyInstanceUID", pool)
    out.to_csv(f"{R}/data/labels_v4/training_labels_v5_clean.csv", index=False)
    json.dump(summary, open(f"{R}/data/labels_v4/training_labels_v5_clean_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
