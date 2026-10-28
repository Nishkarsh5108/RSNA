"""Plan section 13, Track B: multi-teacher soft targets for the s6fall fine-tune (s6ft). Nothing here reads gold labels.

  P_report = 0.5 labels_v3 (our report labeller: Qwen3-4B both orders + gpt56sol + report_v2) + 0.5 nartaa's JEV blend
             (labels_v1_blendjev, the targets s6fall was trained on); one source alone where the other lacks the study
  P_vis    = our 5-fold out-of-fold image models (data/labels_v3/pl_g_r1_oof_only.csv: E1p fold 0 + O1-O4), Platt-calibrated
             per finding to P_report on the training pool (soft-label logistic fit on logit(P_vis); gold not used)
  over-called findings  (Medial OA, Lateral OA, PF OA, Baker's):                    P_report > 0.6          -> 0.45 P_report + 0.55 P_vis
  under-called findings (Synovitis, Fracture, Lateral Meniscus, ACL, Contusion):    0.08 <= P_report <= 0.45 -> 0.70 P_vis + 0.30 P_report
  otherwise P_report.  (The brief's rules; the brief's gold-prevalence calibration is replaced by the pool fit so that gold
  stays sealed.)
Usage: python -I src/emergency/s6ft_targets.py   -> data/labels_v4/training_labels_v4_superior.csv + _summary.json
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize

R = "D:/Coding/kaggle/RSNA"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
OVER = ("Medial OA", "Lateral OA", "PF OA", "Baker's")
UNDER = ("Synovitis", "Fracture", "Lateral Meniscus", "ACL", "Contusion")
JEV = "D:/rsna_work/public_weights/rsna-knee-s6fall-weights/training_code/labels/labels_v1_blendjev.parquet"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(x, t):
    """Soft-label logistic fit t ~ sigmoid(a x + b); returns (a, b)."""
    def f(ab):
        z = ab[0] * x + ab[1]
        return np.mean(np.logaddexp(0, z) - t * z)
    r = minimize(f, np.array([1.0, 0.0]), method="L-BFGS-B")
    return r.x


def main():
    gold = set(pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID)
    v3 = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    jev = pd.read_parquet(JEV).set_index("StudyInstanceUID")[LAB]
    vis = pd.read_csv(f"{R}/data/labels_v3/pl_g_r1_oof_only.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    pool = sorted((set(v3.index) | set(jev.index)) - gold)
    pool = [u for u in pool if u in vis.index]
    assert not (set(pool) & gold)
    a = v3.reindex(pool).values.astype(float)
    b = jev.reindex(pool).values.astype(float)
    rep = np.where(np.isnan(a), b, np.where(np.isnan(b), a, 0.5 * a + 0.5 * b))
    keep = np.isfinite(rep).all(axis=1)
    pool = [u for u, k in zip(pool, keep) if k]
    rep, a, b = rep[keep], a[keep], b[keep]
    pv = vis.loc[pool].values.astype(float)
    assert np.isfinite(pv).all()
    cal = np.zeros_like(pv)
    summary = {"pool": len(pool), "both_report_sources": int((np.isfinite(a).all(1) & np.isfinite(b).all(1)).sum()), "per_finding": {}}
    tgt = rep.copy()
    for j, c in enumerate(LAB):
        ab = platt(logit(pv[:, j]), rep[:, j])
        cal[:, j] = 1 / (1 + np.exp(-(ab[0] * logit(pv[:, j]) + ab[1])))
        if c in OVER:
            m = rep[:, j] > 0.6
            tgt[m, j] = 0.45 * rep[m, j] + 0.55 * cal[m, j]
        elif c in UNDER:
            m = (rep[:, j] >= 0.08) & (rep[:, j] <= 0.45)
            tgt[m, j] = 0.70 * cal[m, j] + 0.30 * rep[m, j]
        else:
            m = np.zeros(len(pool), bool)
        both = np.isfinite(a[:, j]) & np.isfinite(b[:, j])
        summary["per_finding"][c] = {
            "platt_a_b": [round(float(x), 4) for x in ab], "cells_changed": int(m.sum()),
            "mean_report": round(float(rep[:, j].mean()), 4), "mean_vis_cal": round(float(cal[:, j].mean()), 4),
            "mean_target": round(float(tgt[:, j].mean()), 4),
            "mean_abs_change": round(float(np.abs(tgt[:, j] - rep[:, j]).mean()), 4),
            "corr_v3_jev": round(float(np.corrcoef(a[both, j], b[both, j])[0, 1]), 4)}
    os.makedirs(f"{R}/data/labels_v4", exist_ok=True)
    out = pd.DataFrame(np.clip(tgt, 0, 1), columns=LAB)
    out.insert(0, "StudyInstanceUID", pool)
    out.to_csv(f"{R}/data/labels_v4/training_labels_v4_superior.csv", index=False)
    json.dump(summary, open(f"{R}/data/labels_v4/training_labels_v4_superior_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
