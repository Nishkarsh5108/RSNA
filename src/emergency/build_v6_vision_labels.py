"""Track 1 Step 1B: Build pure vision-distilled targets training_labels_v6_vision_pure.csv.

Combines three vision teachers:
  P_r6_oof (0.9367 Gold), P_s6_oof (0.9267 Gold), P_s6ft_v2 (0.9366 Gold)
Quantile-calibrated per finding onto the clinical report distribution.
Then ditches false negative report omissions on Synovitis, Contusion, Lateral Meniscus,
Medial Meniscus, Fracture, ACL when P_report < 0.25.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

R = "D:/Coding/kaggle/RSNA"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
OMISSION_FINDINGS = {"Synovitis", "Contusion", "Lateral Meniscus", "Medial Meniscus", "Fracture", "ACL"}

JEV_PATH = "D:/rsna_work/public_weights/rsna-knee-s6fall-weights/training_code/labels/labels_v1_blendjev.parquet"
V3_PATH = f"{R}/data/labels_v3/training_labels_v3.csv"
GOLD_PATH = f"{R}/data/labels_v3/gold_labels_original.csv"

R6_OOF_PATH = f"{R}/output/emergency_research/track2/r6_oof_cache.csv"
S6_OOF_PATH = f"{R}/output/emergency_research/track2/s6_oof_cache.csv"
S6FT_V2_PATH = f"{R}/output/emergency_research/track2/s6ft_v2_cache.csv"

OUT_CSV = f"{R}/data/labels_v4/training_labels_v6_vision_pure.csv"
OUT_SUMMARY = f"{R}/data/labels_v4/training_labels_v6_vision_pure_summary.json"


def quantile_map(src, ref):
    """Map the ranks of src onto the sorted values of ref per column."""
    mapped = np.empty_like(src)
    for j in range(src.shape[1]):
        order = np.argsort(src[:, j], kind="mergesort")
        mapped[order, j] = np.sort(ref[:, j])
    return mapped


def main():
    gold = set(pd.read_csv(GOLD_PATH, dtype={"StudyInstanceUID": str}).StudyInstanceUID)
    v3 = pd.read_csv(V3_PATH, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    jev = pd.read_parquet(JEV_PATH).set_index("StudyInstanceUID")[LAB]

    # Load vision teachers
    assert os.path.exists(R6_OOF_PATH), f"Missing {R6_OOF_PATH}"
    assert os.path.exists(S6_OOF_PATH), f"Missing {S6_OOF_PATH}"
    assert os.path.exists(S6FT_V2_PATH), f"Missing {S6FT_V2_PATH}"

    r6_df = pd.read_csv(R6_OOF_PATH, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    s6_df = pd.read_csv(S6_OOF_PATH, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    s6ft_df = pd.read_csv(S6FT_V2_PATH, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]

    # Study pool (non-gold intersection)
    pool = sorted((set(r6_df.index) & set(s6_df.index) & set(s6ft_df.index) & (set(v3.index) | set(jev.index))) - gold)
    assert not (set(pool) & gold), "CRITICAL: Gold study found in training pool!"
    print(f"Total non-gold training studies in pool: {len(pool)}", flush=True)

    # 1. Report reference P_rep
    a = v3.reindex(pool).values.astype(float)
    b = jev.reindex(pool).values.astype(float)
    rep = np.where(np.isnan(a), b, np.where(np.isnan(b), a, 0.5 * a + 0.5 * b))
    keep = np.isfinite(rep).all(axis=1)
    pool = [u for u, k in zip(pool, keep) if k]
    rep = rep[keep]
    print(f"Studies with valid report reference: {len(pool)}", flush=True)

    # 2. Extract vision predictions
    p_r6 = r6_df.loc[pool].values.astype(float)
    p_s6 = s6_df.loc[pool].values.astype(float)
    p_s6ft = s6ft_df.loc[pool].values.astype(float)

    assert np.isfinite(p_r6).all() and np.isfinite(p_s6).all() and np.isfinite(p_s6ft).all()

    # 3. Quantile calibration onto report marginals
    qm_r6 = quantile_map(p_r6, rep)
    qm_s6 = quantile_map(p_s6, rep)
    qm_s6ft = quantile_map(p_s6ft, rep)

    # Super vision teacher
    p_vis_super = 0.50 * qm_r6 + 0.25 * qm_s6 + 0.25 * qm_s6ft

    # 4. Target assembly with report false-negative correction
    targets = np.zeros_like(rep)
    omission_counts = {}

    for j, c in enumerate(LAB):
        p_vis_c = p_vis_super[:, j]
        p_rep_c = rep[:, j]

        if c in OMISSION_FINDINGS:
            # Report says negative (< 0.25): do NOT let report pull down visual pathology
            mask_neg = (p_rep_c < 0.25)
            targets[:, j] = 0.75 * p_vis_c + 0.25 * p_rep_c
            # On negative report where vision detected finding: boost to max(p_vis, 0.80*p_vis + 0.20*p_rep)
            targets[mask_neg, j] = np.maximum(p_vis_c[mask_neg], 0.80 * p_vis_c[mask_neg] + 0.20 * p_rep_c[mask_neg])
            omission_counts[c] = int((mask_neg & (p_vis_c > 0.40)).sum())
        else:
            targets[:, j] = 0.75 * p_vis_c + 0.25 * p_rep_c

    targets = np.clip(targets, 0.0, 1.0)

    # Save
    os.makedirs(f"{R}/data/labels_v4", exist_ok=True)
    out_df = pd.DataFrame(targets, columns=LAB)
    out_df.insert(0, "StudyInstanceUID", pool)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(out_df)} studies.", flush=True)

    summary = {
        "studies": len(pool),
        "weights": {"r6_oof": 0.50, "s6_oof": 0.25, "s6ft_v2": 0.25},
        "blend_ratio": "75% vision + 25% report (with 100% vision override on omitted negatives)",
        "omission_corrections_pvis_gt_0.4": omission_counts,
        "per_finding_means": {
            c: {
                "report_mean": round(float(rep[:, j].mean()), 4),
                "vis_super_mean": round(float(p_vis_super[:, j].mean()), 4),
                "target_mean": round(float(targets[:, j].mean()), 4),
                "corr_vis_rep": round(float(pd.Series(p_vis_super[:, j]).corr(pd.Series(rep[:, j]))), 4)
            }
            for j, c in enumerate(LAB)
        }
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
