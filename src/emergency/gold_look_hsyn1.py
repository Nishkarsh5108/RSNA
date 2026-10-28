"""Pre-registered gold look H-SYN1 (reports/emergency_experiment_plan.md §3.1).

Synovitis AUC on gold-58 of (a) the synovitis output and (b) the mean within-set rank of the synovitis
and effusion outputs, for the sealed F1+F2+F3 ensemble (equal-weight probability mean, no TTA).
No weights are fitted. Writes output/emergency_research/gold_look_hsyn1.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

R = "D:/Coding/kaggle/RSNA"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RUNS = ["F1_cnxtT_s3_k8_attn12_pf_all_s42", "F2_cnxtT_s3_k8_attn12_pf_all_s7", "F3_cnxtT_s3_k8_attn12_pf_all_s11"]


def boot_auc_diff(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ps, uids = [], None
    for r in RUNS:
        z = np.load(f"{R}/output/imaging_v5/runs/{r}/gold_preds_SEALED.npz", allow_pickle=True)
        if uids is None:
            uids = z["uids"]
        assert (z["uids"] == uids).all()
        ps.append(z["p"])
    p = np.mean(ps, 0)
    gold = pd.read_csv(f"{R}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID").loc[uids, TARGETS]
    y = gold["Synovitis"].values.astype(int)
    syn, eff = p[:, TARGETS.index("Synovitis")], p[:, TARGETS.index("Effusion")]
    blend = (rankdata(syn) + rankdata(eff)) / 2
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "auc_synovitis_output": float(roc_auc_score(y, syn)),
           "auc_effusion_output_for_synovitis": float(roc_auc_score(y, eff)),
           "auc_rank_mean_syn_eff": float(roc_auc_score(y, blend)),
           "delta_ci95_blend_minus_syn": boot_auc_diff(y, syn, blend)}
    os.makedirs(f"{R}/output/emergency_research", exist_ok=True)
    json.dump(out, open(f"{R}/output/emergency_research/gold_look_hsyn1.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
