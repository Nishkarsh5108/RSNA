"""Paired comparison of saved fold-0 prediction sets (output/emergency_research/preds/*.npz).

  python src/emergency/compare_preds.py REF CAND [CAND2 ...]      (names = file stems, e.g. E1p_..__shift)
Prints macro AUC, paired delta vs REF with a 1,000-resample bootstrap CI, confident-label AUC (teacher <= 0.2 or
>= 0.8) and the Pearson correlation of predictions; appends rows to output/emergency_research/comparisons.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

R = "D:/Coding/kaggle/RSNA"
sys.path.insert(0, f"{R}/src/labeling")
from metrics import paired_macro_auc_diff, safe_auc  # noqa: E402

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
P = f"{R}/output/emergency_research/preds"


def load(name):
    z = np.load(f"{P}/{name}.npz", allow_pickle=True)
    return pd.DataFrame(z["p"], index=z["uids"], columns=TARGETS)


def conf_auc(Ysoft, p):
    vals = []
    for j in range(Ysoft.shape[1]):
        keep = (Ysoft[:, j] <= 0.2) | (Ysoft[:, j] >= 0.8)
        vals.append(safe_auc((Ysoft[keep, j] >= 0.5).astype(float), p[keep, j]))
    return float(np.nanmean(vals))


def main():
    ref, cands = sys.argv[1], sys.argv[2:]
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    pr = load(ref)
    uids = pr.index
    Ys = lab.loc[uids, TARGETS].values
    Y = (Ys >= 0.5).astype(float)
    rows = []
    for name in [ref] + cands:
        pc = load(name).loc[uids]
        per = [safe_auc(Y[:, j], pc.values[:, j]) for j in range(12)]
        row = {"time": time.strftime("%Y-%m-%d %H:%M"), "ref": ref, "cand": name, "macro_auc": float(np.nanmean(per)),
               "conf_auc": conf_auc(Ys, pc.values), **{f"auc_{t}": v for t, v in zip(TARGETS, per)}}
        if name != ref:
            d, lo, hi, ple0 = paired_macro_auc_diff(Y, pc.values, pr.values, n_boot=1000, seed=9)
            row.update({"delta": d, "delta_lo": lo, "delta_hi": hi, "p_delta_le_0": ple0,
                        "pearson_vs_ref": float(np.mean([np.corrcoef(pc.values[:, j], pr.values[:, j])[0, 1] for j in range(12)]))})
        rows.append(row)
        extra = f" delta {row['delta']:+.4f} [{row['delta_lo']:+.4f}, {row['delta_hi']:+.4f}] p<=0 {row['p_delta_le_0']:.3f} r {row['pearson_vs_ref']:.3f}" if name != ref else ""
        print(f"{name:55s} macro {row['macro_auc']:.4f} conf {row['conf_auc']:.4f}{extra}")
    out = f"{R}/output/emergency_research/comparisons.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)


if __name__ == "__main__":
    main()
