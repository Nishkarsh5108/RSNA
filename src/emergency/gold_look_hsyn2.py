"""Pre-registered label-level gold look H-SYN2 (reports/emergency_experiment_plan.md §3.2).

Synovitis label AUC on gold-58: final teacher vs teacher with synovitis := effusion where the labeller's
order-averaged assertion for synovitis is 'not mentioned' (category C). Writes output/emergency_research/gold_look_hsyn2.json.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "D:/Coding/kaggle/RSNA"
CATS = ["pA", "pB", "pC", "pD", "pE"]


def main():
    gold_uids = pd.read_csv(f"{R}/data/labels_v3/gold_uids_LOCKED.csv").iloc[:, 0]
    # The training-label file has the gold labels copied into the gold rows, so the teacher's genuine gold-row
    # predictions are rebuilt from its sources: mean(order-averaged LLM s0, gpt56sol, report_v2), where
    # gpt56sol + report_v2 = 2 * mean_gpt56_v2 (report_v2.csv itself has no gold rows).
    TG = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
    mean_pub = pd.read_csv(f"{R}/data/labels_v3/label_sets/mean_gpt56_v2.csv").set_index("StudyInstanceUID")
    raw = pd.read_csv(f"{R}/data/labels_v3/llm_qwen3_4b_assert_v3.1_raw.csv", low_memory=False).set_index("StudyInstanceUID")
    rev = pd.read_csv(f"{R}/data/labels_v3/llm_qwen3_4b_assert_v3.1_reversed_raw.csv", low_memory=False).set_index("StudyInstanceUID")
    gold = pd.read_csv(f"{R}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
    g = list(gold_uids)
    pr = (raw.loc[g, [f"Synovitis__{c}" for c in CATS]].values + rev.loc[g, [f"Synovitis__{c}" for c in CATS]].values) / 2
    silent = pr.argmax(1) == 2
    y = gold.loc[g, "Synovitis"].values.astype(int)
    def s0(t, uids):
        p = (raw.loc[uids, [f"{t}__{c}" for c in CATS]].values + rev.loc[uids, [f"{t}__{c}" for c in CATS]].values) / 2
        return p[:, 0] + 0.5 * p[:, 1] + 0.25 * p[:, 4]
    teach = pd.DataFrame({t: (s0(t, g) + 2 * mean_pub.loc[g, t].values) / 3 for t in TG}, index=g)
    macro = float(np.mean([roc_auc_score(gold.loc[g, t].values.astype(int), teach[t].values) for t in TG]))
    a = teach["Synovitis"].values
    b = np.where(silent, teach["Effusion"].values, a)
    # also the training corpus: how many cells would change
    allu = raw.index
    pra = (raw.loc[allu, [f"Synovitis__{c}" for c in CATS]].values + rev.loc[allu, [f"Synovitis__{c}" for c in CATS]].values) / 2
    out = {"n_gold": len(g), "teacher_macro_auc_check": macro, "gold_silent_cells": int(silent.sum()),
           "auc_teacher": float(roc_auc_score(y, a)), "auc_filled": float(roc_auc_score(y, b)),
           "delta": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
           "gold_positive_rate_when_silent": float(y[silent].mean()) if silent.any() else None,
           "corpus_silent_fraction": float((pra.argmax(1) == 2).mean())}
    json.dump(out, open(f"{R}/output/emergency_research/gold_look_hsyn2.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
