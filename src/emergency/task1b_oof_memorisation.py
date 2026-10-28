"""Task 1, Raptor memorisation (descriptive; added 2026-10-20 ~08:30 IST after the registered analyses; decides nothing).

How far do the all-data legs fit the studies they trained on? On oof_300, against labels_v3 confident cells
(<= 0.2 negative, >= 0.8 positive; macro = mean over the findings that have both classes):
  S6 held-out fold   out-of-sample: the one fold model that never trained on the study (sha1(uid)[:8] % 4)
  S6 4-fold mean     3 of its 4 fold models trained on the study
  s6fall, R384, R224 all-data models: in-sample
Each leg's gold_58 macro (out-of-sample, image reads) is copied from task2_gold_eval.json. The two sets use different
label sources, so only differences between legs within one set are read; the cross-set gap is shown, not interpreted
as a train-test gap on its own. Paired bootstrap (10,000, seed 0) of each leg minus the S6 held-out fold on oof_300.
Usage: python -I src/emergency/task1b_oof_memorisation.py
"""
import glob
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


def load(folder):
    return {str(z["uid"]): z for z in (np.load(f, allow_pickle=True) for f in glob.glob(f"{T1}/{folder}/*.npz"))}


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, M, S, idx=None):
    if idx is not None:
        Y, M, S = Y[idx], M[idx], S[idx]
    return np.nanmean([auc(Y[M[:, j], j], S[M[:, j], j]) for j in range(12)])


def main():
    st = pd.read_csv(f"{T1}/task1_studies.csv", dtype={"StudyInstanceUID": str})
    ids = list(st[st["set"] == "oof_300"].StudyInstanceUID)
    s6, s6f, rap = load("s6"), load("s6fall"), load("raptor")
    miss = {k: sum(u not in L for u in ids) for k, L in (("S6", s6), ("S6F", s6f), ("Raptor", rap))}
    assert not any(miss.values()), f"missing oof_300 outputs: {miss}"
    fold = [int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % 4 for u in ids]
    S = {"S6 held-out fold": np.stack([s6[u]["z_plain"][f] for u, f in zip(ids, fold)]),
         "S6 4-fold mean": np.stack([s6[u]["z_plain"].mean(0) for u in ids]),
         "s6fall": np.stack([s6f[u]["z_plain"][0] for u in ids]),
         "R384 pair": np.stack([rap[u]["p384_pair"] for u in ids]),
         "R224 pair": np.stack([rap[u]["p224_pair"] for u in ids])}
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    soft = lab.loc[ids, LAB].values.astype(float)
    M = (soft <= 0.2) | (soft >= 0.8)
    Y = (soft >= 0.8).astype(int)
    gold = json.load(open(f"{T1}/task2_gold_eval.json", encoding="utf-8"))["legs"]
    gold_key = {"S6 4-fold mean": "S6", "s6fall": "S6F", "R384 pair": "R384", "R224 pair": "R224"}
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(10000)]
    ref = S["S6 held-out fold"]
    res = {"n_oof": len(ids), "legs": {}}
    for k, v in S.items():
        row = {"oof300_macro": round(macro(Y, M, v), 4),
               "oof300_per_finding": {c: round(auc(Y[M[:, j], j], v[M[:, j], j]), 3) for j, c in enumerate(LAB)}}
        if k != "S6 held-out fold":
            d = np.array([macro(Y, M, v, b) - macro(Y, M, ref, b) for b in B])
            row["vs_S6_heldout"] = [round(macro(Y, M, v) - macro(Y, M, ref), 4)] + [round(x, 4) for x in np.percentile(d, [2.5, 97.5])]
        if k in gold_key:
            row["gold58_macro"] = gold[gold_key[k]]["macro"]
        res["legs"][k] = row
        print(f"{k:17s} oof_300 {row['oof300_macro']:.4f}"
              + (f"  vs S6 held-out {row['vs_S6_heldout'][0]:+.4f} [{row['vs_S6_heldout'][1]:+.4f}, {row['vs_S6_heldout'][2]:+.4f}]" if "vs_S6_heldout" in row else "")
              + (f"  | gold_58 {row['gold58_macro']:.4f}" if "gold58_macro" in row else ""))
    json.dump(res, open(f"{T1}/task1b_oof_memorisation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
