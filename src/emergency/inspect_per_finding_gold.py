"""Print comprehensive 12-class per-finding Gold-58 AUC table for:
R6 (0.9367), R7 (0.9317), S6 (0.9267), s6fall (0.9314), s6ft_v1 (0.9279), s6ft_v2 (0.9366),
Raptor 384 (0.9229), Raptor 224 (0.9160).
"""
import glob
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]

GROUPS = {
    "Group 1 (Ligaments & Menisci)": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "Group 2 (Cartilage & Bone)": ["Medial OA", "Lateral OA", "PF OA", "Fracture", "Contusion"],
    "Group 3 (Fluid & Synovium)": ["Effusion", "Baker's", "Synovitis"],
}

def sig(z):
    return 1.0 / (1.0 + np.exp(-z))

def load(folder):
    out = {}
    for f in glob.glob(f"{T1}/{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["uid"])] = {k: z[k] for k in z.files if k != "uid"}
    return out

def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def macro(Y, S):
    return np.nanmean([auc(Y[:, j], S[:, j]) for j in range(Y.shape[1])])

def main():
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    rap = load("raptor")
    s6f = load("s6fall")
    s6 = load("s6")
    s6ft_v1 = load("s6ft")
    s6ft_v2 = load("s6ft_v2")
    r6_df = pd.read_csv(f"{T1}/r6/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    r7_df = pd.read_csv(f"{T1}/r7/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in s6ft_v1 and u in s6ft_v2 and u in r6_df.index and u in r7_df.index]
    print(f"Total gold studies with all models: {len(ids)}")
    assert len(ids) == 58, f"Expected 58 studies, found {len(ids)}"

    Y = gold.loc[ids, LAB].values.astype(int)

    models = {
        "Raptor 384": np.stack([rap[u]["p384_pair"] for u in ids]),
        "Raptor 224": np.stack([rap[u]["p224_pair"] for u in ids]),
        "R6 (352px)": r6_df.loc[ids, LAB].values.astype(float),
        "R7 (384px)": r7_df.loc[ids, LAB].values.astype(float),
        "S6 (288px)": sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids])),
        "s6fall (288px)": sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
        "s6ft_v1 (288px)": sig(np.stack([s6ft_v1[u]["z_plain"][0] for u in ids])),
        "s6ft_v2 (288px)": sig(np.stack([s6ft_v2[u]["z_plain"][0] for u in ids])),
    }

    # Print positive counts per label
    pos_counts = {c: int(Y[:, j].sum()) for j, c in enumerate(LAB)}
    print("\nGold-58 Class Positives:", pos_counts)

    # Calculate AUCs
    results = {}
    for name, P in models.items():
        aucs = [auc(Y[:, j], P[:, j]) for j in range(12)]
        mean_auc = np.mean(aucs)
        results[name] = aucs + [mean_auc]

    cols = LAB + ["MACRO AUC"]
    df_res = pd.DataFrame(results, index=cols).T
    print("\n=== PER-FINDING GOLD-58 AUC TABLE ===")
    print(df_res.round(4).to_string())

    print("\n=== GROUP MEAN AUC TABLE ===")
    for gname, glabels in GROUPS.items():
        g_indices = [LAB.index(c) for c in glabels]
        g_scores = {}
        for name, P in models.items():
            g_auc = np.mean([auc(Y[:, j], P[:, j]) for j in g_indices])
            g_scores[name] = round(g_auc, 4)
        print(f"\n{gname} ({', '.join(glabels)}):")
        for mname, score in sorted(g_scores.items(), key=lambda x: -x[1]):
            print(f"  {mname:16s}: {score:.4f}")

    # Best model per finding
    print("\n=== BEST MODEL PER FINDING ===")
    for j, c in enumerate(LAB):
        best_m = max(models.keys(), key=lambda m: auc(Y[:, j], models[m][:, j]))
        best_auc = auc(Y[:, j], models[best_m][:, j])
        print(f"  {c:16s}: {best_m:16s} ({best_auc:.4f})")

if __name__ == "__main__":
    main()
