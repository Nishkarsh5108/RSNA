"""Simulate and evaluate final-max v3 and final-strict-v2 blends on Gold-58.
Explores:
1. Standard final-max blend replacing s6fall with s6ft_v2 (or 50/50 s6fall + s6ft_v2)
2. Specialist compact slot: per-finding combination of R6, R7, s6ft_v2, s6ft_v1, s6fall
3. final-strict-v2 blend with s6ft_v2 and s6ft_v1
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]

W_RAPTOR_C_V27 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
W_RAPTOR_C_MAX = dict(W_RAPTOR_C_V27, Fracture=0.65)
A224 = 0.35

def sig(z):
    return 1.0 / (1.0 + np.exp(-z))

def load(folder):
    out = {}
    for f in glob.glob(f"{T1}/{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["uid"])] = {k: z[k] for k in z.files if k != "uid"}
    return out

def pct(v):
    return pd.Series(np.asarray(v, float)).rank(method="average", pct=True).values

def zlogit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return (z - z.mean()) / (z.std() + 1e-9)

def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def macro(Y, S):
    return np.nanmean([auc(Y[:, j], S[:, j]) for j in range(12)])

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
    Y = gold.loc[ids, LAB].values.astype(int)

    L = {
        "R384": np.stack([rap[u]["p384_pair"] for u in ids]),
        "R224": np.stack([rap[u]["p224_pair"] for u in ids]),
        "R6": r6_df.loc[ids, LAB].values.astype(float),
        "R7": r7_df.loc[ids, LAB].values.astype(float),
        "S6": sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids])),
        "S6F": sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
        "S6FT1": sig(np.stack([s6ft_v1[u]["z_plain"][0] for u in ids])),
        "S6FT2": sig(np.stack([s6ft_v2[u]["z_plain"][0] for u in ids])),
    }

    # Evaluate Raptor slot
    raptor_slot_probs = (1 - A224) * L["R384"] + A224 * L["R224"]

    def blend_t2(compact_probs):
        # T1
        b1 = np.zeros((len(ids), 12))
        bz = np.zeros((len(ids), 12))
        for j, c in enumerate(LAB):
            rr_pct = (1 - A224) * pct(L["R384"][:, j]) + A224 * pct(L["R224"][:, j])
            rc_pct = pct(compact_probs[:, j])
            wr = W_RAPTOR_C_MAX.get(c, 0.5)
            b1[:, j] = wr * rr_pct + (1 - wr) * rc_pct

            rr_z = (1 - A224) * zlogit(L["R384"][:, j]) + A224 * zlogit(L["R224"][:, j])
            rc_z = zlogit(compact_probs[:, j])
            bz[:, j] = wr * rr_z + (1 - wr) * rc_z

        b2 = np.column_stack([pct(bz[:, j]) for j in range(12)])
        return 0.5 * b1 + 0.5 * b2

    print("=== FINAL-MAX VARIANTS ON GOLD-58 ===")
    # Base: 0.60 R6 + 0.40 S6F
    c_base = 0.60 * L["R6"] + 0.40 * L["S6F"]
    score_base = macro(Y, blend_t2(c_base))
    print(f"Base (R6 0.60 + s6fall 0.40): {score_base:.4f}")

    # V1: 0.60 R6 + 0.40 S6FT2
    c_v1 = 0.60 * L["R6"] + 0.40 * L["S6FT2"]
    score_v1 = macro(Y, blend_t2(c_v1))
    print(f"V1 (R6 0.60 + s6ft_v2 0.40): {score_v1:.4f} (delta: {score_v1 - score_base:+.4f})")

    # V2: 0.60 R6 + 0.20 S6F + 0.20 S6FT2
    c_v2 = 0.60 * L["R6"] + 0.20 * L["S6F"] + 0.20 * L["S6FT2"]
    score_v2 = macro(Y, blend_t2(c_v2))
    print(f"V2 (R6 0.60 + s6fall 0.20 + s6ft_v2 0.20): {score_v2:.4f} (delta: {score_v2 - score_base:+.4f})")

    # V3 (with s6ft_v1 specialist): 0.60 R6 + 0.25 S6FT2 + 0.15 S6FT1
    c_v3 = 0.60 * L["R6"] + 0.25 * L["S6FT2"] + 0.15 * L["S6FT1"]
    score_v3 = macro(Y, blend_t2(c_v3))
    print(f"V3 (R6 0.60 + s6ft_v2 0.25 + s6ft_v1 0.15): {score_v3:.4f} (delta: {score_v3 - score_base:+.4f})")

    # V4 (Trio blend): 0.60 R6 + 0.20 S6FT2 + 0.10 S6FT1 + 0.10 S6F
    c_v4 = 0.60 * L["R6"] + 0.20 * L["S6FT2"] + 0.10 * L["S6FT1"] + 0.10 * L["S6F"]
    score_v4 = macro(Y, blend_t2(c_v4))
    print(f"V4 (R6 0.60 + s6ft_v2 0.20 + s6ft_v1 0.10 + s6fall 0.10): {score_v4:.4f} (delta: {score_v4 - score_base:+.4f})")

    # V5 (Per-finding specialist compact slot):
    # Where s6ft_v1 excels (MCL, Lat Meniscus): boost s6ft_v1
    # Where s6ft_v2 excels (Effusion, Baker's, Synovitis, Med Meniscus, Contusion): boost s6ft_v2
    c_spec = np.zeros_like(L["R6"])
    for j, c in enumerate(LAB):
        if c in ["MCL", "Lateral Meniscus"]:
            # s6ft_v1 is king
            c_spec[:, j] = 0.55 * L["R6"][:, j] + 0.30 * L["S6FT1"][:, j] + 0.15 * L["S6FT2"][:, j]
        elif c in ["Effusion", "Baker's", "Synovitis"]:
            # s6ft_v2 is king
            c_spec[:, j] = 0.55 * L["R6"][:, j] + 0.35 * L["S6FT2"][:, j] + 0.10 * L["S6FT1"][:, j]
        elif c in ["Medial OA", "Lateral OA", "PF OA", "Fracture"]:
            # Cartilage / bone
            c_spec[:, j] = 0.60 * L["R6"][:, j] + 0.20 * L["S6FT2"][:, j] + 0.10 * L["S6FT1"][:, j] + 0.10 * L["S6F"][:, j]
        else: # ACL, Contusion
            c_spec[:, j] = 0.60 * L["R6"][:, j] + 0.25 * L["S6FT2"][:, j] + 0.15 * L["S6F"][:, j]

    score_spec = macro(Y, blend_t2(c_spec))
    print(f"V5 (Per-finding Specialist Compact Slot): {score_spec:.4f} (delta: {score_spec - score_base:+.4f})")

    # With R7 included (0.45 R6 + 0.15 R7 + 0.25 S6FT2 + 0.15 S6FT1):
    c_r7 = 0.45 * L["R6"] + 0.15 * L["R7"] + 0.25 * L["S6FT2"] + 0.15 * L["S6FT1"]
    score_r7 = macro(Y, blend_t2(c_r7))
    print(f"V6 (R6 0.45 + R7 0.15 + s6ft_v2 0.25 + s6ft_v1 0.15): {score_r7:.4f} (delta: {score_r7 - score_base:+.4f})")

    print("\n=== FINAL-STRICT-V2 VARIANTS ON GOLD-58 ===")
    # Base: 0.50 Raptor + 0.25 S6 + 0.25 S6F
    strict_base = np.column_stack([0.50 * pct(L["R384"][:, j]) + 0.25 * pct(L["S6"][:, j]) + 0.25 * pct(L["S6F"][:, j]) for j in range(12)])
    score_strict_base = macro(Y, strict_base)
    print(f"Strict Base (Raptor 0.50 + S6 0.25 + s6fall 0.25): {score_strict_base:.4f}")

    # Strict V1: 0.50 Raptor + 0.25 S6 + 0.25 S6FT2
    strict_v1 = np.column_stack([0.50 * pct(L["R384"][:, j]) + 0.25 * pct(L["S6"][:, j]) + 0.25 * pct(L["S6FT2"][:, j]) for j in range(12)])
    score_strict_v1 = macro(Y, strict_v1)
    print(f"Strict V1 (Raptor 0.50 + S6 0.25 + s6ft_v2 0.25): {score_strict_v1:.4f} (delta: {score_strict_v1 - score_strict_base:+.4f})")

    # Strict V2: 0.50 Raptor + 0.25 S6 + 0.15 S6FT2 + 0.10 S6FT1
    strict_v2 = np.column_stack([0.50 * pct(L["R384"][:, j]) + 0.25 * pct(L["S6"][:, j]) + 0.15 * pct(L["S6FT2"][:, j]) + 0.10 * pct(L["S6FT1"][:, j]) for j in range(12)])
    score_strict_v2 = macro(Y, strict_v2)
    print(f"Strict V2 (Raptor 0.50 + S6 0.25 + s6ft_v2 0.15 + s6ft_v1 0.10): {score_strict_v2:.4f} (delta: {score_strict_v2 - score_strict_base:+.4f})")

    # Strict V3 (with Raptor 224 + T2 transform):
    def strict_t2(w_rap384, w_rap224, w_s6, w_ft2, w_ft1, w_s6f):
        b1 = np.zeros((len(ids), 12))
        bz = np.zeros((len(ids), 12))
        for j in range(12):
            b1[:, j] = (w_rap384 * pct(L["R384"][:, j]) + w_rap224 * pct(L["R224"][:, j]) +
                        w_s6 * pct(L["S6"][:, j]) + w_ft2 * pct(L["S6FT2"][:, j]) +
                        w_ft1 * pct(L["S6FT1"][:, j]) + w_s6f * pct(L["S6F"][:, j]))
            bz[:, j] = (w_rap384 * zlogit(L["R384"][:, j]) + w_rap224 * zlogit(L["R224"][:, j]) +
                        w_s6 * zlogit(L["S6"][:, j]) + w_ft2 * zlogit(L["S6FT2"][:, j]) +
                        w_ft1 * zlogit(L["S6FT1"][:, j]) + w_s6f * zlogit(L["S6F"][:, j]))
        b2 = np.column_stack([pct(bz[:, j]) for j in range(12)])
        return 0.5 * b1 + 0.5 * b2

    strict_t2_score = macro(Y, strict_t2(0.35, 0.15, 0.25, 0.15, 0.10, 0.0))
    print(f"Strict V3 (T2 transform: R384 0.35, R224 0.15, S6 0.25, s6ft_v2 0.15, s6ft_v1 0.10): {strict_t2_score:.4f} (delta: {strict_t2_score - score_strict_base:+.4f})")

if __name__ == "__main__":
    main()
