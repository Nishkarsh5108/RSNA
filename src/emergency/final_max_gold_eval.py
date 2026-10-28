"""Plan section 13, Track A: gold_58 check of the final-max blend, T1 (percentile ranks) vs T2 (0.5 T1 + 0.5 pct of the
same weights on per-leg standardised logits). The owner's rule: use whichever scores higher on gold_58.

Legs (each from its own code, run locally): R384 / R224 = nartaa CoAtNet plain + mirror pair (task1/raptor), R6 = heliosli's
v4c worker with flip TTA + stackers, exactly v27's stage (task1/r6/predictions.csv), S6F = nartaa s6fall plain (task1/s6fall).
Blend per finding (the brief's weights, fixed in plan 13):
  Raptor slot = 0.65 R384 + 0.35 R224; compact slot = 0.60 R6 + 0.40 S6F
  final = w_c * Raptor slot + (1 - w_c) * compact slot, w_c = v27's W_RAPTOR_C + Fracture 0.65, else 0.5
References (descriptive): v27's own A3 blend (compact = R6, no Fracture override) and final-strict v1.
Bootstrap: 10,000 paired resamples of the 58 studies (seed 0).
Usage: python -I src/emergency/final_max_gold_eval.py   -> output/emergency_research/task1/final_max_gold_eval.json
"""
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
W_RAPTOR_C_V27 = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35}
W_RAPTOR_C_MAX = dict(W_RAPTOR_C_V27, Fracture=0.65)
A224, W_R6, W_S6F = 0.35, 0.60, 0.40


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


def macro(Y, S, idx=None):
    if idx is not None:
        Y, S = Y[idx], S[idx]
    return np.nanmean([auc(Y[:, j], S[:, j]) for j in range(12)])


def blend(L, f, wmap, compact):
    """f: per-leg column transform (pct or zlogit); compact: list of (leg, weight)."""
    out = np.zeros((len(L["R384"]), 12))
    for j, c in enumerate(LAB):
        rr = (1 - A224) * f(L["R384"][:, j]) + A224 * f(L["R224"][:, j])
        rc = sum(w * f(L[k][:, j]) for k, w in compact)
        wr = wmap.get(c, 0.5)
        out[:, j] = wr * rr + (1 - wr) * rc
    return out


def main():
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    rap, s6f, s6 = load("raptor"), load("s6fall"), load("s6")
    r6 = pd.read_csv(f"{T1}/r6/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index]
    assert len(ids) == 58, f"only {len(ids)} gold studies have every leg"
    Y = gold.loc[ids, LAB].values.astype(int)
    L = {"R384": np.stack([rap[u]["p384_pair"] for u in ids]), "R224": np.stack([rap[u]["p224_pair"] for u in ids]),
         "R6": r6.loc[ids, LAB].values.astype(float), "S6F": sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
         "S6": sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids]))}
    assert np.isfinite(L["R6"]).all() and ((L["R6"] >= 0) & (L["R6"] <= 1)).all()
    compact = [("R6", W_R6), ("S6F", W_S6F)]
    V = {"T1": blend(L, pct, W_RAPTOR_C_MAX, compact)}
    tz = blend(L, zlogit, W_RAPTOR_C_MAX, compact)
    V["T2"] = 0.5 * V["T1"] + 0.5 * np.column_stack([pct(tz[:, j]) for j in range(12)])
    V["v27_A3"] = blend(L, pct, W_RAPTOR_C_V27, [("R6", 1.0)])                 # reference: v27's own blend
    V["final_strict_v1"] = np.column_stack([0.5 * pct(L["R384"][:, j]) + 0.25 * pct(L["S6"][:, j]) + 0.25 * pct(L["S6F"][:, j])
                                            for j in range(12)])                 # reference: our 0.956 pick
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(10000)]
    res = {"n": len(ids), "legs": {k: round(macro(Y, v), 4) for k, v in L.items()},
           "legs_per_finding": {k: dict(zip(LAB, [round(auc(Y[:, j], v[:, j]), 3) for j in range(12)])) for k, v in L.items()},
           "variants": {k: round(macro(Y, v), 4) for k, v in V.items()}, "paired": {}}
    for a_, b_ in (("T2", "T1"), ("T1", "v27_A3"), ("T1", "final_strict_v1"), ("T2", "final_strict_v1")):
        d = np.array([macro(Y, V[a_], b) - macro(Y, V[b_], b) for b in B])
        res["paired"][f"{a_} - {b_}"] = [round(macro(Y, V[a_]) - macro(Y, V[b_]), 4)] + [round(x, 4) for x in np.percentile(d, [2.5, 97.5])]
    res["chosen"] = "T2" if res["variants"]["T2"] > res["variants"]["T1"] else "T1"
    res["rule"] = "owner's rule (plan 13): whichever of T1 / T2 scores higher on gold_58; ties keep T1 (v27's transform)"
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{T1}/final_max_gold_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
