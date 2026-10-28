"""Task 2 (plan section 12): gold-58 evaluation of the public legs and of the pre-registered final-strict upgrades.

Inputs (from task1_compact_preds.py / task1_raptor_preds.py / optional reader preds): per-study npz files under
output/emergency_research/task1/{s6,s6fall,raptor}/. Labels: data/labels_v3/gold_labels_original.csv (image reads).
Every rule is the plan's section-12 rule; nothing here is tuned (D2's L2 strength C = 0.1 was fixed when this file was
written, 2026-10-20 ~01:00 IST, before any gold prediction existed).

Scores per leg (the quantities the Kaggle stages blend):
  R384, R224   nartaa CoAtNet, plain + anatomical-mirror average (p_pair)
  S6           sigmoid(mean over 4 folds of the plain logits) = heliosli's stage output;  S6m: mirror-TTA logits instead
  S6F          nartaa s6fall plain;  S6Fm: mirror-TTA
Blends (per finding, percentile ranks within the 58 studies):
  v1  = 0.5 R384 + 0.25 S6 + 0.25 S6F                              (final-strict as submitted)
  A1  = Raptor slot 0.65 R384 + 0.35 R224, then 0.5 / 0.25 / 0.25  (A2 adds the reader at A-strict+'s per-finding weights)
  B   = v1 with S6m and S6Fm
  C1  = v1 weights on probit ranks  Phi^-1((rank - 0.5) / N);  C2 = v1 weights on per-leg standardised logits
  D1  = compartment schedule (ACL/MCL/MedMen 0.55/0.225/0.225; PF OA, Med OA, Lat OA, Baker's, Synovitis, LatMen 0.40/0.30/0.30)
  D2  = v1, with Synovitis and Contusion replaced by leave-one-study-out L2 logistic calibrations on cross-finding blend
        scores (Synovitis: own, Effusion, Medial/Lateral/PF OA, ACL; Contusion: own, ACL, Fracture), C = 0.1
Bootstrap: 10,000 paired resamples of the 58 studies (seed 0); macro AUC = mean of the 12 per-finding AUCs.
Adoption (plan 12): delta >= +0.0015, CI lower >= 0, delta > 0 on >= 7 of 12 findings.

Usage: python -I src/emergency/task2_gold_eval.py [--reader <csv with StudyInstanceUID + 12 probabilities>]
Writes output/emergency_research/task1/task2_gold_eval.json and prints the tables.
"""
import argparse
import glob
import json

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
READER_W = {"Baker's": 0.035, "Contusion": 0.03, "Medial OA": 0.025, "ACL": 0.025, "Medial Meniscus": 0.02, "MCL": 0.02,
            "Lateral Meniscus": 0.008, "Fracture": 0.008, "Lateral OA": 0.008, "PF OA": 0.008, "Effusion": 0.004,
            "Synovitis": 0.003}
NOTCH = ("ACL", "MCL", "Medial Meniscus")
COMPART = ("PF OA", "Medial OA", "Lateral OA", "Baker's", "Synovitis", "Lateral Meniscus")


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def load_leg(folder):
    out = {}
    for f in glob.glob(f"{T1}/{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["uid"])] = {k: z[k] for k in z.files if k != "uid"}
    return out


def pct(v):
    return pd.Series(np.asarray(v, float)).rank(method="average", pct=True).values


def probit(v):
    r = rankdata(v, method="average")
    return norm.ppf((r - 0.5) / len(v))


def zstd(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-9)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, S, idx=None):
    if idx is not None:
        Y, S = Y[idx], S[idx]
    return np.nanmean([auc(Y[:, j], S[:, j]) for j in range(Y.shape[1])])


def blend(legs, weights, transform=pct):
    """legs: dict name -> [N, 12] score; weights: dict name -> scalar or [12] array (per finding)."""
    N = next(iter(legs.values())).shape[0]
    out = np.zeros((N, 12))
    for j in range(12):
        for name, w in weights.items():
            wj = w[j] if np.ndim(w) else w
            if wj:
                out[:, j] += wj * transform(legs[name][:, j])
    return out


def loso_calibrate(Y, S, target, feats, C=0.1):
    from sklearn.linear_model import LogisticRegression
    j = LAB.index(target)
    X = np.column_stack([logit(np.clip(S[:, LAB.index(f)], 1e-4, 1 - 1e-4)) for f in (target,) + feats])
    out = np.zeros(len(Y))
    for i in range(len(Y)):
        tr = np.arange(len(Y)) != i
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=2000).fit((X[tr] - mu) / sd, Y[tr, j])
        out[i] = m.decision_function(((X[i] - mu) / sd)[None])[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    s6, s6f, rap = load_leg("s6"), load_leg("s6fall"), load_leg("raptor")
    ids = [u for u in gold.index if u in s6 and u in s6f and u in rap]
    print(f"gold studies with every leg: {len(ids)} of {len(gold)}")
    Y = gold.loc[ids, LAB].values.astype(int)
    legs = {
        "R384": np.stack([rap[u]["p384_pair"] for u in ids]),
        "R224": np.stack([rap[u]["p224_pair"] for u in ids]),
        "R384plain": np.stack([rap[u]["p384_plain"] for u in ids]),
        "S6": sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids])),
        "S6m": sig(np.stack([np.nanmean(s6[u]["z_mirror"], 0) for u in ids])),
        "S6F": sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
        "S6Fm": sig(np.stack([s6f[u]["z_mirror"][0] for u in ids])),
    }
    if a.reader:
        rd = pd.read_csv(a.reader, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        if all(u in rd.index for u in ids):
            legs["READER"] = rd.loc[ids, LAB].values.astype(float)
    res = {"n": len(ids), "legs": {}, "variants": {}}
    print("\nsingle legs on gold (macro AUC | mean predicted prob vs prevalence | Brier):")
    prev = Y.mean(0)
    for k, v in legs.items():
        res["legs"][k] = {"macro": round(macro(Y, v), 4), "per_finding": dict(zip(LAB, [round(auc(Y[:, j], v[:, j]), 3) for j in range(12)])),
                          "mean_pred_minus_prev": round(float((v.mean(0) - prev).mean()), 3), "brier": round(float(((v - Y) ** 2).mean()), 4)}
        print(f"  {k:10s} {res['legs'][k]['macro']:.4f} | mean pred - prevalence {res['legs'][k]['mean_pred_minus_prev']:+.3f} | Brier {res['legs'][k]['brier']:.4f}")
    w_v1 = {"R384": 0.5, "S6": 0.25, "S6F": 0.25}
    V = {"v1": blend(legs, w_v1)}
    rap_slot = 0.65 * np.column_stack([pct(legs["R384"][:, j]) for j in range(12)]) + \
        0.35 * np.column_stack([pct(legs["R224"][:, j]) for j in range(12)])
    V["A1"] = blend({"RS": rap_slot, "S6": legs["S6"], "S6F": legs["S6F"]}, {"RS": 0.5, "S6": 0.25, "S6F": 0.25})
    if "READER" in legs:
        wr = np.array([READER_W[c] for c in LAB])
        V["A2"] = np.column_stack([(1 - wr[j]) * pct(V["A1"][:, j]) + wr[j] * pct(legs["READER"][:, j]) for j in range(12)])
    V["B"] = blend({"R384": legs["R384"], "S6": legs["S6m"], "S6F": legs["S6Fm"]}, w_v1)
    V["C1_probit"] = blend(legs, w_v1, transform=probit)
    lg = {"R384": logit(legs["R384"]), "S6": logit(legs["S6"]), "S6F": logit(legs["S6F"])}
    V["C2_stdlogit"] = blend(lg, w_v1, transform=zstd)
    wR = np.array([0.55 if c in NOTCH else 0.40 if c in COMPART else 0.5 for c in LAB])
    wS = np.array([0.225 if c in NOTCH else 0.30 if c in COMPART else 0.25 for c in LAB])
    V["D1_compartment"] = blend(legs, {"R384": wR, "S6": wS, "S6F": wS})
    d2 = V["v1"].copy()
    d2[:, LAB.index("Synovitis")] = loso_calibrate(Y, V["v1"], "Synovitis", ("Effusion", "Medial OA", "Lateral OA", "PF OA", "ACL"))
    d2[:, LAB.index("Contusion")] = loso_calibrate(Y, V["v1"], "Contusion", ("ACL", "Fracture"))
    V["D2_crossfinding"] = d2
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(a.boot)]
    base_full = macro(Y, V["v1"])
    base_boot = np.array([macro(Y, V["v1"], b) for b in B])
    print(f"\nbaseline v1 (final-strict as submitted): gold macro {base_full:.4f}")
    print(f"{'variant':16s} {'macro':>7s} {'delta':>8s} {'95% CI':>18s} {'+findings':>9s}  adopt?")
    for k, S in V.items():
        if k == "v1":
            continue
        full = macro(Y, S)
        boot = np.array([macro(Y, S, b) for b in B]) - base_boot
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per = [auc(Y[:, j], S[:, j]) - auc(Y[:, j], V["v1"][:, j]) for j in range(12)]
        npos = int(sum(d > 0 for d in per))
        adopt = (full - base_full >= 0.0015) and lo >= 0 and npos >= 7
        res["variants"][k] = {"macro": round(full, 4), "delta": round(full - base_full, 4), "ci95": [round(lo, 4), round(hi, 4)],
                              "findings_improved": npos, "per_finding_delta": dict(zip(LAB, [round(d, 4) for d in per])),
                              "adopt": bool(adopt)}
        print(f"{k:16s} {full:7.4f} {full - base_full:+8.4f} [{lo:+.4f}, {hi:+.4f}] {npos:6d}/12   {'YES' if adopt else 'no'}")
    res["baseline_v1_macro"] = round(base_full, 4)
    json.dump(res, open(f"{T1}/task2_gold_eval.json", "w"), indent=1)
    print(f"\nwrote {T1}/task2_gold_eval.json")


if __name__ == "__main__":
    main()
