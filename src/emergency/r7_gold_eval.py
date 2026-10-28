"""Plan section 14 B: gold_58 test of R7 (heliosli bundle 8c7b666e) in final-max's compact slot.

Legs as in final_max_gold_eval.py, plus R7 = task1/r7/predictions.csv (r7_gold_preds.py: R6's v4c worker, v27's arguments,
flip TTA, R7's own fluid and state stackers). Transform T2 as chosen in 13a, Raptor weights W_RAPTOR_C_MAX.
  base: compact 0.60 R6 + 0.40 S6F (final-max v1 / v2)
  C1:   compact 0.60 R7 + 0.40 S6F (R7 replaces R6 at equal cost; the only candidate that fits 9 h)
Information only (do not fit 9 h next to v27; never adopted): M1 = 0.35 R6 + 0.35 R7 + 0.15 S6 + 0.15 S6F (the brief's),
  R6R7 = 0.30 R6 + 0.30 R7 + 0.40 S6F.
Rule (plan 14 B, registered 06:10 IST before this file ran): C1 is adopted when the paired-bootstrap 95 % CI of
(C1 - base) gold macro AUC has its lower bound >= 0. Bootstrap: 10,000 paired resamples of the 58 studies (seed 0).
Usage: python -I src/emergency/r7_gold_eval.py   -> output/emergency_research/task1/r7_gold_eval.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # -I leaves the script's folder off sys.path
import final_max_gold_eval as FM

LAB, T1 = FM.LAB, FM.T1


def main():
    gold = pd.read_csv(f"{FM.R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    rap, s6f, s6 = FM.load("raptor"), FM.load("s6fall"), FM.load("s6")
    r6 = pd.read_csv(f"{T1}/r6/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    r7 = pd.read_csv(f"{T1}/r7/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index and u in r7.index]
    assert len(ids) == 58, f"only {len(ids)} gold studies have every leg"
    Y = gold.loc[ids, LAB].values.astype(int)
    L = {"R384": np.stack([rap[u]["p384_pair"] for u in ids]), "R224": np.stack([rap[u]["p224_pair"] for u in ids]),
         "R6": r6.loc[ids, LAB].values.astype(float), "R7": r7.loc[ids, LAB].values.astype(float),
         "S6F": FM.sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
         "S6": FM.sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids]))}
    assert all(np.isfinite(v).all() for v in L.values())

    def t2(compact):
        b1 = FM.blend(L, FM.pct, FM.W_RAPTOR_C_MAX, compact)
        bz = FM.blend(L, FM.zlogit, FM.W_RAPTOR_C_MAX, compact)
        return 0.5 * b1 + 0.5 * np.column_stack([FM.pct(bz[:, j]) for j in range(12)])

    V = {"base": t2([("R6", 0.6), ("S6F", 0.4)]), "C1": t2([("R7", 0.6), ("S6F", 0.4)]),
         "M1_info": t2([("R6", 0.35), ("R7", 0.35), ("S6", 0.15), ("S6F", 0.15)]),
         "R6R7_info": t2([("R6", 0.3), ("R7", 0.3), ("S6F", 0.4)])}
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(10000)]
    res = {"n": len(ids), "legs": {k: round(FM.macro(Y, v), 4) for k, v in L.items()},
           "legs_per_finding": {k: dict(zip(LAB, [round(FM.auc(Y[:, j], v[:, j]), 3) for j in range(12)]))
                                for k, v in L.items() if k in ("R6", "R7", "S6F")},
           "spearman_r6_r7": {c: round(float(pd.Series(L["R6"][:, j]).corr(pd.Series(L["R7"][:, j]), method="spearman")), 3)
                              for j, c in enumerate(LAB)},
           "variants": {k: round(FM.macro(Y, v), 4) for k, v in V.items()}, "paired": {}}
    for k in ("C1", "M1_info", "R6R7_info"):
        d = np.array([FM.macro(Y, V[k], b) - FM.macro(Y, V["base"], b) for b in B])
        lo, hi = np.percentile(d, [2.5, 97.5])
        res["paired"][f"{k} - base"] = [round(FM.macro(Y, V[k]) - FM.macro(Y, V["base"]), 4), round(lo, 4), round(hi, 4)]
    d = np.array([FM.macro(Y, L["R7"], b) - FM.macro(Y, L["R6"], b) for b in B])
    res["paired"]["R7 - R6 (legs)"] = [round(FM.macro(Y, L["R7"]) - FM.macro(Y, L["R6"]), 4)] + [round(x, 4) for x in np.percentile(d, [2.5, 97.5])]
    res["adopt_C1"] = bool(res["paired"]["C1 - base"][1] >= 0)
    res["rule"] = "plan 14 B: C1 adopted iff the CI lower bound of (C1 - base) >= 0; M1 / R6R7 are information only (runtime)"
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{T1}/r7_gold_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
