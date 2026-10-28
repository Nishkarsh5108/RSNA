"""Plan section 13, Track D: gold_58 adoption test of s6ft (our fine-tune of s6fall) for both final picks.

Legs as in final_max_gold_eval.py, plus S6FT = s6ft plain logits (task1/s6ft, from task1_compact_preds.py --leg s6ft).
Pick 1, final-max (T2, as chosen in 13a): base compact slot 0.60 R6 + 0.40 S6F
  V1: 0.60 R6 + 0.40 S6FT          V2: 0.60 R6 + 0.20 S6F + 0.20 S6FT
Pick 2, final-strict-v2 (percentile ranks): base 0.50 Raptor + 0.25 S6 + 0.25 S6F (final-strict v1, 0.956 LB)
  V1: 0.50 Raptor + 0.25 S6 + 0.25 S6FT          V2: 0.50 Raptor + 0.25 S6 + 0.125 S6F + 0.125 S6FT
Rule (plan 13, registered 18:21 IST before any s6ft score existed): a variant passes when the paired-bootstrap 95 % CI
of (variant - base) gold macro AUC has its lower bound >= 0. If both pass, the one with the larger delta; if neither
passes, the pick stays as it is. Bootstrap: 10,000 paired resamples of the 58 studies (seed 0).
Usage: python -I src/emergency/s6ft_gold_eval.py [leg folder, default s6ft]   -> output/emergency_research/task1/<leg>_gold_eval.json
(plan 14 C: the same rule for s6ft_v2, `s6ft_gold_eval.py s6ft_v2`)
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
    leg = sys.argv[1] if len(sys.argv) > 1 else "s6ft"
    gold = pd.read_csv(f"{FM.R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    rap, s6f, s6, s6ft = FM.load("raptor"), FM.load("s6fall"), FM.load("s6"), FM.load(leg)
    r6 = pd.read_csv(f"{T1}/r6/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index and u in s6ft]
    assert len(ids) == 58, f"only {len(ids)} gold studies have every leg"
    Y = gold.loc[ids, LAB].values.astype(int)
    L = {"R384": np.stack([rap[u]["p384_pair"] for u in ids]), "R224": np.stack([rap[u]["p224_pair"] for u in ids]),
         "R6": r6.loc[ids, LAB].values.astype(float), "S6F": FM.sig(np.stack([s6f[u]["z_plain"][0] for u in ids])),
         "S6": FM.sig(np.stack([s6[u]["z_plain"].mean(0) for u in ids])),
         "S6FT": FM.sig(np.stack([s6ft[u]["z_plain"][0] for u in ids]))}
    assert all(np.isfinite(v).all() for v in L.values())

    def t2(compact):
        b1 = FM.blend(L, FM.pct, FM.W_RAPTOR_C_MAX, compact)
        bz = FM.blend(L, FM.zlogit, FM.W_RAPTOR_C_MAX, compact)
        return 0.5 * b1 + 0.5 * np.column_stack([FM.pct(bz[:, j]) for j in range(12)])

    def strict(w_s6f, w_s6ft):
        return np.column_stack([0.5 * FM.pct(L["R384"][:, j]) + 0.25 * FM.pct(L["S6"][:, j]) +
                                w_s6f * FM.pct(L["S6F"][:, j]) + w_s6ft * FM.pct(L["S6FT"][:, j]) for j in range(12)])

    picks = {"final_max": {"base": t2([("R6", 0.6), ("S6F", 0.4)]), "V1": t2([("R6", 0.6), ("S6FT", 0.4)]),
                           "V2": t2([("R6", 0.6), ("S6F", 0.2), ("S6FT", 0.2)])},
             "final_strict_v2": {"base": strict(0.25, 0.0), "V1": strict(0.0, 0.25), "V2": strict(0.125, 0.125)}}
    rng = np.random.default_rng(0)
    B = [rng.integers(0, len(ids), len(ids)) for _ in range(10000)]
    res = {"n": len(ids), "legs": {k: round(FM.macro(Y, v), 4) for k, v in L.items()},
           "legs_per_finding": {k: dict(zip(LAB, [round(FM.auc(Y[:, j], v[:, j]), 3) for j in range(12)]))
                                for k, v in L.items() if k in ("S6F", "S6FT")}, "picks": {}}
    for name, V in picks.items():
        r = {"gold": {k: round(FM.macro(Y, v), 4) for k, v in V.items()}, "paired": {}, "pass": {}}
        for k in ("V1", "V2"):
            d = np.array([FM.macro(Y, V[k], b) - FM.macro(Y, V["base"], b) for b in B])
            lo, hi = np.percentile(d, [2.5, 97.5])
            r["paired"][f"{k} - base"] = [round(FM.macro(Y, V[k]) - FM.macro(Y, V["base"]), 4), round(lo, 4), round(hi, 4)]
            r["pass"][k] = bool(lo >= 0)
        ok = [k for k in ("V1", "V2") if r["pass"][k]]
        r["adopt"] = max(ok, key=lambda k: r["paired"][f"{k} - base"][0]) if ok else None
        res["picks"][name] = r
    res["rule"] = "plan 13: CI lower bound >= 0; both pass -> the larger delta; none -> the pick is unchanged"
    res["leg"] = leg
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{T1}/{leg}_gold_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
