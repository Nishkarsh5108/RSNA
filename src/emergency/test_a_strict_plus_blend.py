"""CPU test of the edited blend cell (cell 12) of the A-strict+ notebook on fake stage outputs.

Checks: it runs; the output equals a hand computation of 0.7 * (0.65 * r384 + 0.35 * r224) + 0.3 * rS6 per finding
(percentile ranks), including the fallback rows (a study S6 did not score takes Raptor's rank).
Usage: python -I src/emergency/test_a_strict_plus_blend.py notebooks/candidate_A/a_strict_plus/rsna-knee-a-strict-plus.ipynb <empty work_dir>
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

nb = json.load(open(sys.argv[1], encoding="utf-8"))
code = "".join(nb["cells"][12]["source"])
assert "W_RAPTOR, W_OURS = 0.7, 0.3" in code and "W_RAPTOR_C = {}" in code

root = Path(sys.argv[2])
DATA, WORK, pred = root / "data", root / "work", root / "pred"
for p in (DATA, WORK, pred / "worker0"):
    p.mkdir(parents=True, exist_ok=True)
labels = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
rng = np.random.default_rng(0)
ids = [f"1.2.3.{i}" for i in range(7)]
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
pd.DataFrame({"StudyInstanceUID": ids, **{c: 0.5 for c in labels}}).to_csv(DATA / "sample_submission.csv", index=False)


def frame(v):
    return pd.DataFrame({"StudyInstanceUID": ids, **{c: v[:, j] for j, c in enumerate(labels)}})


p384, p224, ps6 = (rng.random((7, 12)) for _ in range(3))
frame(p384).to_csv(WORK / "raptor_raw.csv", index=False)
frame(p384).to_csv(WORK / "raptor_sub.csv", index=False)       # same order as raw, so the Spearman check passes
frame(p224).to_csv(WORK / "raptor224_sub.csv", index=False)
ps6_sub = ps6.copy()
ps6_sub[6] = ps6[:6].mean(axis=0)                               # study 6 not scored by S6: filled with the column mean
frame(ps6_sub).to_csv(WORK / "ours_sub.csv", index=False)
with open(pred / "worker0" / "predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["StudyInstanceUID"] + labels)
    for i in range(6):                                          # the worker reports only the 6 scored studies
        w.writerow([ids[i]] + [f"{x:.6f}" for x in ps6[i]])

g = {"DATA": DATA, "WORK": WORK, "pred": pred, "R224_OK": True, "RAPTOR_SECONDS": 1.0, "NB_T0": time.time() - 5,
     "json": json, "time": time, "display": print}
exec(compile(code, "cell12", "exec"), g)

out = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str})
assert out["StudyInstanceUID"].tolist() == ids


def pct(v):
    return pd.Series(v).rank(method="average", pct=True).values


exp = np.zeros((7, 12))
for j in range(12):
    rr = 0.65 * pct(p384[:, j]) + 0.35 * pct(p224[:, j])
    ro = np.full(7, np.nan)
    ro[:6] = pct(ps6[:6, j])                                    # S6 ranked over its own scored rows only
    b = 0.7 * rr + 0.3 * ro
    b[6] = rr[6]                                                # unscored by S6 -> Raptor's rank
    exp[:, j] = b
diff = np.abs(out[labels].values - exp).max()
print("max |submission - hand computation| =", diff)
assert diff < 1e-6
print("BLEND_CELL_TEST_PASSED")
