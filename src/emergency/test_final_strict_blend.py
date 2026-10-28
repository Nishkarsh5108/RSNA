"""CPU test of the final-strict blend cell (Raptor 0.5 / S6 0.25 / s6fall 0.25, else A-strict's 0.7 / 0.3) on fake outputs.

Cases (each checked against a hand computation from percentile ranks):
  1. s6fall not run          -> identical to the unmodified A-strict blend cell on the same inputs
  2. s6fall used             -> 0.5 rR + 0.25 rS6 + 0.25 rS6F; Raptor-failed / S6-failed studies renormalised over the rest
  3. s6fall failed on 1 of 40 studies (<= 3 %) -> that study takes A-strict's 0.7 / 0.3, the others the 3-leg blend
  4. s6fall failed on 3 of 40 studies (> 3 %)  -> s6fall dropped: identical to case 1
Usage: python -I src/emergency/test_final_strict_blend.py <final-strict ipynb> <a-strict ipynb> <empty work dir>
"""
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


def cell(path, head):
    nb = json.load(open(path, encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith(head))


new_code = cell(sys.argv[1], "# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5")
old_code = cell(sys.argv[2], "# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.7")
root = Path(sys.argv[3])
N = 40
ids = [f"1.2.3.{i}" for i in range(N)]
rng = np.random.default_rng(1)


def pct(v):
    return pd.Series(np.asarray(v, float)).rank(method="average", pct=True).values


def setup(case_dir, s6f_rows_failed):
    DATA, WORK, pred = case_dir / "data", case_dir / "work", case_dir / "pred"
    for p in (DATA, WORK / "k1" / "B", pred / "worker0"):
        p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{c: 0.5 for c in LAB}}).to_csv(DATA / "sample_submission.csv", index=False)
    pr, ps, pf = (rng.random((N, 12)) for _ in range(3))
    pr[5] = 0.5                                                      # Raptor failed on study 5 (all 0.5)
    ps_sub = ps.copy()
    ps_sub[7] = np.delete(ps, 7, axis=0).mean(axis=0)                # S6 failed on study 7 (column-mean fill)

    def frame(v):
        return pd.DataFrame({"StudyInstanceUID": ids, **{c: v[:, j] for j, c in enumerate(LAB)}})
    frame(pr).to_csv(WORK / "raptor_raw.csv", index=False)
    frame(pr).to_csv(WORK / "raptor_sub.csv", index=False)
    frame(ps_sub).to_csv(WORK / "ours_sub.csv", index=False)
    with open(pred / "worker0" / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID"] + LAB)
        for i in range(N):
            if i != 7:
                w.writerow([ids[i]] + [f"{x:.8f}" for x in ps[i]])
    pf_out = pf.copy()
    pf_out[list(s6f_rows_failed)] = np.nan
    b = frame(pf_out)
    for j, c in enumerate(LAB):
        b["z|" + c] = 0.0
    b.to_csv(WORK / "k1" / "B" / "B_probs.csv", index=False)
    return DATA, WORK, pred, pr, ps_sub, pf


def run(code, DATA, WORK, pred, s6f_ok):
    g = {"DATA": DATA, "WORK": WORK, "pred": pred, "RAPTOR_SECONDS": 1.0, "NB_T0": time.time() - 5, "json": json,
         "time": time, "display": lambda *a, **k: None, "S6F_OK": s6f_ok, "S6F_PROBS": WORK / "k1" / "B" / "B_probs.csv",
         "S6F_SECONDS": 12.0}
    exec(compile(code, "blend", "exec"), g)
    return pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, LAB].values


def expect(pr, ps_sub, pf, s6f_failed, use_s6f):
    rap_fail = np.zeros(N, bool); rap_fail[5] = True
    our_fail = np.zeros(N, bool); our_fail[7] = True
    s6f_fail = np.zeros(N, bool)
    s6f_fail[list(s6f_failed)] = True
    if not use_s6f:
        s6f_fail[:] = True
    out = np.zeros((N, 12))
    for j in range(12):
        rr, ro = pct(pr[:, j]), pct(ps_sub[:, j])
        v = pf[:, j].copy()
        if use_s6f:
            v[s6f_fail] = v[~s6f_fail].mean()
        rs = pct(v)
        for i in range(N):
            if s6f_fail[i]:
                legs = [(rr[i], 0.7, not rap_fail[i]), (ro[i], 0.3, not our_fail[i])]
            else:
                legs = [(rr[i], 0.5, not rap_fail[i]), (ro[i], 0.25, not our_fail[i]), (rs[i], 0.25, True)]
            den = sum(w for _, w, ok in legs if ok)
            out[i, j] = sum(r * w for r, w, ok in legs if ok) / den if den > 0 else 0.5
    return out


for name, failed, ok, use in (("1_not_run", (), False, False), ("2_used", (), True, True),
                              ("3_one_failed", (11,), True, True), ("4_three_failed", (11, 12, 13), True, False)):
    d = root / name
    DATA, WORK, pred, pr, ps_sub, pf = setup(d, failed)
    got = run(new_code, DATA, WORK, pred, ok)
    receipt = json.loads((WORK / "blend_receipt.json").read_text())
    want = expect(pr, ps_sub, pf, failed, use)
    err = np.abs(got - want).max()
    msg = f"case {name}: max |diff| vs hand computation {err:.2e}"
    if not use:
        old = run(old_code, DATA, WORK, pred, ok)                     # unmodified A-strict cell on the same inputs
        msg += f" | vs unmodified A-strict cell {np.abs(got - old).max():.2e}"
        assert np.abs(got - old).max() < 1e-6
    assert err < 1e-6, msg
    print(msg, "| receipt s6fall:", receipt.get("s6fall"), "| weights:", receipt["weights"])
print("ALL BLEND CASES PASS")
