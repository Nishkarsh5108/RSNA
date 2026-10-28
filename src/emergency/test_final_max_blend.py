"""CPU test of final-max's A3 blend cell (plan section 13) on fake stage outputs, against hand computations.

Cases (t1 and t2 each; Raptor fails on study 5, R6 on study 7, s6fall as listed; R224 present unless stated):
  1. s6fall not run        -> v27's own A3 cell on the same inputs (every finding except Fracture, whose weight is 0.65 now)
  2. s6fall used           -> raptor slot 0.65 R384 + 0.35 R224, compact 0.60 R6 + 0.40 s6fall, w_c per finding
  3. s6fall failed on 1/40 -> that study's compact slot is R6 alone
  4. s6fall failed on 3/40 (> 3 %) -> s6fall dropped: identical to case 1
  5. R224 not run, s6fall used
  6. study 9: R6 and s6fall both failed -> Raptor slot alone
Usage: python -I src/emergency/test_final_max_blend.py <final-max ipynb (t1)> <final-max ipynb (t2)> <v27 ipynb> <empty dir>
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
WC = {"Medial OA": 0.5, "Lateral OA": 0.2, "PF OA": 0.6, "Synovitis": 0.8, "Baker's": 0.35, "Fracture": 0.65}
HEAD = "# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5."
N = 40
ids = [f"1.2.3.{i}" for i in range(N)]


def cell(path, head):
    nb = json.load(open(path, encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith(head))


def pct(v):
    return pd.Series(np.asarray(v, float)).rank(method="average", pct=True).values


def pct_ok(v, ok):
    r = np.full(len(v), np.nan)
    r[ok] = pct(np.asarray(v, float)[ok])
    return r


def zl_ok(v, ok):
    r = np.full(len(v), np.nan)
    p = np.clip(np.asarray(v, float)[ok], 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    r[ok] = (z - z.mean()) / (z.std() + 1e-9)
    return r


def setup(d, seed, s6f_failed, r6_both=()):
    rng = np.random.default_rng(seed)
    DATA, WORK, pred = d / "data", d / "work", d / "pred"
    for p in (DATA, WORK / "k1" / "B", pred / "worker0"):
        p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ids, **{c: 0.5 for c in LAB}}).to_csv(DATA / "sample_submission.csv", index=False)
    pr, p224, ps, pf = (rng.uniform(0.01, 0.99, (N, 12)) for _ in range(4))
    pr[5] = 0.5                                              # Raptor failed on study 5 (all 0.5)
    r6_fail = {7} | set(r6_both)
    ps_sub = ps.copy()
    for i in r6_fail:                                        # R6 failed: column-mean fill in ours_sub, absent from predictions
        ps_sub[i] = np.delete(ps, sorted(r6_fail), axis=0).mean(axis=0)

    def frame(v):
        return pd.DataFrame({"StudyInstanceUID": ids, **{c: v[:, j] for j, c in enumerate(LAB)}})
    frame(pr).to_csv(WORK / "raptor_raw.csv", index=False)
    frame(pr).to_csv(WORK / "raptor_sub.csv", index=False)
    frame(p224).to_csv(WORK / "raptor224_sub.csv", index=False)
    frame(ps_sub).to_csv(WORK / "ours_sub.csv", index=False)
    with open(pred / "worker0" / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["StudyInstanceUID"] + LAB)
        for i in range(N):
            if i not in r6_fail:
                w.writerow([ids[i]] + [f"{x:.10f}" for x in ps[i]])
    pf_out = pf.copy()
    pf_out[sorted(set(s6f_failed) | set(r6_both))] = np.nan
    b = frame(pf_out)
    b.to_csv(WORK / "k1" / "B" / "B_probs.csv", index=False)
    return DATA, WORK, pred, pr, p224, ps_sub, pf, r6_fail


def run(code, DATA, WORK, pred, s6f_ok, r224_ok):
    g = {"DATA": DATA, "WORK": WORK, "pred": pred, "RAPTOR_SECONDS": 1.0, "NB_T0": time.time() - 5, "json": json, "time": time,
         "display": lambda *a, **k: None, "R224_OK": r224_ok, "S6F_OK": s6f_ok, "S6F_PROBS": WORK / "k1" / "B" / "B_probs.csv",
         "S6F_SECONDS": 12.0}
    exec(compile(code, "blend", "exec"), g)
    return pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids, LAB].values


def expect(pr, p224, ps_sub, pf, r6_fail, s6f_fail_rows, use_s6f, use_224, transform):
    rap_ok = np.ones(N, bool); rap_ok[5] = False
    r6_ok = np.ones(N, bool); r6_ok[sorted(r6_fail)] = False
    s6_ok = np.ones(N, bool); s6_ok[sorted(s6f_fail_rows)] = False
    if not use_s6f:
        s6_ok[:] = False

    def slot(f, j, neutral):
        rr = f(pr[:, j], rap_ok)
        if use_224:
            rr = 0.65 * rr + 0.35 * f(p224[:, j], rap_ok)
        ro = f(ps_sub[:, j], r6_ok)
        rs = f(pf[:, j], s6_ok) if s6_ok.any() else np.full(N, np.nan)
        wr = WC.get(LAB[j], 0.5)
        out = np.zeros(N)
        for i in range(N):
            comp = [(ro[i], 0.6, r6_ok[i]), (rs[i], 0.4, s6_ok[i])] if use_s6f else [(ro[i], 1.0, r6_ok[i])]
            den = sum(w for _, w, ok in comp if ok)
            rc = sum(v * w for v, w, ok in comp if ok) / den if den > 0 else None
            if rap_ok[i] and rc is not None:
                out[i] = wr * rr[i] + (1 - wr) * rc
            elif rap_ok[i]:
                out[i] = rr[i]
            elif rc is not None:
                out[i] = rc
            else:
                out[i] = neutral
        return out
    res = np.zeros((N, 12))
    for j in range(12):
        b = slot(pct_ok, j, 0.5)
        if transform == "t2":
            b = 0.5 * b + 0.5 * pct(slot(zl_ok, j, 0.0))
        res[:, j] = b
    return res


def main():
    nb_t1, nb_t2, v27, root = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    codes = {"t1": cell(nb_t1, HEAD), "t2": cell(nb_t2, HEAD)}
    old = cell(v27, HEAD)
    assert "TRANSFORM = 't1'" in codes["t1"] and "TRANSFORM = 't2'" in codes["t2"]
    cases = (("1_not_run", (), False, False, True, ()), ("2_used", (), True, True, True, ()),
             ("3_one_failed", (11,), True, True, True, ()), ("4_three_failed", (11, 12, 13), True, False, True, ()),
             ("5_no_r224", (), True, True, False, ()), ("6_r6_and_s6f_failed", (), True, True, True, (9,)))
    for tr in ("t1", "t2"):
        for k, (name, failed, ok, use, r224, both) in enumerate(cases):
            d = root / tr / name
            DATA, WORK, pred, pr, p224, ps_sub, pf, r6_fail = setup(d, 10 + k, failed, both)
            got = run(codes[tr], DATA, WORK, pred, ok, r224)
            rec = json.loads((WORK / "blend_receipt.json").read_text())["final_max"]
            want = expect(pr, p224, ps_sub, pf, r6_fail, set(failed) | set(both), use, r224, tr)
            err = np.abs(got - want).max()
            msg = f"{tr} case {name}: max |diff| vs hand computation {err:.2e} | receipt s6fall: {rec['s6fall']}"
            assert np.isfinite(got).all() and ((got >= 0) & (got <= 1)).all(), msg
            assert err < 1e-6, msg
            if not use and tr == "t1":
                ref = run(old, DATA, WORK, pred, ok, r224)              # v27's own cell on the same inputs
                keep = [j for j, c in enumerate(LAB) if c != "Fracture"]
                d_ = np.abs(got[:, keep] - ref[:, keep]).max()
                msg += f" | vs v27 cell (non-Fracture) {d_:.2e}"
                assert d_ < 1e-6, msg
            print(msg)
    print("ALL FINAL-MAX BLEND CASES PASS")


if __name__ == "__main__":
    main()
