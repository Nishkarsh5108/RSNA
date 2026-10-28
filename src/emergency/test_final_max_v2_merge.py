"""CPU test of final-max v2's new code: the s6fall merge after R6's controller (cell 10) feeding v1's unchanged A3 blend.

Fake run: 40 studies over two workers; s6fall missing on study 11 (x failed) and study 7 (R6 failed there too, so neither
worker wrote it). Checks (1) the merged csv: test.csv order, the workers' values, NaN rows for 7 and 11, S6F_OK True;
(2) the A3 cell on it equals test_final_max_blend.py's hand computation with s6fall failed on {7, 11} (t2): 1 failure
among the 39 studies R6 scored is under v2's 3 % rule, so s6fall stays (v1's count, 2 of 40, would have dropped it);
(3) s6fall also missing on 12 and 13 (3 of R6's 39 > 3 %) -> dropped, A3 = R6 alone in the compact slot;
(4) no x csv at all -> S6F_OK False and A3 = R6 alone in the compact slot.
Usage: python -I src/emergency/test_final_max_v2_merge.py <final-max v2 ipynb> <empty dir>
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_final_max_blend as T   # noqa: E402

MARK = "# [max v2] s6fall from the same pass"


def merge_code(path):
    nb = json.load(open(path, encoding="utf-8"))
    s = next("".join(c["source"]) for c in nb["cells"] if "".join(c["source"]).startswith("deadline = T0 + DEADLINE_HOURS * 3600"))
    i = s.index(MARK)
    j = s.index("\nif PROBE_CLASS:")
    assert i < j, "merge block must come before the probe / receipt part"
    return s[i:j]


def main():
    nb, root = sys.argv[1], Path(sys.argv[2])
    code_m, code_b = merge_code(nb), T.cell(nb, T.HEAD)
    assert "TRANSFORM = 't2'" in code_b
    for name, x_present, miss, use in (("with_x", True, {7, 11}, True), ("x_dropped", True, {7, 11, 12, 13}, False),
                                       ("no_x", False, {7, 11}, False)):
        d = root / name
        DATA, WORK, pred, pr, p224, ps_sub, pf, r6_fail = T.setup(d, 7, (11,))
        (pred / "worker1").mkdir(exist_ok=True)
        if x_present:
            for w, part in (("worker0", range(0, T.N, 2)), ("worker1", range(1, T.N, 2))):
                rows = [[T.ids[i]] + list(pf[i]) for i in part if i not in miss]
                pd.DataFrame(rows, columns=["StudyInstanceUID"] + T.LAB).to_csv(pred / w / "x_E_prior_s42_fall.csv", index=False)
                (pred / w / "x_receipt.json").write_text(json.dumps({"gpu_s": 3.0, "render_s": 1.0, "failed": {}}))
        g = {"pred": pred, "DATA": DATA, "WORK": WORK, "json": json, "shutil": shutil}
        exec(compile(code_m, "merge", "exec"), g)
        if x_present:
            m = pd.read_csv(g["S6F_PROBS"], dtype={"StudyInstanceUID": str})
            assert m["StudyInstanceUID"].tolist() == T.ids, "merged rows are not in test.csv order"
            v = m[T.LAB].values
            bad = ~np.isfinite(v).all(axis=1)
            assert set(np.where(bad)[0]) == miss, np.where(bad)
            assert np.abs(v[~bad] - pf[~bad]).max() < 1e-9 and g["S6F_OK"] is True
        else:
            assert g["S6F_OK"] is False
        shutil.rmtree(WORK / "k1")                       # v2 has no a2b stage: the blend must read the merged file only
        gb = {"DATA": DATA, "WORK": WORK, "pred": pred, "RAPTOR_SECONDS": 1.0, "NB_T0": 0.0, "json": json, "time": __import__("time"),
              "display": lambda *a, **k: None, "R224_OK": True, "S6F_OK": g["S6F_OK"], "S6F_PROBS": g["S6F_PROBS"],
              "S6F_SECONDS": g["S6F_SECONDS"]}
        exec(compile(code_b, "blend", "exec"), gb)
        got = pd.read_csv(WORK / "submission.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[T.ids, T.LAB].values
        want = T.expect(pr, p224, ps_sub, pf, r6_fail, miss, use, True, "t2")
        err = np.abs(got - want).max()
        rec = json.loads((WORK / "blend_receipt.json").read_text())["final_max"]
        print(f"{name}: max |diff| vs hand computation {err:.2e} | receipt s6fall: {rec['s6fall']}")
        assert err < 1e-6
    print("FINAL-MAX V2 MERGE + BLEND PASS")


if __name__ == "__main__":
    main()
