"""Unit test for final-max v3 merge and blend logic.
"""
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]

def main():
    root = Path("D:/rsna_work/tmp/test_final_max_v3")
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    pred = root / "pred"
    WORK = root / "work"
    DATA = root / "data"
    pred.mkdir(); WORK.mkdir(); DATA.mkdir()

    # Create dummy test.csv
    test_ids = [f"test_{i}" for i in range(10)]
    pd.DataFrame({"StudyInstanceUID": test_ids}).to_csv(DATA / "test.csv", index=False)
    pd.DataFrame(columns=["StudyInstanceUID"] + LAB).to_csv(DATA / "sample_submission.csv", index=False)

    # Create worker dirs
    (pred / "worker0").mkdir()
    (pred / "worker1").mkdir()

    for w, s_ids in [("worker0", test_ids[:5]), ("worker1", test_ids[5:])]:
        df_f = pd.DataFrame(np.random.rand(len(s_ids), 12), columns=LAB)
        df_f.insert(0, "StudyInstanceUID", s_ids)
        df_f.to_csv(pred / w / "x_E_prior_s42_fall.csv", index=False)

        df_v2 = pd.DataFrame(np.random.rand(len(s_ids), 12), columns=LAB)
        df_v2.insert(0, "StudyInstanceUID", s_ids)
        df_v2.to_csv(pred / w / "x_s6ft_v2_ema.csv", index=False)

        df_v1 = pd.DataFrame(np.random.rand(len(s_ids), 12), columns=LAB)
        df_v1.insert(0, "StudyInstanceUID", s_ids)
        df_v1.to_csv(pred / w / "x_s6ft_v1_ema.csv", index=False)

        (pred / w / "x_receipt.json").write_text(json.dumps({"gpu_s": 2.0, "render_s": 1.0, "failed": {}}))

    # Read notebook
    nb = json.load(open("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb", encoding="utf-8"))
    c10 = "".join(nb["cells"][10]["source"])
    c12 = "".join(nb["cells"][12]["source"])

    # Extract merge block from c10
    mark = "# [max v3] collect in-pass models"
    i = c10.index(mark)
    j = c10.index("\nif PROBE_CLASS:")
    merge_code = c10[i:j]

    # Execute Cell 10 merge logic
    g10 = {"DATA": DATA, "WORK": WORK, "pred": pred, "json": json, "shutil": shutil}
    exec(compile(merge_code, "cell10_merge", "exec"), g10)
    assert g10["S6F_OK"] is True
    assert g10["S6FT2_OK"] is True
    assert g10["S6FT1_OK"] is True
    print("CELL 10 MERGE TEST PASSED")

    # Set up globals for Cell 12
    raw = pd.DataFrame(np.random.rand(10, 12), columns=LAB, index=test_ids)
    rsub = raw.copy()
    rsub.insert(0, "StudyInstanceUID", test_ids)
    rsub.to_csv(WORK / "raptor_raw.csv", index=False)
    rsub.to_csv(WORK / "raptor_sub.csv", index=False)
    rsub.to_csv(WORK / "raptor224_sub.csv", index=False)
    rsub.to_csv(WORK / "ours_sub.csv", index=False)

    g12 = {
        "DATA": DATA, "WORK": WORK, "pred": pred, "labels": LAB,
        "cols": ["StudyInstanceUID"] + LAB, "test_ids": test_ids, "R224_OK": True,
        "W_RAPTOR": 0.5, "W_OURS": 0.5, "A224": 0.35, "RAPTOR_SECONDS": 5.0, "NB_T0": 0.0,
        "csv": __import__("csv"), "json": json, "pd": pd, "np": np, "time": __import__("time"),
        "display": lambda *a, **k: None,
        "S6F_PROBS": g10["S6F_PROBS"], "S6F_OK": g10["S6F_OK"], "S6FT2_OK": g10["S6FT2_OK"], "S6FT1_OK": g10["S6FT1_OK"]
    }
    # Create fake worker predictions.csv
    p_ours = raw.copy()
    p_ours.insert(0, "StudyInstanceUID", test_ids)
    p_ours.to_csv(pred / "worker0" / "predictions.csv", index=False)

    exec(compile(c12, "cell12", "exec"), g12)
    sub = pd.read_csv(WORK / "submission.csv")
    assert len(sub) == 10
    assert np.isfinite(sub[LAB].values).all()
    assert ((sub[LAB].values >= 0) & (sub[LAB].values <= 1)).all()
    print("CELL 12 BLEND TEST PASSED! Output shape:", sub.shape)
    shutil.rmtree(root)

if __name__ == "__main__":
    main()
