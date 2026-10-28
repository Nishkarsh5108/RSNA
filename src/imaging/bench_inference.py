"""Measure the end-to-end inference pipeline (header index -> decode -> predict -> submission) on N
training studies presented as a fake test set, to estimate hidden-test runtime from measurements.
Usage: python bench_inference.py MODEL_DIR [N_STUDIES] [WORKERS] [--tta] [--fast]
Writes D:/rsna_work/tmp/bench_inference/{submission.csv, inference_report.json} and prints timings.
The fake test set is built from fold-0 (pseudo-val) studies only; nothing is written to the repo.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import predict  # noqa: E402
from study_index import TARGETS  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
W = "D:/rsna_work/tmp/bench_inference"


def main():
    model_dir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    tta = "--tta" in sys.argv
    fast = "--fast" in sys.argv
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(W)
    idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train_v2.csv")
    studies = idx[idx.fold == 0].StudyInstanceUID.head(n).tolist()
    ts = pd.read_csv(f"{R}/data/Raw_dicom/train_series.csv")
    ts[ts.StudyInstanceUID.isin(studies)].to_csv(f"{W}/test_series.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": studies}).to_csv(f"{W}/test.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": studies, **{t: 0.5 for t in TARGETS}}).to_csv(f"{W}/sample_submission.csv", index=False)
    cfg = dict(test_csv=f"{W}/test.csv", test_series_csv=f"{W}/test_series.csv", test_dicom_root=f"{R}/data/Raw_dicom/train_series",
               sample_submission=f"{W}/sample_submission.csv", model_dir=model_dir, cache_dir=f"{W}/cache",
               out_path=f"{W}/submission.csv", workers=workers, batch_size=4, device="cuda", loader_workers=0, tta_shift=tta,
               fast_index=fast)
    t0 = time.time()
    sub, rep = predict.run(cfg)
    total = time.time() - t0
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID").loc[sub.StudyInstanceUID, TARGETS]
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score((lab[t] >= 0.5).astype(int), sub[t]) for t in TARGETS if (lab[t] >= 0.5).nunique() == 2]
    out = {"n_studies": n, "workers": workers, "tta_shift": tta, "fast_index": fast, "n_models": rep["n_models"],
           "seconds_total": round(total, 1), "seconds_preprocess": rep["seconds_preprocess"], "seconds_predict": rep["seconds_predict"],
           "per_study_s": round(total / n, 2), "extrapolated_1300_min": round(total / n * 1300 / 60, 1),
           "sanity_pseudo_auc_on_these_studies": round(float(np.mean(aucs)), 3)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
