"""Time the CPU stages of the inference pipeline on N fold-0 training studies presented as a test set
(efficiency analysis). Stages: full header pass (current), fast index (file counts only), decode of the
selected series for v2 and hr preprocessing. Nothing is written to the repo; caches go to D:/rsna_work/tmp.

  python src/emergency/bench_preprocess.py [N] [WORKERS]
"""
import json
import os
import shutil
import sys
import time

import pandas as pd

R = "D:/Coding/kaggle/RSNA"
sys.path.insert(0, f"{R}/src/imaging")
from predict import build_test_index, preprocess_selected  # noqa: E402

W = "D:/rsna_work/tmp/bench_preprocess"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(W)
    idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train_v2.csv")
    studies = idx[idx.fold == 0].StudyInstanceUID.head(n).tolist()
    ts = pd.read_csv(f"{R}/data/Raw_dicom/train_series.csv")
    ts[ts.StudyInstanceUID.isin(studies)].to_csv(f"{W}/test_series.csv", index=False)
    root = f"{R}/data/Raw_dicom/train_series"
    out = {"n_studies": n, "workers": workers}
    t = time.time()
    tidx, fails, geos = build_test_index(f"{W}/test_series.csv", root, workers, log=lambda *a: None)
    out["header_pass_s"] = round(time.time() - t, 1)
    t = time.time()
    fidx, ffails, fgeos = build_test_index(f"{W}/test_series.csv", root, workers, log=lambda *a: None, fast=True)
    out["fast_index_pass_s"] = round(time.time() - t, 1)
    cols = [c for c in tidx.columns if c.endswith("_uid") or c.endswith("_step") or c in ("side",)]
    a, b = tidx.set_index("StudyInstanceUID")[cols], fidx.set_index("StudyInstanceUID").loc[tidx.StudyInstanceUID, cols]
    out["fast_vs_full_index_mismatch_cells"] = {c: int((a[c].astype(str) != b[c].astype(str)).sum()) for c in cols
                                                if (a[c].astype(str) != b[c].astype(str)).any()}
    t = time.time()
    nfiles = {s: len(os.listdir(f"{root}/{r.StudyInstanceUID}/{s}")) for r in ts[ts.StudyInstanceUID.isin(studies)].itertuples()
              for s in [r.SeriesInstanceUID]}
    out["file_count_pass_s"] = round(time.time() - t, 2)
    out["files_all_series"] = int(sum(nfiles.values()))
    for pp in ("v2", "hr"):
        t = time.time()
        _, f2 = preprocess_selected(tidx, geos, root, f"{W}/cache_{pp}", [0, 1, 2], workers, preproc=pp, log=lambda *a: None)
        out[f"decode_{pp}_s"] = round(time.time() - t, 1)
        out[f"decode_{pp}_failures"] = len([x for x in f2 if x["stage"] == "decode"])
    sel = [tidx.at[i, f"{s}_uid"] for i in range(len(tidx)) for s in ("sag_fs", "cor_fs", "axi_fs", "sag_nfs", "cor_nfs", "axi_nfs")]
    out["files_selected_series"] = int(sum(nfiles.get(s, 0) for s in sel if isinstance(s, str)))
    out["per_study_s"] = {k[:-2]: round(v / n, 3) for k, v in out.items() if k.endswith("_s") and isinstance(v, float)}
    print(json.dumps(out, indent=1))
    json.dump(out, open(f"{R}/output/emergency_research/bench_preprocess_{n}_{workers}w.json", "w"), indent=1)


if __name__ == "__main__":
    main()
