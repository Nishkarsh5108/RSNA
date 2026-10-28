"""Verify that inference preprocessing reproduces the training cache bit-exactly.

1. Local test studies (15 series): infer_lib.preprocess_series vs output/v2/preprocessed.
2. A random sample of TRAIN series: same comparison, plus the rebuilt study index (slots,
   side, step) vs data/imaging_v5/study_index_train_v2.csv for those studies.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_lib import build_index_row, preprocess_series  # noqa: E402
from study_index import SLOT_NAMES  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
res = []
ts = pd.read_csv(f"{R}/data/Raw_dicom/test_series.csv")
for r in ts.itertuples():
    a, _ = preprocess_series(f"{R}/data/Raw_dicom/test_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}")
    b = np.load(f"{R}/output/v2/preprocessed/{r.SeriesInstanceUID}.npy")
    res.append(("test", r.SeriesInstanceUID, a.shape == b.shape and np.array_equal(a, b), int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else -1))
idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train_v2.csv").set_index("StudyInstanceUID")
trs = pd.read_csv(f"{R}/data/Raw_dicom/train_series.csv")
studies = idx.sample(6, random_state=3).index
idx_mism = []
for st in studies:
    meta = trs[trs.StudyInstanceUID == st]
    geos = {}
    for r in meta.itertuples():
        a, g = preprocess_series(f"{R}/data/Raw_dicom/train_series/{st}/{r.SeriesInstanceUID}")
        b = np.load(f"{R}/output/v2/preprocessed/{r.SeriesInstanceUID}.npy")
        geos[r.SeriesInstanceUID] = g
        res.append(("train", r.SeriesInstanceUID, a.shape == b.shape and np.array_equal(a, b), int(np.abs(a.astype(int) - b.astype(int)).max()) if a.shape == b.shape else -1))
    row = build_index_row(st, meta, geos)
    for k in ["side"] + [f"{n}_uid" for n in SLOT_NAMES] + [f"{n}_step" for n in SLOT_NAMES]:
        v_new, v_ref = row[k], idx.loc[st, k]
        same = (v_new == v_ref) or (pd.isna(v_ref) and v_new is None)
        if not same:
            idx_mism.append((st[-8:], k, v_new, v_ref))
df = pd.DataFrame(res, columns=["split", "series", "bit_exact", "max_abs_diff"])
print(df.groupby("split").bit_exact.agg(["sum", "count"]))
print("max abs diff overall:", df.max_abs_diff.max())
print("index mismatches (inference rebuild vs training index):", idx_mism if idx_mism else "none")
