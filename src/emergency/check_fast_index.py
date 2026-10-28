"""Agreement of the fast inference index (one header per series) with the TRAINING index v2 on N fold-0 studies:
selected series per slot, 2.5D step and knee side. Usage: python src/emergency/check_fast_index.py [N]"""
import os
import sys

import pandas as pd

R = "D:/Coding/kaggle/RSNA"
sys.path.insert(0, f"{R}/src/imaging")
from predict import build_test_index  # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    W = "D:/rsna_work/tmp/check_fast_index"
    os.makedirs(W, exist_ok=True)
    idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train_v2.csv")
    studies = idx[idx.fold == 0].StudyInstanceUID.head(n).tolist()
    ts = pd.read_csv(f"{R}/data/Raw_dicom/train_series.csv")
    ts[ts.StudyInstanceUID.isin(studies)].to_csv(f"{W}/series.csv", index=False)
    fidx, fails, _ = build_test_index(f"{W}/series.csv", f"{R}/data/Raw_dicom/train_series", 6, log=print, fast=True)
    a = idx.set_index("StudyInstanceUID").loc[studies]
    b = fidx.set_index("StudyInstanceUID").loc[studies]
    out = {}
    for s in ("sag_fs", "cor_fs", "axi_fs", "sag_nfs", "cor_nfs", "axi_nfs"):
        out[f"{s}_uid"] = int((a[f"{s}_uid"].fillna("") != b[f"{s}_uid"].fillna("")).sum())
        both = a[f"{s}_uid"].notna() & b[f"{s}_uid"].notna()
        out[f"{s}_step"] = int((a.loc[both, f"{s}_step"].astype(int) != b.loc[both, f"{s}_step"].astype(int)).sum())
    out["side"] = int((a.side != b.side).sum())
    print("studies", n, "header failures", len(fails), "mismatches:", out)
    diff = a.index[a.side != b.side]
    if len(diff):
        print(pd.concat([a.loc[diff, ["side", "side_source"]], b.loc[diff, ["side", "side_source", "side_ambiguous"]]], axis=1))


if __name__ == "__main__":
    main()
