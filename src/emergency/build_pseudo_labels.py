"""Pseudo-label targets from out-of-fold (OOF) image-model predictions (experiment PL, emergency plan).

For every non-gold training study, the OOF prediction comes from the model that did NOT train on that
study's fold (fold k -> run validated on fold k). Target = alpha * report teacher + (1 - alpha) * OOF
(soft, per cell). Gold rows are copied unchanged from the teacher file (they are never trained on).

Usage:
  python src/emergency/build_pseudo_labels.py --alpha 0.5 --out data/labels_v3/pl_r1_a50.csv \
      --runs 0=E1p_cnxtT_s3_k8_attn12_posfix 1=O1_... 2=O2_... 3=O3_... 4=O4_...
Writes the label CSV (git-ignored data/labels_v3/) and a summary JSON next to it.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

R = "D:/Coding/kaggle/RSNA"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=f"{R}/data/labels_v3/training_labels_v3.csv")
    ap.add_argument("--index", default=f"{R}/data/imaging_v5/study_index_train_v2.csv")
    ap.add_argument("--runs-root", default=f"{R}/output/imaging_v5/runs")
    ap.add_argument("--runs", nargs="+", required=True, help="fold=run_id (one per fold 0..4); several runs per fold are averaged")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the report teacher")
    ap.add_argument("--teacher-only", default="", help="comma-separated findings kept at the teacher label (alpha = 1), "
                    "e.g. PL-g: 'ACL,MCL,Lateral Meniscus,PF OA,Synovitis'")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    idx = pd.read_csv(a.index)
    teacher = pd.read_csv(a.teacher).set_index("StudyInstanceUID")
    by_fold = {}
    for spec in a.runs:
        f, run = spec.split("=")
        by_fold.setdefault(int(f), []).append(run)
    oof_parts = []
    for f, runs in sorted(by_fold.items()):
        ps = []
        for run in runs:
            v = pd.read_csv(f"{a.runs_root}/{run}/val_preds_best.csv").set_index("StudyInstanceUID")[TARGETS]
            cfg = json.load(open(f"{a.runs_root}/{run}/config.json"))
            assert int(cfg["val_fold"]) == f and not cfg.get("train_all_folds"), f"{run} is not an OOF run for fold {f}"
            ps.append(v)
        p = sum(ps) / len(ps)
        want = set(idx[idx.fold == f].StudyInstanceUID)
        assert set(p.index) == want, f"fold {f}: OOF rows do not match the fold"
        oof_parts.append(p)
    oof = pd.concat(oof_parts)
    non_gold = idx[idx.fold >= 0].StudyInstanceUID
    missing = set(non_gold) - set(oof.index)
    assert not missing, f"{len(missing)} non-gold studies without OOF predictions (folds given: {sorted(by_fold)})"

    t = teacher.loc[oof.index, TARGETS]
    y = a.alpha * t + (1 - a.alpha) * oof
    keep = [c.strip() for c in a.teacher_only.split(",") if c.strip()]
    assert all(c in TARGETS for c in keep), keep
    for c in keep:
        y[c] = t[c]
    out = teacher.copy()
    out.loc[y.index, TARGETS] = y.values
    out.loc[y.index, "label_source"] = f"pl_alpha{a.alpha:g}"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    oof.reset_index().to_csv(a.out.replace(".csv", "_oof_only.csv"), index=False)
    tmp = a.out + ".tmp"                       # atomic: a training queue may be waiting for this file to appear
    out.reset_index().to_csv(tmp, index=False)
    os.replace(tmp, a.out)

    summ = {"alpha": a.alpha, "teacher_only": keep, "runs": {str(k): v for k, v in by_fold.items()}, "n_studies": int(len(y)), "per_class": {}}
    for c in TARGETS:
        tb = (t[c] >= 0.5).astype(int)
        summ["per_class"][c] = {
            "teacher_mean": round(float(t[c].mean()), 4), "oof_mean": round(float(oof[c].mean()), 4),
            "pearson_teacher_oof": round(float(np.corrcoef(t[c], oof[c])[0, 1]), 4),
            "oof_auc_vs_teacher_bin": round(float(roc_auc_score(tb, oof[c])), 4) if tb.nunique() == 2 else None,
            "cells_moved_gt_0.25": round(float((y[c] - t[c]).abs().gt(0.25).mean()), 4)}
    summ["oof_macro_auc_vs_teacher_bin"] = round(float(np.mean([v["oof_auc_vs_teacher_bin"] for v in summ["per_class"].values()
                                                                if v["oof_auc_vs_teacher_bin"] is not None])), 4)
    json.dump(summ, open(a.out.replace(".csv", "_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "per_class"}, indent=1))


if __name__ == "__main__":
    main()
