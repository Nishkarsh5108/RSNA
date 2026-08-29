"""
Calibrate + ensemble pseudo-label sources for the RSNA knee abnormality dataset.

Inputs (paths as given, edit at the top if yours differ):
  - available_labels.csv        : the 58 human ground-truth studies
  - train_folds_qwen14b_awq_v2.csv : Qwen2.5-14B-Instruct-AWQ pseudo-labels
  - train_folds_sota_qwen25.csv    : Qwen2.5-7B-Instruct pseudo-labels
  - train_folds_with_pseudo.csv    : public/community pseudo-labels (pseudo_* columns)

IMPORTANT — read before trusting the calibration numbers:
All three pseudo-label files were generated with `combine_first()`, which keeps the
*original ground truth* for the 58 known studies and only fills in model predictions
for the other 4,349. That means none of these files contain the model's actual raw
predictions on the 58 GT rows — they contain the ground truth itself, copied in.
Fitting calibration against that would just be fitting a source to its own answer key
(perfect by construction, tells you nothing).

This script auto-detects that per source (comparing each source's values on the 58 GT
studies against available_labels.csv) and SKIPS calibration for any source found to be
leaked, printing a clear warning instead of silently producing a fake-looking fit. If a
source is ever added whose 58-row values are genuinely the model's own predictions (not
ground truth), the script will calibrate it automatically — no code change needed.

Ensembling, unlike calibration, is still fully valid here: the 4,349 unlabeled rows are
genuine, independent predictions from each source, so blending them is meaningful even
though calibration against the 58 GT rows currently isn't possible.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DATA_DIR = r"D:\Coding\kaggle\RSNA\data\main_data"

GT_PATH = os.path.join(DATA_DIR, "available_labels.csv")

SOURCES = {
    "qwen14b_awq": {
        "path": os.path.join(DATA_DIR, "train_folds_qwen14b_awq_v2.csv"),
        "col_prefix": "",  # target columns are bare, e.g. "ACL"
    },
    "qwen7_5b": {
        "path": os.path.join(DATA_DIR, "train_folds_sota_qwen25.csv"),
        "col_prefix": "",
    },
    "public_pseudo": {
        "path": os.path.join(DATA_DIR, "train_folds_with_pseudo.csv"),
        "col_prefix": "pseudo_",  # target columns are "pseudo_ACL", etc.
    },
}

# Ensemble blend weights, applied only on the 4,349 unlabeled rows (the 58 GT rows
# always use the true label regardless of weights). Set a source's weight to 0.0 to
# exclude it. public_pseudo defaults to a low weight: docs/label_generation_walkthrough.md
# documents community feedback that its underlying labels run ~60% accuracy on
# unlabeled reports, well below both Qwen sources' ~80% GT-benchmark scores.
ENSEMBLE_WEIGHTS = {
    "qwen14b_awq": 0.5,
    "qwen7_5b": 0.5,
    "public_pseudo": 0.0,
}

OUT_PATH = os.path.join(DATA_DIR, "train_folds_ensembled.csv")

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

LEAKAGE_TOLERANCE = 1e-6  # abs diff below this counts as "identical to ground truth"
N_CALIBRATION_FOLDS = 5


# -------------------------------------------------------------
# LOAD
# -------------------------------------------------------------
def load_source(name, spec):
    df = pd.read_csv(spec["path"])
    prefix = spec["col_prefix"]
    rename_map = {f"{prefix}{col}": col for col in TARGET_COLS}
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing expected columns: {missing}")
    df = df.rename(columns=rename_map)
    keep_cols = ["StudyInstanceUID"] + TARGET_COLS
    return df[keep_cols].copy()


print("=" * 100)
print(" LOADING GROUND TRUTH + PSEUDO-LABEL SOURCES")
print("=" * 100)

gt_df = pd.read_csv(GT_PATH)[["StudyInstanceUID"] + TARGET_COLS].copy()
for col in TARGET_COLS:
    gt_df[col] = gt_df[col].astype(float)
print(f"Ground truth: {len(gt_df)} studies")

source_dfs = {}
for name, spec in SOURCES.items():
    source_dfs[name] = load_source(name, spec)
    print(f"[{name}] {len(source_dfs[name])} rows loaded from {spec['path']}")


# -------------------------------------------------------------
# LEAKAGE DETECTION on the 58 GT rows
# -------------------------------------------------------------
print("\n" + "=" * 100)
print(" LEAKAGE CHECK: does each source's value on the 58 GT rows match the true label exactly?")
print("=" * 100)

leaked_sources = set()
for name, df in source_dfs.items():
    merged = pd.merge(gt_df, df, on="StudyInstanceUID", suffixes=("_true", "_src"))
    if len(merged) == 0:
        print(f"[{name}] WARNING: no overlap with GT StudyInstanceUIDs — cannot check.")
        continue

    diffs = []
    for col in TARGET_COLS:
        diffs.append(np.abs(merged[f"{col}_true"].astype(float) - merged[f"{col}_src"].astype(float)))
    max_diff = pd.concat(diffs, axis=1).max(axis=1)
    identical_rate = (max_diff < LEAKAGE_TOLERANCE).mean()

    if identical_rate > 0.98:
        leaked_sources.add(name)
        print(f"[{name}] {identical_rate*100:.1f}% of the 58 GT rows are identical to ground truth "
              f"-> LEAKED (this file's values on these rows are the true labels, not model predictions). "
              f"Calibration will be SKIPPED for this source.")
    else:
        print(f"[{name}] {identical_rate*100:.1f}% identical to ground truth -> looks like genuine "
              f"raw predictions. Calibration will be attempted.")

if len(leaked_sources) == len(source_dfs):
    print("\nAll sources are leaked on the 58 GT rows. This is expected given how the generation "
          "scripts use combine_first() to preserve ground truth. Calibration is skipped entirely "
          "below; only ensembling on the 4,349 unlabeled rows is performed.")
    print("To get genuine calibration, save the model's raw prediction dataframe for the 58 GT rows "
          "(before merging with true labels — e.g. `gt_pred_df` in label_generation_kaggle_v2.py) "
          "to its own CSV and point a new entry in SOURCES at it.")


# -------------------------------------------------------------
# CALIBRATION (only for non-leaked sources, cross-validated so it's honest)
# -------------------------------------------------------------
def cross_validated_platt_calibration(raw_probs, true_labels, n_splits=N_CALIBRATION_FOLDS):
    """
    Fit Platt scaling (1-feature logistic regression) per class, evaluated out-of-fold
    so the reported before/after metrics aren't just fitting-and-scoring on the same points.
    Returns out-of-fold calibrated probabilities (same length as input).
    """
    n = len(raw_probs)
    calibrated = np.full(n, np.nan)
    unique_classes = np.unique(true_labels)

    if len(unique_classes) < 2:
        # Degenerate: every label is the same class, calibration is meaningless.
        return raw_probs.copy()

    min_class_count = int(np.bincount(true_labels.astype(int)).min())
    if min_class_count < 2:
        # Too few positives (or negatives) to hold any out in a fold; skip calibration.
        return raw_probs.copy()

    skf = StratifiedKFold(n_splits=min(n_splits, min_class_count), shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(raw_probs.reshape(-1, 1), true_labels):
        y_train = true_labels[train_idx]
        if len(np.unique(y_train)) < 2:
            # This fold's training split has only one class present; fall back to raw prob.
            calibrated[val_idx] = raw_probs[val_idx]
            continue
        clf = LogisticRegression()
        clf.fit(raw_probs[train_idx].reshape(-1, 1), y_train)
        calibrated[val_idx] = clf.predict_proba(raw_probs[val_idx].reshape(-1, 1))[:, 1]

    return calibrated


calibrated_predictions = {}  # name -> {col: DataFrame column of calibrated probs, aligned to gt_df}

if leaked_sources != set(source_dfs.keys()):
    print("\n" + "=" * 100)
    print(" CROSS-VALIDATED CALIBRATION (non-leaked sources only)")
    print("=" * 100)

    for name, df in source_dfs.items():
        if name in leaked_sources:
            continue
        merged = pd.merge(gt_df, df, on="StudyInstanceUID", suffixes=("_true", "_src"))
        print(f"\n[{name}]")
        print(f"{'PATHOLOGY':<20} | {'AUC raw':<10} | {'AUC calibrated':<15} | {'Brier raw':<10} | {'Brier calibrated':<10}")
        for col in TARGET_COLS:
            y_true = merged[f"{col}_true"].astype(float).values.astype(int)
            y_raw = merged[f"{col}_src"].astype(float).values
            y_cal = cross_validated_platt_calibration(y_raw, y_true)

            if len(np.unique(y_true)) > 1:
                auc_raw = roc_auc_score(y_true, y_raw)
                auc_cal = roc_auc_score(y_true, y_cal)
            else:
                auc_raw = auc_cal = np.nan
            brier_raw = brier_score_loss(y_true, y_raw)
            brier_cal = brier_score_loss(y_true, y_cal)

            print(f"{col:<20} | {auc_raw:<10.4f} | {auc_cal:<15.4f} | {brier_raw:<10.4f} | {brier_cal:<10.4f}")


# -------------------------------------------------------------
# ENSEMBLE on the full dataset (58 GT rows keep true labels; 4,349 unlabeled rows get
# a weighted blend of the available sources' raw predictions)
# -------------------------------------------------------------
print("\n" + "=" * 100)
print(" ENSEMBLING")
print("=" * 100)
print(f"Blend weights: {ENSEMBLE_WEIGHTS}")

active_sources = {name: w for name, w in ENSEMBLE_WEIGHTS.items() if w > 0 and name in source_dfs}
if not active_sources:
    raise ValueError("No sources have a positive weight in ENSEMBLE_WEIGHTS — nothing to ensemble.")
weight_sum = sum(active_sources.values())

# Start from the ground-truth studies as the base (keeps original labels for the 58 known rows).
base_df = pd.read_csv(SOURCES["qwen14b_awq"]["path"])[["StudyInstanceUID", "Report"] + TARGET_COLS].copy()
base_df = base_df.set_index("StudyInstanceUID")

gt_mask = base_df.index.isin(gt_df["StudyInstanceUID"])
gt_df_indexed = gt_df.set_index("StudyInstanceUID")

blended = pd.DataFrame(index=base_df.index, columns=TARGET_COLS, dtype=float)
for name, weight in active_sources.items():
    src = source_dfs[name].set_index("StudyInstanceUID").reindex(base_df.index)
    for col in TARGET_COLS:
        contribution = src[col].astype(float) * (weight / weight_sum)
        blended[col] = blended[col].fillna(0.0).add(contribution, fill_value=0.0)

# Overwrite the 58 known studies with true ground truth (never blend those).
for col in TARGET_COLS:
    blended.loc[gt_mask, col] = gt_df_indexed.loc[base_df.index[gt_mask], col].values

final_df = base_df.copy()
for col in TARGET_COLS:
    final_df[col] = blended[col]
final_df = final_df.reset_index()

print(f"Final blended dataset: {len(final_df)} rows "
      f"({gt_mask.sum()} ground-truth passthrough, {(~gt_mask).sum()} ensembled).")

# -------------------------------------------------------------
# RE-STRATIFY INTO 5 FOLDS (recomputed fresh since blended targets differ from any
# single source's targets, so an inherited fold column wouldn't necessarily match)
# -------------------------------------------------------------
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
strat_target = (final_df[TARGET_COLS] >= 0.5).sum(axis=1)

final_df["fold"] = -1
for fold, (t_idx, v_idx) in enumerate(sgkf.split(final_df, strat_target, groups=final_df["StudyInstanceUID"])):
    final_df.loc[v_idx, "fold"] = fold

final_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved ensembled dataset to: {OUT_PATH}")
