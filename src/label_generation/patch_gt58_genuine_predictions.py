# Replaces the 58 ground-truth rows' label columns in train_folds_qwen14b_awq_v2.csv and
# train_folds_sota_qwen25.csv -- currently copied ground truth (from combine_first in the
# original generation scripts) -- with the genuine regenerated predictions in
# gt58_regenerated_qwen_predictions.csv. Report text, fold, and the other ~4,349 rows are
# left untouched. Run locally (not on Kaggle) -- this only edits local CSV files.
#
# After this runs, consensus_label_aggregation.py's leakage auto-detection will pick up
# that qwen14b_awq and qwen7_5b are genuine on the next run, with no changes needed there.

import os
import pandas as pd

DATA_DIR = r"D:\Coding\kaggle\RSNA\data\main_data\to_use"

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# filename -> suffix used for that model's columns in gt58_regenerated_qwen_predictions.csv
PATCH_TARGETS = {
    "train_folds_qwen14b_awq_v2.csv": "14b",
    "train_folds_sota_qwen25.csv": "7b",
}

regenerated = pd.read_csv(os.path.join(DATA_DIR, "gt58_regenerated_qwen_predictions.csv"))
regenerated = regenerated.set_index("StudyInstanceUID")

gt_df = pd.read_csv(os.path.join(DATA_DIR, "available_labels.csv")).set_index("StudyInstanceUID")

for filename, suffix in PATCH_TARGETS.items():
    path = os.path.join(DATA_DIR, filename)
    backup_path = path + ".bak"

    df = pd.read_csv(path)

    if not os.path.exists(backup_path):
        df.to_csv(backup_path, index=False)
        print(f"[{filename}] backed up original to {os.path.basename(backup_path)}")
    else:
        print(f"[{filename}] backup already exists at {os.path.basename(backup_path)}, leaving it as-is")

    df = df.set_index("StudyInstanceUID")

    # Before/after leakage check -- confirms the patch actually replaced copied ground
    # truth with real predictions, mirroring the check in consensus_label_aggregation.py.
    common_before = gt_df.index.intersection(df.index)
    diff_before = (gt_df.loc[common_before, TARGET_COLS] - df.loc[common_before, TARGET_COLS]).abs().max(axis=1)
    identical_before = (diff_before < 1e-6).mean()

    patched_rows = 0
    for uid in regenerated.index:
        if uid not in df.index:
            print(f"[{filename}] WARNING: {uid} not found in this file, skipping")
            continue
        for col in TARGET_COLS:
            df.loc[uid, col] = regenerated.loc[uid, f"{col}_{suffix}"]
        patched_rows += 1

    diff_after = (gt_df.loc[common_before, TARGET_COLS] - df.loc[common_before, TARGET_COLS]).abs().max(axis=1)
    identical_after = (diff_after < 1e-6).mean()

    df = df.reset_index()
    df.to_csv(path, index=False)

    print(f"[{filename}] patched {patched_rows}/{len(regenerated)} rows")
    print(f"[{filename}] rows identical to ground truth: {identical_before*100:.1f}% before -> {identical_after*100:.1f}% after")
    print(f"[{filename}] saved in place\n")
