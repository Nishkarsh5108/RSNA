"""Unseal: score the stored gold-58 predictions ONCE, after the final model selection is locked.

  python evaluate_sealed.py --final RUN_A,RUN_B [--ref E1_cnxtT_s3_k8_attn12]

* Refuses to run if reports/eda/imaging/GOLD_UNSEALED.json exists (gold is scored once; --force
  overrides and is recorded). The selection passed with --final is written to that file BEFORE
  any gold metric is computed.
* For every run: gold macro AUC with a bootstrap CI and per-class AUCs; pseudo-val (fold 0)
  macro AUC with a CI for runs whose val fold was a holdout; paired deltas against --ref.
* For the locked final selection: gold AUC of the probability average (the submitted ensemble).
Writes reports/eda/imaging/experiment_results*.csv (git-ignored; derived from competition data).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import macro_auc_ci, paired_macro_auc_diff, safe_auc  # noqa: E402
from study_index import TARGETS  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
RUNS = f"{R}/output/imaging_v5/runs"
OUT = f"{R}/reports/eda/imaging"
LOCK = f"{OUT}/GOLD_UNSEALED.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True, help="comma-separated run ids of the locked final selection")
    ap.add_argument("--ref", default="E1_cnxtT_s3_k8_attn12")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    final = a.final.split(",")
    if os.path.exists(LOCK) and not a.force:
        sys.exit(f"gold already unsealed ({json.load(open(LOCK))}); refusing to score again (use --force, which is recorded)")
    for rid in final:
        if not os.path.exists(f"{RUNS}/{rid}/gold_preds_SEALED.npz"):
            sys.exit(f"final run {rid} has no sealed gold predictions")
    prev = json.load(open(LOCK)) if os.path.exists(LOCK) else None
    json.dump({"unsealed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "locked_final_selection": final, "reference": a.ref,
               "forced_rerun_of": prev}, open(LOCK, "w"), indent=1)

    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train.csv")
    va_uids = idx[idx.fold == 0].StudyInstanceUID.values
    Yva = (lab.loc[va_uids, TARGETS].values >= 0.5).astype(float)
    gold = lab[lab.label_source == "gold"]
    Yg = gold[TARGETS].values

    rows, per_rows, golds, vals = [], [], {}, {}
    for rd in sorted(glob.glob(f"{RUNS}/*/")):
        rid = os.path.basename(rd.rstrip("/\\"))
        if rid == "smoke" or not os.path.exists(f"{rd}/gold_preds_SEALED.npz"):
            continue
        z = np.load(f"{rd}/gold_preds_SEALED.npz", allow_pickle=True)
        G = pd.DataFrame(z["p"], index=z["uids"], columns=TARGETS).loc[gold.index].values
        golds[rid] = G
        mp, lo, hi = macro_auc_ci(Yg, G, n_boot=2000, seed=7)
        summ = json.load(open(f"{rd}/summary.json")) if os.path.exists(f"{rd}/summary.json") else {}
        holdout = summ.get("val_is_holdout", True)
        row = {"run": rid, "final": rid in final, "gold_macro_auc": mp, "gold_lo": lo, "gold_hi": hi,
               "pseudo_val_macro_auc": summ.get("best_val_macro_auc") if holdout else np.nan,
               "val_is_holdout": holdout, "best_epoch": summ.get("best_epoch"),
               "train_min": summ.get("total_train_min"), "peak_mem_gb": summ.get("peak_mem_gb")}
        vp = f"{rd}/val_preds_best.csv"
        if holdout and os.path.exists(vp):
            V = pd.read_csv(vp).set_index("StudyInstanceUID").loc[va_uids, TARGETS].values
            vals[rid] = V
            vm, vlo, vhi = macro_auc_ci(Yva, V, n_boot=1000, seed=7)
            row.update({"pseudo_val_lo": vlo, "pseudo_val_hi": vhi})
        rows.append(row)
        for j, t in enumerate(TARGETS):
            per_rows.append({"run": rid, "target": t, "gold_auc": safe_auc(Yg[:, j], G[:, j]),
                             "gold_n_pos": int(Yg[:, j].sum()),
                             "pseudo_val_auc": safe_auc(Yva[:, j], vals[rid][:, j]) if rid in vals else np.nan})
    ens = np.mean([golds[r] for r in final], 0)
    em, elo, ehi = macro_auc_ci(Yg, ens, n_boot=2000, seed=7)
    rows.append({"run": "FINAL_ENSEMBLE(" + "+".join(final) + ")", "final": True, "gold_macro_auc": em, "gold_lo": elo,
                 "gold_hi": ehi, "val_is_holdout": False})
    for j, t in enumerate(TARGETS):
        per_rows.append({"run": "FINAL_ENSEMBLE", "target": t, "gold_auc": safe_auc(Yg[:, j], ens[:, j]),
                         "gold_n_pos": int(Yg[:, j].sum()), "pseudo_val_auc": np.nan})
    res = pd.DataFrame(rows).sort_values(["final", "gold_macro_auc"], ascending=False)
    if a.ref in golds:
        d = []
        for rid in golds:
            if rid == a.ref:
                continue
            g = paired_macro_auc_diff(Yg, golds[rid], golds[a.ref], n_boot=2000, seed=9)
            v = paired_macro_auc_diff(Yva, vals[rid], vals[a.ref], n_boot=1000, seed=9) if (rid in vals and a.ref in vals) else (np.nan,) * 4
            d.append({"run": rid, "vs": a.ref, "gold_delta": g[0], "gold_lo": g[1], "gold_hi": g[2], "gold_p_le_0": g[3],
                      "pseudo_val_delta": v[0], "pseudo_val_lo": v[1], "pseudo_val_hi": v[2], "pseudo_val_p_le_0": v[3]})
        g = paired_macro_auc_diff(Yg, ens, golds[a.ref], n_boot=2000, seed=9)
        d.append({"run": "FINAL_ENSEMBLE", "vs": a.ref, "gold_delta": g[0], "gold_lo": g[1], "gold_hi": g[2], "gold_p_le_0": g[3]})
        pd.DataFrame(d).to_csv(f"{OUT}/experiment_paired_vs_{a.ref}.csv", index=False)
    res.to_csv(f"{OUT}/experiment_results.csv", index=False)
    pd.DataFrame(per_rows).to_csv(f"{OUT}/experiment_results_per_class.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(res.round(4).to_string(index=False))
    print(f"\ngold n = {len(gold)} studies; positives per class:", dict(zip(TARGETS, Yg.sum(0).astype(int))))


if __name__ == "__main__":
    main()
