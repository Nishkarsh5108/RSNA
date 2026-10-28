"""Rebuild output/emergency_research/experiments.csv from the run folders, saved prediction sets and the
hand-written notes (output/emergency_research/experiment_notes.json: hypothesis, decision, LB, gold, status).

Metrics are recomputed from saved predictions (never typed in): pseudo-val macro / confident-label / per-finding AUC
for every prediction set in output/emergency_research/preds/<run>__<variant>.npz.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

R = "D:/Coding/kaggle/RSNA"
ER = f"{R}/output/emergency_research"
RUNS = f"{R}/output/imaging_v5/runs"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def sha256(path, cache={}):
    if path not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        cache[path] = h.hexdigest()
    return cache[path]


def main():
    notes = json.load(open(f"{ER}/experiment_notes.json", encoding="utf-8"))
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    rows = []
    for exp_id, n in notes.items():
        run = n.get("run_id")
        row = {"id": exp_id, "hypothesis": n.get("hypothesis", ""), "config": n.get("config", ""),
               "model_family": n.get("model_family", ""), "label_source": n.get("label_source", ""),
               "seed": n.get("seed", ""), "validation_split": n.get("validation_split", ""), "run_id": run or "",
               "pred_variant": n.get("pred_variant", "")}
        if run and os.path.isdir(f"{RUNS}/{run}"):
            cfg = json.load(open(f"{RUNS}/{run}/config.json"))
            row["seed"] = cfg.get("seed", row["seed"])
            if os.path.exists(f"{RUNS}/{run}/summary.json"):
                sm = json.load(open(f"{RUNS}/{run}/summary.json"))
                row["train_min"] = round(sm["total_train_min"] + sm["total_val_min"], 1)
                row["peak_vram_gb"] = round(sm["peak_mem_gb"], 2)
                row["status"] = n.get("status", "completed")
            else:
                h = f"{RUNS}/{run}/history.csv"
                row["status"] = n.get("status", f"running ({len(pd.read_csv(h)) if os.path.exists(h) else 0} epochs done)")
            ck = f"{RUNS}/{run}/ckpt_best.pt"
            if os.path.exists(ck) and row["status"] == "completed":
                row["checkpoint"] = ck
                row["checkpoint_sha256"] = sha256(ck)
        else:
            row["status"] = n.get("status", "")
        pv = n.get("pred_variant")
        if pv and os.path.exists(f"{ER}/preds/{pv}.npz"):
            z = np.load(f"{ER}/preds/{pv}.npz", allow_pickle=True)
            Ys = lab.loc[z["uids"], TARGETS].values
            Y = (Ys >= 0.5).astype(int)
            per = [auc(Y[:, j], z["p"][:, j]) for j in range(12)]
            conf = []
            for j in range(12):
                k = (Ys[:, j] <= 0.2) | (Ys[:, j] >= 0.8)
                conf.append(auc(Y[k, j], z["p"][k, j]))
            row["pseudoval_macro_auc"] = round(float(np.nanmean(per)), 4)
            row["pseudoval_conf_auc"] = round(float(np.nanmean(conf)), 4)
            row.update({f"auc_{t}": round(v, 4) for t, v in zip(TARGETS, per)})
            row["windows_per_study"] = round(float(z["windows_per_study"]), 1)
        for k in ("delta_vs_ref", "gold_macro_auc", "public_lb", "infer_s_per_study_local", "decision"):
            if k in n:
                row[k] = n[k]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"{ER}/experiments.csv", index=False)
    print(df[[c for c in ["id", "status", "pseudoval_macro_auc", "pseudoval_conf_auc", "public_lb", "decision"] if c in df]].to_string())


if __name__ == "__main__":
    main()
