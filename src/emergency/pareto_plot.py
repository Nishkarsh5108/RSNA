"""Accuracy-runtime Pareto plot for inference configurations (efficiency analysis).

Accuracy: pseudo-val fold-0 macro AUC of saved prediction sets (single fold-0 models or their averages).
Runtime per study (local RTX 4050 + 4 CPU workers), all MEASURED on 2026-10-18 21:00 (idle GPU, 50 studies):
  preprocessing (fast index + decode) 1.006 s for v2 alone, 1.014 s for v2 + hr decoded once (bench_clean_*.log)
  network: 1.60 ms per 224 px window, 3.46 ms per 320 px window (bench_forward.json: ConvNeXt-T, 3 slots)
Writes reports/eda/emergency/pareto_accuracy_runtime.png and output/emergency_research/pareto_points.csv.
Usage: python src/emergency/pareto_plot.py  (edit POINTS below as prediction sets are added)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
ER = f"{R}/output/emergency_research"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# name -> (prediction set, list of (size, windows per study, n_bags) per model, preprocessing versions decoded)
POINTS = {
    "v2 1 model K8 (1 bag)": ("E1p_cnxtT_s3_k8_attn12_posfix__train", [(224, 24, 1)], ["v2"]),
    "v2 1 model K8 + shift": ("E1p_cnxtT_s3_k8_attn12_posfix__shift", [(224, 24, 2)], ["v2"]),
    "v2 1 model all windows": ("E1p_cnxtT_s3_k8_attn12_posfix__all", [(224, 84, 1)], ["v2"]),
    "v2 1 model K4": ("E1p_cnxtT_s3_k8_attn12_posfix__k4", [(224, 12, 1)], ["v2"]),
    "v2 1 model K6": ("E1p_cnxtT_s3_k8_attn12_posfix__k6", [(224, 18, 1)], ["v2"]),
    "v2 2 seeds K8 + shift": ("ENS_E1p+E1s2__shift", [(224, 24, 2), (224, 24, 2)], ["v2"]),
    "hr 1 model K12": ("E9_hr_cnxtT_s3_k6_attn12_c320_pf__train", [(320, 36, 1)], ["hr"]),
    "hr 1 model K12 + shift": ("E9_hr_cnxtT_s3_k6_attn12_c320_pf__shift", [(320, 36, 2)], ["hr"]),
    "hr 1 model all windows": ("E9_hr_cnxtT_s3_k6_attn12_c320_pf__all", [(320, 84, 1)], ["hr"]),
    "v2 + hr (1 + 1) shift": ("ENS_E1p+E9shift__mix", [(224, 24, 2), (320, 36, 2)], ["v2", "hr"]),
}
PRE = {("v2",): 1.006, ("hr",): 1.006, ("hr", "v2"): 1.014}       # s/study, measured (hr alone assumed = v2 alone)
MS_PER_WINDOW = {224: 0.00160, 320: 0.00346}                      # s, measured


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def main():
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    extra = json.load(open(f"{ER}/pareto_extra_points.json")) if os.path.exists(f"{ER}/pareto_extra_points.json") else {}
    rows = []
    for name, (pset, models, pps) in {**POINTS, **{k: tuple(v) for k, v in extra.items()}}.items():
        f = f"{ER}/preds/{pset}.npz"
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        Y = (lab.loc[z["uids"], TARGETS].values >= 0.5).astype(int)
        a = float(np.nanmean([auc(Y[:, j], z["p"][:, j]) for j in range(12)]))
        net = sum(MS_PER_WINDOW[s] * w * b for s, w, b in models)
        t = PRE[tuple(sorted(pps))] + net
        rows.append({"config": name, "pred_set": pset, "pseudoval_macro_auc": round(a, 4), "s_per_study_local": round(t, 3),
                     "min_per_1300_local": round(t * 1300 / 60, 1), "network_s": round(net, 3)})
    df = pd.DataFrame(rows).sort_values("s_per_study_local")
    df["pareto"] = df.pseudoval_macro_auc >= df.pseudoval_macro_auc.cummax()
    df.to_csv(f"{ER}/pareto_points.csv", index=False)
    os.makedirs(f"{R}/reports/eda/emergency", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, r in df.iterrows():
        ax.scatter(r.min_per_1300_local, r.pseudoval_macro_auc, c="tab:red" if r.pareto else "tab:gray", s=40)
        ax.annotate(r.config, (r.min_per_1300_local, r.pseudoval_macro_auc), fontsize=7, xytext=(4, 2), textcoords="offset points")
    p = df[df.pareto]
    ax.plot(p.min_per_1300_local, p.pseudoval_macro_auc, c="tab:red", lw=1)
    ax.set_xlabel("local runtime for 1,300 studies (min; measured per-study costs, fast index + one decode)")
    ax.set_ylabel("pseudo-val macro AUC (fold 0, n = 870)")
    ax.set_title("Accuracy-runtime trade-off (red = Pareto front); local, not Kaggle timings")
    fig.tight_layout()
    fig.savefig(f"{R}/reports/eda/emergency/pareto_accuracy_runtime.png", dpi=130)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
