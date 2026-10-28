"""Compare finished runs on the PSEUDO-labelled validation fold only (gold-58 stays sealed).

For every run with summary.json + val_preds_best.csv:
  * macro AUC (teacher binarised at 0.5) with a study-bootstrap 95% CI, best/last epoch, cost;
  * paired bootstrap delta vs a reference run (same studies, same resamples);
  * per-class AUCs;
  * between-run prediction correlation and the pseudo-val AUC of 2-model probability averages
    (evidence for or against ensembling).
Writes reports/eda/imaging/pseudo_val_*.csv (git-ignored; derived from competition data).
Usage: python analyze_pseudo_val.py [REFERENCE_RUN]   (default E1_cnxtT_s3_k8_attn12)
"""
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import macro_auc_ci, paired_macro_auc_diff, safe_auc  # noqa: E402
from study_index import TARGETS  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
RUNS = f"{R}/output/imaging_v5/runs"
OUT = f"{R}/reports/eda/imaging"


def subgroup_table(Y, P, uids, idx):
    """Macro AUC per subgroup of the val fold (classes with one label value in a subgroup are undefined
    and left out; n_defined says how many of the 12 entered the mean)."""
    v = idx.set_index("StudyInstanceUID").loc[uids]
    groups = {"all": np.ones(len(v), bool)}
    for lang in v.lang.value_counts().index:
        groups[f"lang={lang}"] = (v.lang == lang).values
    for side in ("L", "R"):
        groups[f"side={side}"] = (v.side == side).values
    for src in ("tag", "geometry"):
        groups[f"side_source={src}"] = (v.side_source == src).values
    groups["side_conflict"] = v.side_conflict.astype(bool).values
    groups["sag_or_cor_FS_missing (fallback)"] = (v.sag_fs_uid.isna() | v.cor_fs_uid.isna()).values
    groups["all_3_FS_present"] = (v.sag_fs_uid.notna() & v.cor_fs_uid.notna() & v.axi_fs_uid.notna()).values
    rows = []
    for g, m in groups.items():
        r = {"subgroup": g, "n": int(m.sum())}
        for k, p in P.items():
            per = [safe_auc(Y[m, j], p[m, j]) for j in range(len(TARGETS))]
            r[k] = float(np.nanmean(per)) if not all(np.isnan(per)) else np.nan
            r[f"n_defined_{k}"] = int(np.sum(~np.isnan(per)))
        rows.append(r)
    return pd.DataFrame(rows)


def plot_distributions(Y, p, run, path):
    """Per-class histograms of predicted probability for teacher-positive vs teacher-negative studies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(16, 9))
    bins = np.linspace(0, 1, 26)
    for j, (t, ax) in enumerate(zip(TARGETS, axs.flat)):
        ax.hist(p[Y[:, j] == 0, j], bins=bins, alpha=0.6, density=True, label=f"teacher neg (n={int((Y[:, j] == 0).sum())})")
        ax.hist(p[Y[:, j] == 1, j], bins=bins, alpha=0.6, density=True, label=f"teacher pos (n={int((Y[:, j] == 1).sum())})")
        ax.set_title(f"{t}: AUC {safe_auc(Y[:, j], p[:, j]):.3f}", fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle(f"{run}: pseudo-val predicted probability by teacher label (fold 0, n={len(Y)})")
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


def main(ref="E1_cnxtT_s3_k8_attn12", n_boot=1000):
    os.makedirs(OUT, exist_ok=True)
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    notes = {}
    if os.path.exists(f"{R}/output/imaging_v5/queue.json"):
        notes = {q["run_id"]: q.get("note", "") for q in json.load(open(f"{R}/output/imaging_v5/queue.json"))}
    preds, rows = {}, []
    for rd in sorted(glob.glob(f"{RUNS}/*/")):
        rid = os.path.basename(rd.rstrip("/\\"))
        if rid == "smoke" or not (os.path.exists(f"{rd}/summary.json") and os.path.exists(f"{rd}/val_preds_best.csv")):
            continue
        summ = json.load(open(f"{rd}/summary.json"))
        if not summ.get("val_is_holdout", True):
            continue                                   # all-fold final models: fold 0 is in training
        preds[rid] = pd.read_csv(f"{rd}/val_preds_best.csv").set_index("StudyInstanceUID")[TARGETS]
        cfg = json.load(open(f"{rd}/config.json")) if os.path.exists(f"{rd}/config.json") else {}
        rows.append({"run": rid, "best_epoch": summ.get("best_epoch"), "last_epoch_auc": summ.get("last_val_macro_auc"),
                     "train_min": summ.get("total_train_min"), "val_min": summ.get("total_val_min"),
                     "peak_mem_gb": summ.get("peak_mem_gb"), "n_train": summ.get("n_train"),
                     "backbone": cfg.get("backbone", "frozen effnet-b0 features" if rid.startswith("E0") else None),
                     "slots": cfg.get("slots"), "agg": cfg.get("agg"), "preproc": cfg.get("preproc", "v2"),
                     "note": notes.get(rid, "")})
    uids = sorted(set.intersection(*[set(p.index) for p in preds.values()]))
    Y = (lab.loc[uids, TARGETS].values >= 0.5).astype(float)
    P = {k: v.loc[uids].values for k, v in preds.items()}
    Ysoft = lab.loc[uids, TARGETS].values
    conf = ~((Ysoft > 0.2) & (Ysoft < 0.8))              # teacher confident: |2y-1| >= 0.6 (85% of cells)
    for r in rows:
        m, lo, hi = macro_auc_ci(Y, P[r["run"]], n_boot=n_boot, seed=7)
        r.update({"pseudo_val_auc": m, "ci_lo": lo, "ci_hi": hi,
                  "conf_label_auc": float(np.nanmean([safe_auc(Y[conf[:, j], j], P[r["run"]][conf[:, j], j])
                                                      for j in range(len(TARGETS))]))})
        if ref in P and r["run"] != ref:
            d, dlo, dhi, p_le0 = paired_macro_auc_diff(Y, P[r["run"]], P[ref], n_boot=n_boot, seed=9)
            r.update({f"delta_vs_ref": d, "delta_lo": dlo, "delta_hi": dhi, "p_delta_le_0": p_le0})
    res = pd.DataFrame(rows).sort_values("pseudo_val_auc", ascending=False)
    res.insert(1, "ref", ref)
    res.to_csv(f"{OUT}/pseudo_val_results.csv", index=False)
    per = pd.DataFrame({k: [safe_auc(Y[:, j], v[:, j]) for j in range(len(TARGETS))] for k, v in P.items()}, index=TARGETS)
    per.to_csv(f"{OUT}/pseudo_val_per_class.csv")
    keys = [k for k in res.run if not k.startswith("E0")]
    corr, ens = pd.DataFrame(index=keys, columns=keys, dtype=float), []
    for a, b in itertools.product(keys, keys):
        corr.loc[a, b] = np.mean([np.corrcoef(P[a][:, j], P[b][:, j])[0, 1] for j in range(len(TARGETS))])
    for a, b in itertools.combinations(keys, 2):
        e = (P[a] + P[b]) / 2
        ens.append({"a": a, "b": b, "corr": corr.loc[a, b],
                    "auc_a": res.set_index("run").pseudo_val_auc[a], "auc_b": res.set_index("run").pseudo_val_auc[b],
                    "auc_mean_ab": np.nanmean([safe_auc(Y[:, j], e[:, j]) for j in range(len(TARGETS))])})
    corr.round(3).to_csv(f"{OUT}/pseudo_val_pred_correlation.csv")
    idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train.csv")
    sub_runs = {k: P[k] for k in dict.fromkeys([ref, *res.run[:3]]) if k in P}
    sg = subgroup_table(Y, sub_runs, uids, idx)
    sg.to_csv(f"{OUT}/pseudo_val_subgroups.csv", index=False)
    for k in sub_runs:
        plot_distributions(Y, P[k], k, f"{OUT}/pred_distributions_{k}.png")
    ens = pd.DataFrame(ens)
    if len(ens):
        ens["gain_over_best"] = ens.auc_mean_ab - ens[["auc_a", "auc_b"]].max(1)
        ens.sort_values("auc_mean_ab", ascending=False).to_csv(f"{OUT}/pseudo_val_pair_ensembles.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    cols = ["run", "pseudo_val_auc", "ci_lo", "ci_hi", "delta_vs_ref", "delta_lo", "delta_hi", "p_delta_le_0",
            "conf_label_auc", "best_epoch", "last_epoch_auc", "train_min", "peak_mem_gb"]
    print(f"pseudo-val fold 0: n={len(uids)} studies (teacher >= 0.5); reference {ref}")
    print(res[[c for c in cols if c in res]].round(4).to_string(index=False))
    print("\nper-class pseudo-val AUC")
    print(per[res.run].round(3).to_string())
    print("\nsubgroup macro AUC (pseudo-val)")
    print(sg[["subgroup", "n", *sub_runs]].round(3).to_string(index=False))
    if len(ens):
        print("\nbest 2-model averages")
        print(ens.sort_values("auc_mean_ab", ascending=False).head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main(*(sys.argv[1:2]))
