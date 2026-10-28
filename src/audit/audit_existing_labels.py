"""Audit every pre-existing label source in data/main_data against the official
58 gold studies, and profile them on the 4,349 unlabeled studies.

Writes machine-readable tables to reports/eda/label_audit/:
  sources_summary.csv        coverage, leakage, failure signatures per source
  gold_metrics_per_class.csv per source x target metrics vs gold (bootstrap CIs)
  gold_macro.csv             macro AUC with joint bootstrap CI per source
  unlabeled_prevalence.csv   positive rate (>=0.5) and mean score per source/target
  pairwise_agreement.csv     kappa / spearman between sources on unlabeled studies
  gold_errors_by_language.csv thresholded cell errors by report language
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "labeling"))
from metrics import TARGETS, binary_report, macro_auc_ci  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
MD = f"{ROOT}/data/main_data"
OUT = f"{ROOT}/reports/eda/label_audit"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
gold = train[train[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
GOLD = gold.index
prof = pd.read_csv(f"{ROOT}/reports/eda/report_profile.csv").set_index("StudyInstanceUID")
lang = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})


def load_wide(path, prefix="", suffix=""):
    df = pd.read_csv(path).set_index("StudyInstanceUID")
    cols = [f"{prefix}{t}{suffix}" for t in TARGETS]
    sub = df[cols].copy()
    sub.columns = TARGETS
    return sub.astype(float)


sources = {}
desc = {}
regen = pd.read_csv(f"{MD}/to_use/gt58_regenerated_qwen_predictions.csv").set_index("StudyInstanceUID")
for v in ["7b", "14b"]:
    s = regen[[f"{t}_{v}" for t in TARGETS]].copy()
    s.columns = TARGETS
    sources[f"qwen{v}_regen_gt58"] = s.astype(float)
    desc[f"qwen{v}_regen_gt58"] = "Qwen2.5 regenerated predictions on the 58 gold reports only"
sources["qwen7b_file"] = load_wide(f"{MD}/to_use/train_folds_sota_qwen25.csv")
desc["qwen7b_file"] = "train_folds_sota_qwen25.csv (gold rows patched with regenerated preds)"
sources["qwen7b_bak"] = load_wide(f"{MD}/to_use/train_folds_sota_qwen25.csv.bak")
desc["qwen7b_bak"] = "original Qwen2.5-7B file before patch (combine_first with gold)"
sources["qwen14b_file"] = load_wide(f"{MD}/to_use/train_folds_qwen14b_awq_v2.csv")
desc["qwen14b_file"] = "train_folds_qwen14b_awq_v2.csv (gold rows patched)"
sources["qwen14b_bak"] = load_wide(f"{MD}/to_use/train_folds_qwen14b_awq_v2.csv.bak")
desc["qwen14b_bak"] = "original Qwen2.5-14B-AWQ file before patch"
sources["gpt56sol"] = load_wide(f"{MD}/to_use/report_labels_gpt56sol.csv")
desc["gpt56sol"] = "report_labels_gpt56sol.csv (public LLM label set, provenance not recorded in repo)"
sources["report_v2"] = load_wide(f"{MD}/to_use/report_labels_v2.csv")
desc["report_v2"] = "report_labels_v2.csv (verdict+confidence scorer, provenance not recorded)"
sources["public_pseudo"] = load_wide(f"{MD}/to_use/train_folds_with_pseudo.csv", prefix="pseudo_")
desc["public_pseudo"] = "train_folds_with_pseudo.csv pseudo_* columns (community labels)"
wl = pd.read_csv(f"{MD}/without_labels.csv").set_index("StudyInstanceUID")[TARGETS].astype(float)
sources["gemini_gt58"] = wl
desc["gemini_gt58"] = "without_labels.csv: hosted-LLM (Gemini) predictions on gold reports"
sources["consensus"] = load_wide(f"{MD}/to_use/train_folds_consensus.csv")
desc["consensus"] = "train_folds_consensus.csv primary blend"
ecf = pd.read_csv(f"{MD}/extracted_clinical_findings.csv").set_index("StudyInstanceUID")
sources["extracted_findings_cols"] = ecf[TARGETS].astype(float)
desc["extracted_findings_cols"] = "extracted_clinical_findings.csv label columns"

# ---------------------------------------------------------------- summary
summary, per_class, macro, prev = [], [], [], []
for name, df in sources.items():
    df = df[~df.index.duplicated()]
    g = df.reindex(GOLD)
    has_gold = g.notna().all(axis=1)
    identical = ((g - gold).abs().max(axis=1) < 1e-9) & has_gold
    nong = df.drop(index=GOLD, errors="ignore")
    allhalf = (nong == 0.5).all(axis=1)
    row = {
        "source": name, "description": desc[name], "rows": len(df),
        "rows_all12_nonnull": int(df.notna().all(axis=1).sum()),
        "gold_rows_present": int(has_gold.sum()),
        "gold_rows_identical_to_gold": int(identical.sum()),
        "leaked_on_gold": bool(has_gold.sum() and identical.mean() > 0.95),
        "unlabeled_rows": len(nong),
        "unlabeled_rows_all_0.5 (failure signature)": int(allhalf.sum()),
        "unlabeled_cells_exact_0.5": int((nong == 0.5).sum().sum()),
        "unlabeled_cells_nan": int(nong.isna().sum().sum()),
        "distinct_values_median_per_col": float(np.median([df[t].nunique() for t in TARGETS])),
        "values_outside_0_1": int(((df < 0) | (df > 1)).sum().sum()),
    }
    summary.append(row)
    for t in TARGETS:
        if len(nong):
            prev.append({"source": name, "target": t,
                         "pos_rate_ge_0.5": float((nong[t] >= 0.5).mean()),
                         "mean_score": float(nong[t].mean()),
                         "frac_exact_0.5": float((nong[t] == 0.5).mean())})
    if has_gold.sum() == 0 or row["leaked_on_gold"]:
        continue
    gg = g[has_gold]
    yy = gold.loc[gg.index]
    for t in TARGETS:
        r = binary_report(yy[t].values, gg[t].values, n_boot=2000, seed=1)
        r.update({"source": name, "target": t})
        per_class.append(r)
    mp, lo, hi = macro_auc_ci(yy.values, gg.values, n_boot=2000, seed=1)
    macro.append({"source": name, "n_gold": int(has_gold.sum()), "macro_auc": mp,
                  "macro_auc_lo": lo, "macro_auc_hi": hi,
                  "macro_f1": float(np.nanmean([r["f1"] for r in per_class if r["source"] == name])),
                  "macro_balanced_acc": float(np.nanmean([r["balanced_acc"] for r in per_class if r["source"] == name]))})

pd.DataFrame(summary).to_csv(f"{OUT}/sources_summary.csv", index=False)
pc = pd.DataFrame(per_class)
pc = pc[["source", "target"] + [c for c in pc.columns if c not in ("source", "target")]]
pc.to_csv(f"{OUT}/gold_metrics_per_class.csv", index=False)
pd.DataFrame(macro).sort_values("macro_auc", ascending=False).to_csv(f"{OUT}/gold_macro.csv", index=False)
pd.DataFrame(prev).to_csv(f"{OUT}/unlabeled_prevalence.csv", index=False)

# ---------------------------------------------------------------- pairwise agreement (unlabeled)
cmp_names = ["qwen7b_file", "qwen14b_file", "gpt56sol", "report_v2", "public_pseudo", "consensus"]
pair = []
unl = [u for u in train.index if u not in set(GOLD)]
for i, a in enumerate(cmp_names):
    for b in cmp_names[i + 1:]:
        A = sources[a].reindex(unl)
        B = sources[b].reindex(unl)
        for t in TARGETS:
            m = A[t].notna() & B[t].notna()
            pair.append({"a": a, "b": b, "target": t, "n": int(m.sum()),
                         "kappa_at_0.5": cohen_kappa_score(A[t][m] >= 0.5, B[t][m] >= 0.5),
                         "spearman": spearmanr(A[t][m], B[t][m]).correlation,
                         "agree_at_0.5": float(((A[t][m] >= 0.5) == (B[t][m] >= 0.5)).mean())})
pd.DataFrame(pair).to_csv(f"{OUT}/pairwise_agreement.csv", index=False)

# ---------------------------------------------------------------- gold errors by language
lang_rows = []
for name in ["qwen7b_regen_gt58", "qwen14b_regen_gt58", "gpt56sol", "report_v2", "gemini_gt58"]:
    g = sources[name].reindex(GOLD)
    for uid in GOLD:
        if g.loc[uid].isna().any():
            continue
        err = ((g.loc[uid] >= 0.5).astype(int) != gold.loc[uid].astype(int))
        lang_rows.append({"source": name, "StudyInstanceUID": uid, "lang": lang.get(uid),
                          "n_err": int(err.sum()), "n_fp": int(((g.loc[uid] >= 0.5) & (gold.loc[uid] == 0)).sum()),
                          "n_fn": int(((g.loc[uid] < 0.5) & (gold.loc[uid] == 1)).sum())})
lr = pd.DataFrame(lang_rows)
lr.to_csv(f"{OUT}/gold_errors_by_study.csv", index=False)
agg = lr.groupby(["source", "lang"]).agg(n_studies=("StudyInstanceUID", "nunique"),
                                         cell_error_rate=("n_err", lambda x: x.sum() / (12 * len(x))),
                                         fp=("n_fp", "sum"), fn=("n_fn", "sum")).reset_index()
agg.to_csv(f"{OUT}/gold_errors_by_language.csv", index=False)

pd.set_option("display.width", 250)
print(pd.DataFrame(summary).drop(columns=["description"]).to_string(index=False))
print()
print(pd.DataFrame(macro).sort_values("macro_auc", ascending=False).round(4).to_string(index=False))
