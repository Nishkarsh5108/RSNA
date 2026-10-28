"""Validate report-derived label sources against the 58 gold (image-read) studies.

All variants evaluated here were fixed BEFORE the first look at gold results for
labels_v3 (see data/labels_v3/EVALUATION_PROTOCOL.md):
  llm_s0        a-priori LLM score  pA + 0.5 pB + 0.25 pE
  llm_oof_cal   pooled logistic regression on [pA,pB,pC,pE] + per-target intercept,
                leave-one-STUDY-out on gold (out-of-fold only)
  rules         rules-v1.1 state -> {present .9, uncertain .5, historical .3, absent .05, not_mentioned .15}
  ens_*         weight-free means of sources (no gold-fitted weights)
Outputs: reports/eda/label_validation/*.csv|png (aggregate) and
         data/labels_v3/gold_error_analysis.csv (per-cell, contains report excerpts; git-ignored)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS, binary_report, macro_auc_ci, paired_macro_auc_diff, safe_auc  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/reports/eda/label_validation"
os.makedirs(OUT, exist_ok=True)
LET = ["A", "B", "C", "D", "E"]

train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
gold = train[train[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
G = gold.index
prof = pd.read_csv(f"{ROOT}/reports/eda/report_profile.csv").set_index("StudyInstanceUID")
lang = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})

llm = pd.read_csv(f"{ROOT}/data/labels_v3/llm_qwen3_4b_assert_v3.1_raw.csv").drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
rules = pd.read_csv(f"{ROOT}/data/labels_v3/rules_v1_labels.csv").set_index("StudyInstanceUID")


def llm_probs(df):
    return {t: df[[f"{t}__p{L}" for L in LET]].values for t in TARGETS}


P = llm_probs(llm.reindex(G))
s0 = pd.DataFrame({t: P[t][:, 0] + 0.5 * P[t][:, 1] + 0.25 * P[t][:, 4] for t in TARGETS}, index=G)

# out-of-fold calibrated LLM (leave-one-study-out, pooled across targets)
X = np.concatenate([np.column_stack([P[t][:, [0, 1, 2, 4]], np.tile(np.eye(12)[j], (len(G), 1))]) for j, t in enumerate(TARGETS)])
y = np.concatenate([gold[t].values for t in TARGETS])
sid = np.tile(np.arange(len(G)), 12)
oof = np.zeros(len(y))
for i in range(len(G)):
    tr, te = sid != i, sid == i
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr])
    oof[te] = clf.predict_proba(X[te])[:, 1]
cal = pd.DataFrame({t: oof[j * len(G):(j + 1) * len(G)] for j, t in enumerate(TARGETS)}, index=G)
clf_full = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)  # for deployment on unlabeled studies only
pd.Series(clf_full.coef_[0][:4], index=["pA", "pB", "pC", "pE"]).to_csv(f"{OUT}/llm_calibration_coefs.csv")

sources = {
    "llm_s0": s0,
    "llm_oof_cal": cal,
    "rules_v1.1": rules.reindex(G)[TARGETS].astype(float),
}
MD = f"{ROOT}/data/main_data/to_use"
sources["gpt56sol"] = pd.read_csv(f"{MD}/report_labels_gpt56sol.csv").set_index("StudyInstanceUID").reindex(G)[TARGETS]
sources["report_v2"] = pd.read_csv(f"{MD}/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(G)[TARGETS]
reg = pd.read_csv(f"{MD}/gt58_regenerated_qwen_predictions.csv").set_index("StudyInstanceUID")
sources["qwen7b_old"] = pd.DataFrame({t: reg[f"{t}_7b"] for t in TARGETS}).reindex(G).astype(float).clip(0, 1)
sources["qwen14b_old"] = pd.DataFrame({t: reg[f"{t}_14b"] for t in TARGETS}).reindex(G).astype(float).clip(0, 1)


def mean_of(names):
    arr = np.nanmean(np.stack([sources[n].values.astype(float) for n in names]), axis=0)
    return pd.DataFrame(arr, index=G, columns=TARGETS)


sources["ens_llm+rules"] = mean_of(["llm_s0", "rules_v1.1"])
sources["ens_llm+gpt56+v2"] = mean_of(["llm_s0", "gpt56sol", "report_v2"])
sources["ens_all4"] = mean_of(["llm_s0", "rules_v1.1", "gpt56sol", "report_v2"])
sources["ens_gpt56+v2 (public only)"] = mean_of(["gpt56sol", "report_v2"])

rows, macro = [], []
for name, S in sources.items():
    m = S.notna().all(axis=1)
    for t in TARGETS:
        r = binary_report(gold.loc[m, t].values, S.loc[m, t].values.astype(float), n_boot=2000, seed=11)
        r.update({"source": name, "target": t})
        rows.append(r)
    mp, lo, hi = macro_auc_ci(gold[m].values, S[m].values.astype(float), n_boot=2000, seed=11)
    pr = [x for x in rows if x["source"] == name]
    macro.append({"source": name, "n_gold": int(m.sum()), "macro_auc": mp, "lo": lo, "hi": hi,
                  "macro_f1@0.5": np.nanmean([x["f1"] for x in pr]), "macro_bal_acc@0.5": np.nanmean([x["balanced_acc"] for x in pr]),
                  "mean_brier": np.nanmean([x["brier"] for x in pr]), "mean_ece5": np.nanmean([x["ece5"] for x in pr])})
pc = pd.DataFrame(rows)
pc.to_csv(f"{OUT}/per_class_metrics.csv", index=False)
mac = pd.DataFrame(macro).sort_values("macro_auc", ascending=False)
mac.to_csv(f"{OUT}/macro_summary.csv", index=False)

# paired comparisons (same gold studies; report_v2 lacks one gold study -> restrict to common rows)
common = sources["report_v2"].notna().all(axis=1)
comp = []
for a in ["llm_s0", "llm_oof_cal", "rules_v1.1", "ens_llm+rules", "ens_llm+gpt56+v2", "ens_all4"]:
    for b in ["qwen7b_old", "gpt56sol", "report_v2", "ens_gpt56+v2 (public only)"]:
        d, lo, hi, p = paired_macro_auc_diff(gold[common].values, sources[a][common].values.astype(float),
                                             sources[b][common].values.astype(float), n_boot=2000, seed=13)
        comp.append({"a": a, "b": b, "n": int(common.sum()), "delta_macro_auc": d, "ci_lo": lo, "ci_hi": hi, "frac_boot_le_0": p})
pd.DataFrame(comp).to_csv(f"{OUT}/paired_comparisons.csv", index=False)

# per-language (cell level, threshold 0.5) for the main sources
lr = []
for name in ["llm_s0", "rules_v1.1", "gpt56sol", "report_v2", "qwen7b_old"]:
    S = sources[name]
    for L in sorted(lang.loc[G].unique()):
        u = [g for g in G if lang[g] == L and S.loc[g].notna().all()]
        if not u:
            continue
        yy, ss = gold.loc[u].values.ravel(), S.loc[u].values.astype(float).ravel()
        lr.append({"source": name, "lang": L, "n_studies": len(u), "cell_error_rate@0.5": float(((ss >= 0.5) != (yy == 1)).mean()),
                   "fn": int(((ss < 0.5) & (yy == 1)).sum()), "fp": int(((ss >= 0.5) & (yy == 0)).sum()),
                   "pooled_cell_auc": safe_auc(yy, ss)})
pd.DataFrame(lr).to_csv(f"{OUT}/per_language.csv", index=False)

# Question A vs B: what does the REPORT say about gold-positive / gold-negative cells?
arg = {t: np.array(LET)[P[t].argmax(1)] for t in TARGETS}
cells = []
for j, t in enumerate(TARGETS):
    for i, g in enumerate(G):
        rs = rules.loc[g, f"{t}__state"]
        la = arg[t][i]
        both_abs = (la == "D") and rs == "absent"
        both_pos = (la == "A") and rs == "present"
        llm_nm = la == "C"
        yv = int(gold.loc[g, t])
        s = float(s0.loc[g, t])
        if yv == 1 and s < 0.5:
            cat = "FN: report explicitly negative (LLM+rules agree) -> report/gold discordance" if both_abs else (
                "FN: finding not mentioned in report" if llm_nm and rs == "not_mentioned" else "FN: other (extraction or ambiguity)")
        elif yv == 0 and s >= 0.5:
            cat = "FP: report explicitly positive (LLM+rules agree) -> report/gold discordance" if both_pos else "FP: other (extraction, uncertainty or definition)"
        else:
            cat = "correct@0.5"
        cells.append({"StudyInstanceUID": g, "target": t, "lang": lang[g], "gold": yv, "llm_s0": round(s, 3),
                      "llm_argmax": la, "llm_pA": round(P[t][i, 0], 3), "llm_pB": round(P[t][i, 1], 3), "llm_pC": round(P[t][i, 2], 3),
                      "llm_pD": round(P[t][i, 3], 3), "llm_pE": round(P[t][i, 4], 3), "rules_state": rs,
                      "rules_evidence": rules.loc[g, f"{t}__evidence"], "gpt56sol": sources["gpt56sol"].loc[g, t],
                      "report_v2": sources["report_v2"].loc[g, t], "category": cat})
cells = pd.DataFrame(cells)
cells.to_csv(f"{ROOT}/data/labels_v3/gold_error_analysis.csv", index=False)
cat_tab = cells.groupby(["target", "category"]).size().unstack(fill_value=0)
cat_tab.to_csv(f"{OUT}/error_categories_by_target.csv")
avail = pd.DataFrame({t: pd.Series(arg[t]).value_counts() for t in TARGETS}).fillna(0).astype(int).T
avail.to_csv(f"{OUT}/llm_argmax_categories_on_gold.csv")
# P(gold=1 | report category)
cond = cells.groupby(["llm_argmax"]).gold.agg(["mean", "size"]).rename(columns={"mean": "P(gold=1)", "size": "n_cells"})
cond.to_csv(f"{OUT}/gold_rate_by_llm_category.csv")

# reliability diagram for llm_s0 vs llm_oof_cal (pooled cells)
fig, ax = plt.subplots(figsize=(5, 5))
for name in ["llm_s0", "llm_oof_cal", "gpt56sol"]:
    ss = sources[name].values.astype(float).ravel(); yy = gold.values.ravel()
    bins = np.linspace(0, 1, 6); idx = np.clip(np.digitize(ss, bins[1:-1]), 0, 4)
    xs = [ss[idx == b].mean() for b in range(5) if (idx == b).any()]; ys = [yy[idx == b].mean() for b in range(5) if (idx == b).any()]
    ax.plot(xs, ys, marker="o", label=name)
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("predicted"); ax.set_ylabel("gold positive rate"); ax.legend(); ax.set_title("Reliability on gold (pooled 696 cells)")
plt.tight_layout(); plt.savefig(f"{OUT}/reliability_gold.png", dpi=110); plt.close()

pd.set_option("display.width", 250)
print(mac.round(4).to_string(index=False))
print()
print(pd.DataFrame(comp).round(4).to_string(index=False))
print()
print(cat_tab.to_string())
print()
print(cond.round(3).to_string())
print("\ncalibration coefs (pA,pB,pC,pE):", np.round(clf_full.coef_[0][:4], 3))
