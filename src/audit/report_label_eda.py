"""Multilingual report + label-distribution EDA (aggregate outputs only).

Outputs to reports/eda/reports_labels/ (git-ignored).
"""
import collections
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "labeling"))
import rule_extractor as R  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/reports/eda/reports_labels"
os.makedirs(OUT, exist_ok=True)
T = R.TARGETS

train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
prof = pd.read_csv(f"{ROOT}/reports/eda/report_profile.csv").set_index("StudyInstanceUID")
prof["L"] = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})
gold = train[train[T].notna().all(axis=1)][T].astype(int)

# ---------------------------------------------------------------- language x structure
lang_tab = prof.groupby("L").agg(n=("n_chars", "size"), gold=("is_gold", "sum"),
                                 chars_median=("n_chars", "median"), chars_p90=("n_chars", lambda x: x.quantile(.9)),
                                 words_median=("n_words", "median"), truncated_1200=("truncated_at_1200", "mean"),
                                 frac_lost_1200_mean=("frac_lost_at_1200", "mean"),
                                 impression_hdr=("has_impression_hdr", "mean"), findings_hdr=("has_findings_hdr", "mean"),
                                 history_hdr=("has_history_hdr", "mean"), comparison_ref=("has_comparison_ref", "mean"),
                                 date_token=("has_date_token", "mean"), mojibake=("mojibake_suspect", "sum"),
                                 in_dup_group=("dup_group_size", lambda x: (x > 1).sum())).sort_values("n", ascending=False)
lang_tab.to_csv(f"{OUT}/language_structure.csv")

# ---------------------------------------------------------------- cue expressions per language
cues = {"negation": R.NEG_PRE, "negation_post": R.NEG_POST, "uncertainty": R.UNCERTAIN, "historical": R.HISTORICAL,
        "normal_statement": R.NORMAL}
cue_counts = {k: collections.defaultdict(collections.Counter) for k in cues}
per_report = []
for uid, rep in train.Report.items():
    t = R.fold(rep)
    L = prof.loc[uid, "L"]
    row = {"StudyInstanceUID": uid, "L": L}
    for k, pat in cues.items():
        ms = [m.group(0).strip() for m in pat.finditer(t)]
        row[f"n_{k}"] = len(ms)
        cue_counts[k][L].update(ms)
    per_report.append(row)
pr = pd.DataFrame(per_report)
pr.groupby("L")[[c for c in pr.columns if c.startswith("n_")]].mean().round(2).to_csv(f"{OUT}/cue_rate_per_report_by_language.csv")
top_rows = []
for k in cues:
    for L, cnt in cue_counts[k].items():
        for expr, n in cnt.most_common(8):
            top_rows.append({"cue_type": k, "language": L, "expression": expr, "count": n})
pd.DataFrame(top_rows).to_csv(f"{OUT}/top_cue_expressions_by_language.csv", index=False)

# ---------------------------------------------------------------- duplicates
dup = prof[prof.dup_group_size > 1]
dg = dup.groupby("report_hash").agg(size=("L", "size"), lang=("L", "first"), n_gold=("is_gold", "sum"),
                                    chars=("n_chars", "first")).sort_values("size", ascending=False)
dg.to_csv(f"{OUT}/duplicate_report_groups.csv")
gold_in_dup = dg[dg.n_gold > 0]

# ---------------------------------------------------------------- rule states by language (information availability)
rules = pd.read_csv(f"{ROOT}/data/labels_v3/rules_v1_labels.csv").set_index("StudyInstanceUID")
rules["L"] = prof["L"]
nm = pd.DataFrame({t: rules.groupby("L")[f"{t}__state"].apply(lambda s: (s == "not_mentioned").mean()) for t in T})
nm.round(3).to_csv(f"{OUT}/rules_not_mentioned_rate_by_language.csv")
posr = pd.DataFrame({t: rules.groupby("L")[f"{t}__state"].apply(lambda s: (s == "present").mean()) for t in T})
posr.round(3).to_csv(f"{OUT}/rules_present_rate_by_language.csv")

# ---------------------------------------------------------------- gold label structure
g_prev = gold.mean().rename("gold_prevalence")
phi = gold.corr()
phi.round(3).to_csv(f"{OUT}/gold_label_phi_correlation.csv")
n_pos_per_study = gold.sum(axis=1)
co = gold.T.dot(gold)
co.to_csv(f"{OUT}/gold_cooccurrence_counts.csv")
gpt = pd.read_csv(f"{ROOT}/data/main_data/to_use/report_labels_gpt56sol.csv").set_index("StudyInstanceUID")[T]
gpt_ng = (gpt.drop(index=gold.index) >= 0.5).astype(int)
gpt_ng.corr().round(3).to_csv(f"{OUT}/gpt56sol_unlabeled_phi_correlation.csv")
prev_cmp = pd.DataFrame({"gold_prevalence(n=58)": g_prev,
                         "gpt56sol_pos_rate_unlabeled": gpt_ng.mean(),
                         "rules_present_rate_unlabeled": (rules.drop(index=gold.index)[[f"{t}__state" for t in T]] == "present").mean().values,
                         "rules_not_mentioned_rate_unlabeled": (rules.drop(index=gold.index)[[f"{t}__state" for t in T]] == "not_mentioned").mean().values})
prev_cmp.round(3).to_csv(f"{OUT}/prevalence_gold_vs_reports.csv")
by_lang_prev = gpt_ng.join(prof["L"]).groupby("L").mean().round(3)
by_lang_prev.to_csv(f"{OUT}/gpt56sol_pos_rate_by_language.csv")

# figures
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
im = ax[0].imshow(phi.values, cmap="RdBu_r", vmin=-1, vmax=1); ax[0].set_xticks(range(12)); ax[0].set_xticklabels(T, rotation=90, fontsize=8)
ax[0].set_yticks(range(12)); ax[0].set_yticklabels(T, fontsize=8); ax[0].set_title("Gold (n=58) label correlation (phi)")
plt.colorbar(im, ax=ax[0], fraction=0.046)
ax[1].imshow(gpt_ng.corr().values, cmap="RdBu_r", vmin=-1, vmax=1); ax[1].set_xticks(range(12)); ax[1].set_xticklabels(T, rotation=90, fontsize=8)
ax[1].set_yticks(range(12)); ax[1].set_yticklabels(T, fontsize=8); ax[1].set_title("gpt56sol pseudo-labels (n=4,349) correlation")
plt.tight_layout(); plt.savefig(f"{OUT}/label_correlation_gold_vs_pseudo.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(10, 5))
prev_cmp[["gold_prevalence(n=58)", "gpt56sol_pos_rate_unlabeled", "rules_present_rate_unlabeled"]].plot.bar(ax=ax)
ax.set_title("Positive rate: gold (image read, enriched) vs report-derived labels"); plt.tight_layout()
plt.savefig(f"{OUT}/prevalence_gold_vs_reports.png", dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(10, 4))
for L, g in prof.groupby("L"):
    ax.hist(g.n_chars.clip(upper=4000), bins=40, histtype="step", label=L)
ax.axvline(1200, color="k", ls="--"); ax.legend(fontsize=7); ax.set_title("Report length (chars) by language; dashed = old 1200-char truncation")
plt.tight_layout(); plt.savefig(f"{OUT}/report_length_by_language.png", dpi=110); plt.close()

pd.set_option("display.width", 220)
print(lang_tab.round(3).to_string())
print("\nduplicate groups:", len(dg), "rows:", int(dg["size"].sum()), "largest:", dg["size"].head(5).tolist(), "groups containing gold:", len(gold_in_dup))
print("\n", prev_cmp.round(3).to_string())
print("\nnot-mentioned rate by language (rules):\n", nm.round(2).to_string())
print("\npositives per gold study:", n_pos_per_study.describe().round(2).to_dict())
