"""Add-on validation for item-order averaging (post-hoc; see CHANGELOG_after_gold.md #3)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS, binary_report, macro_auc_ci, paired_macro_auc_diff  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/reports/eda/label_validation"
LET = "ABCDE"
tr = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
gold = tr[tr[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
G = gold.index
fwd = pd.read_csv(f"{ROOT}/data/labels_v3/llm_qwen3_4b_assert_v3.1_raw.csv").set_index("StudyInstanceUID").reindex(G)
rev = pd.read_csv(f"{ROOT}/data/labels_v3/llm_qwen3_4b_assert_v3.1_reversed_raw.csv").set_index("StudyInstanceUID").reindex(G)
s0 = lambda d: pd.DataFrame({t: d[f"{t}__pA"] + 0.5 * d[f"{t}__pB"] + 0.25 * d[f"{t}__pE"] for t in TARGETS})
F, R = s0(fwd), s0(rev)
A = (F + R) / 2
MD = f"{ROOT}/data/main_data/to_use"
gpt = pd.read_csv(f"{MD}/report_labels_gpt56sol.csv").set_index("StudyInstanceUID").reindex(G)[TARGETS]
v2 = pd.read_csv(f"{MD}/report_labels_v2.csv").set_index("StudyInstanceUID").reindex(G)[TARGETS]
mean_ = lambda *xs: pd.DataFrame(np.nanmean(np.stack([x.values.astype(float) for x in xs]), axis=0), index=G, columns=TARGETS)
srcs = {"llm_canonical": F, "llm_reversed": R, "llm_order_avg": A,
        "ens(llm_canonical+gpt56+v2)": mean_(F, gpt, v2), "ens(llm_order_avg+gpt56+v2)": mean_(A, gpt, v2),
        "ens(gpt56+v2)": mean_(gpt, v2)}
rows, per = [], []
for k, S in srcs.items():
    mp, lo, hi = macro_auc_ci(gold.values, S.values, n_boot=2000, seed=11)
    rows.append({"source": k, "macro_auc": mp, "lo": lo, "hi": hi})
    for t in TARGETS:
        r = binary_report(gold[t].values, S[t].values, n_boot=0)
        per.append({"source": k, "target": t, "auc": r["auc"]})
pd.DataFrame(rows).to_csv(f"{OUT}/order_avg_macro.csv", index=False)
pd.DataFrame(per).pivot(index="target", columns="source", values="auc").to_csv(f"{OUT}/order_avg_per_class.csv")
common = v2.notna().all(axis=1)
comp = []
for a, b in [("llm_order_avg", "llm_canonical"), ("ens(llm_order_avg+gpt56+v2)", "ens(llm_canonical+gpt56+v2)"),
             ("ens(llm_order_avg+gpt56+v2)", "ens(gpt56+v2)"), ("llm_order_avg", "ens(gpt56+v2)")]:
    d = paired_macro_auc_diff(gold[common].values, srcs[a][common].values, srcs[b][common].values, n_boot=2000, seed=5)
    comp.append({"a": a, "b": b, "delta": d[0], "lo": d[1], "hi": d[2], "frac_le_0": d[3]})
pd.DataFrame(comp).to_csv(f"{OUT}/order_avg_paired.csv", index=False)
amf = np.stack([fwd[[f"{t}__p{L}" for L in LET]].values.argmax(1) for t in TARGETS])
amr = np.stack([rev[[f"{t}__p{L}" for L in LET]].values.argmax(1) for t in TARGETS])
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(comp).round(4).to_string(index=False))
print("gold argmax agreement canonical vs reversed:", round(float((amf == amr).mean()), 3))

# micro-averaged (pooled over all 12 x 58 cells) metrics for the main sources
from sklearn.metrics import f1_score  # noqa: E402
from metrics import safe_auc  # noqa: E402
qw = pd.read_csv(f"{MD}/gt58_regenerated_qwen_predictions.csv").set_index("StudyInstanceUID").reindex(G)
micro_src = {**srcs, "gpt56sol": gpt, "report_v2": v2, "qwen7b_old": pd.DataFrame({t: qw[f"{t}_7b"] for t in TARGETS}).astype(float).clip(0, 1)}
mrows = []
for k, S in micro_src.items():
    m = S.notna().all(axis=1)
    y, s = gold[m].values.ravel(), S[m].values.astype(float).ravel()
    mrows.append({"source": k, "n_cells": len(y), "micro_auc": safe_auc(y, s), "micro_f1@0.5": f1_score(y, s >= 0.5)})
pd.DataFrame(mrows).to_csv(f"{OUT}/micro_metrics.csv", index=False)
print(pd.DataFrame(mrows).round(4).to_string(index=False))
