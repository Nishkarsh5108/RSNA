"""Create the labels_v3 training-target variants for the downstream test.
All variants were specified before any downstream result for the new labels was seen."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/data/labels_v3/label_sets"
LET = ["A", "B", "C", "D", "E"]
train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
gold = train[train[TARGETS].notna().all(axis=1)][TARGETS].astype(float)
llm = pd.read_csv(f"{ROOT}/data/labels_v3/llm_qwen3_4b_assert_v3.1_raw.csv").drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
llm = llm[llm.status == "ok"].reindex(train.index)
rules = pd.read_csv(f"{ROOT}/data/labels_v3/rules_v1_labels.csv").set_index("StudyInstanceUID").reindex(train.index)
gpt = pd.read_csv(f"{ROOT}/data/labels_v3/label_sets/gpt56sol.csv").set_index("StudyInstanceUID").reindex(train.index)[TARGETS]
v2 = pd.read_csv(f"{ROOT}/data/labels_v3/label_sets/report_v2.csv").set_index("StudyInstanceUID").reindex(train.index)[TARGETS]

P = {t: llm[[f"{t}__p{L}" for L in LET]].values for t in TARGETS}
s0 = pd.DataFrame({t: P[t][:, 0] + 0.5 * P[t][:, 1] + 0.25 * P[t][:, 4] for t in TARGETS}, index=train.index)
arg = {t: np.where(np.isnan(P[t]).any(1), -1, np.nan_to_num(P[t]).argmax(1)) for t in TARGETS}
hard = pd.DataFrame({t: pd.Series(arg[t]).map({0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0, 4: 0.25, -1: np.nan}).values for t in TARGETS}, index=train.index)
maskC = s0.copy()
for t in TARGETS:
    maskC.loc[arg[t] == 2, t] = np.nan

# gold-calibrated mapping (pooled LR on pA,pB,pC,pE + target intercepts), fitted on ALL gold, deployed on non-gold.
G = gold.index
X = np.concatenate([np.column_stack([np.nan_to_num(P[t][train.index.get_indexer(G)][:, [0, 1, 2, 4]]), np.tile(np.eye(12)[j], (len(G), 1))]) for j, t in enumerate(TARGETS)])
y = np.concatenate([gold[t].values for t in TARGETS])
clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
cal = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
for j, t in enumerate(TARGETS):
    Xa = np.column_stack([P[t][:, [0, 1, 2, 4]], np.tile(np.eye(12)[j], (len(train), 1))])
    ok = ~np.isnan(Xa).any(1)
    cal.loc[ok, t] = clf.predict_proba(Xa[ok])[:, 1]

rs = rules[TARGETS].astype(float)
ens_all4 = pd.concat([s0, rs, gpt, v2]).groupby(level=0).mean().reindex(train.index)
ens_llm_pub = pd.concat([s0, gpt, v2]).groupby(level=0).mean().reindex(train.index)
silent = pd.DataFrame({t: (arg[t] == 2) & (rules[f"{t}__state"].values == "not_mentioned") for t in TARGETS}, index=train.index)
ens_all4_mask = ens_all4.mask(silent)
ens_llm_pub_mask = ens_llm_pub.mask(silent)

sets = {"llm_s0": s0, "llm_s0_maskC": maskC, "llm_hard": hard, "llm_goldcal_FLAGGED": cal,
        "ens_llm+gpt56+v2": ens_llm_pub, "ens_all4": ens_all4, "ens_all4_mask_silent": ens_all4_mask,
        "ens_llm+gpt56+v2_mask_silent": ens_llm_pub_mask}
paths = {}
for k, df in sets.items():
    p = f"{OUT}/{k}.csv"
    df.index.name = "StudyInstanceUID"
    df.reset_index().to_csv(p, index=False)
    paths[k] = p
paths["mean_gpt56_v2"] = f"{OUT}/mean_gpt56_v2.csv"
json.dump(paths, open(f"{OUT}/v3_sets.json", "w"), indent=1)
print("masked fraction llm_s0_maskC:", float(maskC.loc[~maskC.index.isin(G)].isna().mean().mean()),
      "| ens_all4_mask_silent:", float(ens_all4_mask.loc[~ens_all4_mask.index.isin(G)].isna().mean().mean()))
print(json.dumps(paths, indent=1))
