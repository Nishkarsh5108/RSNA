"""Run the locked rule-based extractor over all reports; write scores, states and evidence."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import rule_extractor as R  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
df = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv")
rows = []
for uid, rep in zip(df.StudyInstanceUID, df.Report):
    res = R.extract(rep)
    row = {"StudyInstanceUID": uid, "ruleset_version": R.RULESET_VERSION}
    for t in R.TARGETS:
        st, ev = res[t]
        row[t] = R.STATE_SCORE[st]
        row[f"{t}__state"] = st
        row[f"{t}__evidence"] = ev
    rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(f"{ROOT}/data/labels_v3/rules_v1_labels.csv", index=False)
st = pd.DataFrame({t: out[f"{t}__state"].value_counts() for t in R.TARGETS}).fillna(0).astype(int)
print(st.T.to_string())
