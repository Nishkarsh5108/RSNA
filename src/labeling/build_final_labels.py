"""Assemble the labels_v3 deliverables with full provenance.

Outputs (data/labels_v3/, git-ignored):
  gold_labels_original.csv      the 58 official labels, untouched
  pseudo_labels_v3.csv          one row per NON-gold study; per target:
                                  {t}               final soft label (mean of available independent sources)
                                  {t}__assertion    LLM argmax category (present/possible/not_mentioned/absent/prior)
                                  {t}__llm_s0, {t}__llm_pA..pE, {t}__rules_state, {t}__rules_evidence,
                                  {t}__gpt56sol, {t}__report_v2, {t}__n_sources, {t}__source_sd, {t}__agree
                                study-level: lang, llm_status, flags, model/prompt/ruleset versions
  training_labels_v3.csv        gold rows (hard 0/1, label_source=gold) + pseudo rows (soft, label_source=pseudo)
  training_labels_v3_masked.csv same, but cells where the LLM says NOT MENTIONED *and* rules found no mention are NaN
  unresolved_studies.csv        every study with a failed/flagged LLM row, empty findings, truncation, duplicate images
  provenance.json               versions, hashes, counts
"""
import hashlib
import json
import warnings
warnings.filterwarnings("ignore", category=Warning)
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from metrics import TARGETS  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/data/labels_v3"
LET = ["A", "B", "C", "D", "E"]
CAT = {"A": "present", "B": "possible", "C": "not_mentioned", "D": "absent", "E": "prior_only"}

train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv").set_index("StudyInstanceUID")
gold_mask = train[TARGETS].notna().all(axis=1)
gold = train.loc[gold_mask, TARGETS].astype(int)
prof = pd.read_csv(f"{ROOT}/reports/eda/report_profile.csv").set_index("StudyInstanceUID")
lang = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})

llm_path = f"{OUT}/llm_qwen3_4b_assert_v3.1_raw.csv"
llm = pd.read_csv(llm_path).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID")
rev_path = f"{OUT}/llm_qwen3_4b_assert_v3.1_reversed_raw.csv"
ORDER_AVG = os.path.exists(rev_path)
if ORDER_AVG:  # average letter probabilities over canonical and reversed item order (CHANGELOG_after_gold.md #3)
    rev = pd.read_csv(rev_path).drop_duplicates("StudyInstanceUID", keep="last").set_index("StudyInstanceUID").reindex(llm.index)
    both_ok = llm["status"].eq("ok") & rev["status"].eq("ok")
    for t in TARGETS:
        cols = [f"{t}__p{L}" for L in LET]
        llm.loc[both_ok, cols] = (llm.loc[both_ok, cols].values + rev.loc[both_ok, cols].values) / 2
    llm["order_averaged"] = both_ok
rules = pd.read_csv(f"{OUT}/rules_v1_labels.csv").set_index("StudyInstanceUID")
MD = f"{ROOT}/data/main_data/to_use"
gpt = pd.read_csv(f"{MD}/report_labels_gpt56sol.csv").set_index("StudyInstanceUID")[TARGETS]
v2 = pd.read_csv(f"{MD}/report_labels_v2.csv").set_index("StudyInstanceUID")[TARGETS]

ids = train.index
ok = llm.reindex(ids)["status"].eq("ok")
rows = pd.DataFrame(index=ids)
rows["lang"] = lang.reindex(ids)
rows["llm_status"] = llm.reindex(ids)["status"].fillna("missing")
rows["report_truncated"] = llm.reindex(ids)["report_truncated"].fillna(False)
for t in TARGETS:
    p = llm.reindex(ids)[[f"{t}__p{L}" for L in LET]]
    s0 = p.iloc[:, 0] + 0.5 * p.iloc[:, 1] + 0.25 * p.iloc[:, 4]
    s0 = s0.where(ok)  # failed/flagged LLM rows contribute nothing (never imputed)
    am = p.values.argmax(1)
    rows[f"{t}__assertion"] = np.where(ok, [CAT[LET[a]] for a in am], "unavailable")
    for li, L in enumerate(LET):
        rows[f"{t}__llm_p{L}"] = p.iloc[:, li].where(ok)
    rows[f"{t}__llm_s0"] = s0
    rows[f"{t}__rules_state"] = rules.reindex(ids)[f"{t}__state"]
    rows[f"{t}__rules_evidence"] = rules.reindex(ids)[f"{t}__evidence"]
    rows[f"{t}__rules"] = rules.reindex(ids)[t]
    rows[f"{t}__gpt56sol"] = gpt.reindex(ids)[t]
    rows[f"{t}__report_v2"] = v2.reindex(ids)[t]
    src = rows[[f"{t}__llm_s0", f"{t}__rules", f"{t}__gpt56sol", f"{t}__report_v2"]].astype(float)
    rows[f"{t}__ens4"] = src.mean(axis=1, skipna=True)
    # PRIMARY (post-hoc, see CHANGELOG_after_gold.md): rules excluded from the average
    rows[t] = rows[[f"{t}__llm_s0", f"{t}__gpt56sol", f"{t}__report_v2"]].astype(float).mean(axis=1, skipna=True)
    rows[f"{t}__n_sources"] = src.notna().sum(axis=1)
    rows[f"{t}__source_sd"] = src.std(axis=1, ddof=0)
    rows[f"{t}__agree"] = ((src >= 0.5).sum(axis=1).isin([0, src.notna().sum(axis=1)])) & (src.notna().sum(axis=1) > 0)
    rows[f"{t}__report_silent"] = (rows[f"{t}__assertion"] == "not_mentioned") & (rows[f"{t}__rules_state"] == "not_mentioned")

meta = {"llm_model": "Qwen/Qwen3-4B-Instruct-2507", "llm_revision": llm["model_revision"].dropna().iloc[0] if "model_revision" in llm else None,
        "prompt_version": llm["prompt_version"].dropna().iloc[0], "prompt_hash": llm["prompt_hash"].dropna().iloc[0],
        "ruleset_version": rules["ruleset_version"].iloc[0], "llm_item_order": "canonical+reversed averaged" if ORDER_AVG else "canonical", "final_label": "mean of available sources {llm_s0, gpt56sol, report_v2}; {t}__ens4 adds rules (see CHANGELOG_after_gold.md)"}
for k, v in meta.items():
    rows[k] = v

pseudo = rows.loc[~gold_mask].copy()
pseudo["label_source"] = "pseudo"
pseudo.reset_index().to_csv(f"{OUT}/pseudo_labels_v3.csv", index=False)
gold.reset_index().to_csv(f"{OUT}/gold_labels_original.csv", index=False)

train_lab = pd.concat([gold.astype(float).assign(label_source="gold", is_human_verified=True),
                       pseudo[TARGETS].assign(label_source="pseudo", is_human_verified=False)])
train_lab.index.name = "StudyInstanceUID"
train_lab = train_lab.reindex(ids)
train_lab.reset_index().to_csv(f"{OUT}/training_labels_v3.csv", index=False)
masked = train_lab.copy()
for t in TARGETS:
    silent = pseudo[f"{t}__report_silent"]
    masked.loc[silent[silent].index, t] = np.nan
masked.reset_index().to_csv(f"{OUT}/training_labels_v3_masked.csv", index=False)

# unresolved / flagged studies (never silently dropped)
dups = pd.read_csv(f"{ROOT}/reports/eda/metadata/duplicate_study_pairs_pixel_verified.csv")
dup_ids = set(dups.loc[dups.shared_identical_slices > 0, ["study_a", "study_b"]].values.ravel())
empty_findings = set(train.index[train.Report.str.strip().str.contains(r"(?i)bevindingen\s*:?\s*$", regex=True)])
unres = []
for u in ids:
    reasons = []
    if rows.loc[u, "llm_status"] != "ok":
        reasons.append(f"llm_status={rows.loc[u, 'llm_status']}")
    if bool(rows.loc[u, "report_truncated"]):
        reasons.append("report_token_truncated")
    if u in empty_findings:
        reasons.append("empty_findings_section")
    if u in dup_ids:
        reasons.append("pixel_duplicate_of_another_study_with_different_report")
    if prof.loc[u, "n_chars"] < 150:
        reasons.append("very_short_template_report")
    if reasons:
        unres.append({"StudyInstanceUID": u, "is_gold": bool(gold_mask[u]), "reasons": ";".join(reasons)})
pd.DataFrame(unres).to_csv(f"{OUT}/unresolved_studies.csv", index=False)

prov = {**meta, "n_studies": int(len(ids)), "n_gold": int(gold_mask.sum()), "n_pseudo": int((~gold_mask).sum()),
        "llm_status_counts": rows["llm_status"].value_counts().to_dict(),
        "n_unresolved_flagged": len(unres),
        "masked_cells_fraction_pseudo": float(masked.loc[~gold_mask, TARGETS].isna().mean().mean()),
        "sha1_llm_raw": hashlib.sha1(open(llm_path, "rb").read()).hexdigest(),
        "sha1_rules": hashlib.sha1(open(f"{OUT}/rules_v1_labels.csv", "rb").read()).hexdigest(),
        "sha1_train_csv": hashlib.sha1(open(f"{ROOT}/data/Raw_dicom/train.csv", "rb").read()).hexdigest()}
json.dump(prov, open(f"{OUT}/provenance.json", "w"), indent=1, default=str)
print(json.dumps(prov, indent=1, default=str))
print("assertion distribution (pseudo):")
print(pd.DataFrame({t: pseudo[f"{t}__assertion"].value_counts() for t in TARGETS}).fillna(0).astype(int).T.to_string())
