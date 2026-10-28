# Label validation report: are the report-derived labels useful?

Scripts: `src/audit/audit_existing_labels.py`, `src/labeling/validate_labels.py`, `src/labeling/metrics.py`.
Tables: `reports/eda/label_audit/`, `reports/eda/label_validation/`.
Per-cell error analysis with report evidence: `data/labels_v3/gold_error_analysis.csv`. It quotes report text, so it is git-ignored and must not be published.

## 1. Reference labels and what they measure

- **Reference.** The 58 studies in the official `train.csv` with all 12 labels, strictly 0/1. No other human labels exist in the repository.
- **Provenance.** These labels are probably an **independent image read**: several participants say so, and RSNA says the evaluation data was "annotated by expert radiologists". The host has not confirmed it on the pages I could read. The gold set is enriched (every study has at least one positive; 4.1 positives per study on average).
- **Consequence.** Agreement between a report labeller and gold mixes two different things:
  - **Question A**: does the report contain the information? (report vs image discordance)
  - **Question B**: does the method extract it correctly?

  §5 separates them as far as the data allows.
- **Metric emphasis.** The competition metric is macro ROC-AUC, so per-class AUC (threshold-free) is the primary statistic. Threshold metrics use 0.5 and are secondary. With n = 58 and 9–35 positives per class, per-class AUC 95% CIs are typically ±0.08–0.15 wide.

## 2. Evaluation-contamination ledger

| Source | Developed or tuned using gold? | Consequence |
|---|---|---|
| Qwen2.5-7B / 14B prompts (old pipeline) | Mapping guide written before the gold benchmark; regenerated on gold after the fact | Gold metrics roughly independent |
| gemini (`without_labels.csv`) | Single zero-shot pass | Independent, but hosted-LLM use is a compliance grey zone |
| gpt56sol, report_v2 (public sets) | Unknown. Public label authors typically iterate prompts against gold (e.g. the DSPy "deathmatch" notebook evaluated programs on gold). | Gold metrics are **probably optimistic** |
| consensus (current training labels) | **Source weights = per-class AUC on the same 58 gold rows**; gold rows copied into the output | Its "honest AUC" in `logs/label_aggregation.txt` is **not honest**. The final consensus can't be evaluated on gold at all. |
| labels_v3 LLM (Qwen3-4B, prompt v3.1) | Developed on 60 non-gold dev studies only; prompt locked before the first gold evaluation; a-priori score `pA + 0.5 pB + 0.25 pE` | **Independent** gold evaluation (one gold report was read once before development; see `data/labels_v3/EVALUATION_PROTOCOL.md`) |
| labels_v3 rules (v1.1) | Developed on non-gold reports; locked before gold evaluation | Independent |
| labels_v3 calibrated LLM | Leave-one-study-out on gold | Out-of-fold estimate only |

## 3. Existing labels: silent failures and structural problems (verified)

1. **`public_pseudo` contains no information.** `to_use/train_folds_with_pseudo.csv` has `pseudo_* = 0.5` for **all 12 targets on all 4,349 unlabeled rows** (52,188 cells) and gold copied into the 58 gold rows. It was nevertheless used as a voter in `consensus_label_aggregation.py`. In the Dawid–Skene/CROWDLAB vote matrix, 0.5 ≥ 0.5 makes it vote **positive on every cell**. Its "suspiciously flat 0.98 sensitivity" in the log is exactly this artefact, not evidence of correlation. The walkthrough's "~60% accuracy" claim for these labels has no basis in this file.
2. **The `0.5` fallback with a `≥ 0.5` threshold turns failures into positives.** The old scripts write `{col: 0.5}` for unparseable outputs, with no failure flag. Any threshold at 0.5 then counts every failure as a positive finding. The MedGemma run in `docs/multilingual_llm_medical_classification_chat.md` (AUC 0.5000 and recall 1.0 on all 12 classes) is the signature of 100% fallback, not of a model "predicting 1.0". In the final Qwen files there are no all-0.5 rows (no total parse failures), but **1,744 (7B) / 1,509 (14B) individual cells are exactly 0.5**: model-emitted "uncertain" values that a ≥ 0.5 threshold counts as positive.
3. **Truncation.** Reports were cut at 1,200 characters. That truncates 37.7% of all reports and 50% of gold reports, typically removing the impression. Long reports are harder for every labeller, but the old Qwen false-negative rate on long gold reports (0.30) is much higher than gpt56sol's (0.12). gpt56sol is untruncated; on short reports the gap is 0.30 vs 0.21. This is consistent with lost impressions [suggestive; n = 29 per group].
4. **Language coverage.** The prompt glossary covered Spanish and Dutch (plus Greek in the 14B variant). Turkish (546 reports), Croatian (406), German (262), Bulgarian (220) and French (96) had no glossary. More importantly, 95% of Greek reports were fed with micro-sign-corrupted text.
5. **`combine_first` with gold.** Every generated file contains gold labels in the gold rows, so they look perfect against gold. The project handled this correctly with leakage detection and regeneration, but the `.bak` files and the consensus still carry the copies.
6. **Folds.** `StratifiedGroupKFold(groups=StudyInstanceUID)` is study-level stratified K-fold. The "patient grouping" wording in the walkthrough is incorrect, because patient IDs are per study.
7. **Values out of range or missing.** One Qwen-7B value lies outside [0,1]; 2 Qwen-14B cells are NaN. `report_v2` misses exactly one study, and it is a gold study.

### 3.1 Baseline reproduction status (mission §6.1)

The old pipeline, as recorded from code:
- Model `Qwen/Qwen2.5-7B-Instruct` (14B variant `Qwen/Qwen2.5-14B-Instruct-AWQ`); revision not pinned.
- vLLM with tensor parallelism 2 (TP=2) on 2×T4; temperature 0; `max_tokens` 150 (14B: 220); stop at `<|im_end|>`.
- Hand-built ChatML prompt with a Spanish/Dutch (plus Greek) glossary; report cut at 1,200 characters.
- Output: a JSON dict of 12 floats, extracted by regex; on failure, 0.5 for all 12 without a flag.
- Runtime claim of "~8–10 min on 2×T4" (unverified).

It was **not re-executed** here. vLLM does not run on Windows, and a 7B fp16 model does not fit in 6 GB. Its outputs, including the user's regenerated genuine predictions for the 58 gold reports (`gt58_regenerated_qwen_predictions.csv`), were evaluated as produced. The failure rate in the final files is 0 total failures; 1,744 (7B) / 1,509 (14B) cells equal exactly 0.5.

## 4. Existing labels vs gold (independent where noted in §2)

Macro AUC with joint study-bootstrap 95% CI (`reports/eda/label_audit/gold_macro.csv`):

| Source | n | Macro AUC [95% CI] | Macro F1 @0.5 | Macro balanced acc @0.5 |
|---|---|---|---|---|
| report_v2 (possibly gold-tuned) | 57 | 0.870 [0.836, 0.901] | 0.732 | 0.806 |
| gpt56sol (possibly gold-tuned) | 58 | 0.835 [0.805, 0.864] | 0.748 | 0.817 |
| gemini (hard 0/1) | 58 | 0.819 [0.789, 0.848] | 0.748 | 0.819 |
| Qwen2.5-7B (old) | 58 | 0.802 [0.761, 0.841] | 0.668 | 0.763 |
| Qwen2.5-14B-AWQ (old) | 58 | 0.795 [0.763, 0.826] | 0.678 | 0.774 |

Per-class AUC (point estimates; full CIs, sensitivity, specificity, Brier and ECE in `gold_metrics_per_class.csv`):

| Target (gold +) | Qwen-7B | Qwen-14B | gpt56sol | report_v2 | gemini |
|---|---|---|---|---|---|
| ACL (24) | 0.948 | 0.933 | 0.931 | 0.997 | 0.868 |
| MCL (9) | 0.740 | 0.825 | 0.921 | 0.976 | 0.929 |
| Medial Meniscus (26) | 0.897 | 0.878 | 0.890 | 0.943 | 0.849 |
| Lateral Meniscus (23) | 0.840 | 0.852 | 0.866 | 0.841 | 0.842 |
| Medial OA (15) | 0.877 | 0.901 | 0.891 | 0.908 | 0.885 |
| Lateral OA (11) | 0.728 | 0.779 | 0.824 | 0.789 | 0.813 |
| PF OA (21) | 0.724 | 0.623 | 0.808 | 0.891 | 0.797 |
| Effusion (35) | 0.728 | 0.743 | 0.696 | 0.830 | 0.652 |
| Synovitis (27) | 0.743 | 0.697 | 0.676 | 0.694 | 0.676 |
| Baker's (12) | 0.841 | 0.761 | 0.903 | 0.932 | 0.904 |
| Contusion (19) | 0.755 | 0.823 | 0.814 | 0.768 | 0.818 |
| Fracture (18) | 0.797 | 0.722 | 0.801 | 0.871 | 0.789 |

Two systematic patterns show up across **all** labellers:
- **Effusion.** Sensitivity is about 1.0 but specificity only about 0.30 for gpt56sol, report_v2 and gemini. These labellers mark 16 of the 23 gold-negative knees as effusion-positive. For at least 9 of those, the local LLM and the independent rules both confirm an **explicit effusion statement** in the report (§7). Small or physiological effusions mentioned in reports appear to be graded negative by the gold readers. That is a definitional threshold difference, not an extraction error.
- **Synovitis.** Sensitivity is about 0.41–0.48 for every labeller. About half of gold-positive synovitis is **not reported** (information absent, Question A).

## 5. New methods (labels_v3): locked, one-shot gold evaluation

**Methods.**
- **LLM (v3.1).** Qwen3-4B-Instruct-2507 (nf4, local) reads the **full** Unicode-normalised report. It returns next-token probabilities over five assertion letters at 12 teacher-forced positions: A present / B possible / C not mentioned / D absent / E prior only. There is no free-text parsing and therefore no failure fallback. All 4,407 rows are `ok`, letter mass ≥ 0.995 everywhere, and nothing was truncated. The score is fixed a priori: `s0 = pA + 0.5·pB + 0.25·pE`.
- **Rules (v1.1).** 9-language ConText-style extractor with pre- and post-negation, uncertainty, history-section removal and evidence spans.

Macro AUC on gold with joint study-bootstrap 95% CI (`reports/eda/label_validation/macro_summary.csv`):

| Source | Macro AUC [95% CI] | Macro F1 @0.5 | Brier (mean) | ECE₅ (mean) |
|---|---|---|---|---|
| ens(LLM canonical + gpt56sol + report_v2) (superseded by the order-averaged version, §11) | **0.884** [0.848, 0.915] | 0.758 | 0.133 | 0.111 |
| ens(gpt56sol + report_v2), public only | 0.882 [0.849, 0.910] | 0.749 | 0.142 | 0.123 |
| ens(all 4 incl. rules) | 0.879 [0.841, 0.912] | 0.758 | 0.133 | 0.119 |
| report_v2 (n=57) | 0.870 [0.836, 0.903] | 0.732 | 0.146 | 0.141 |
| ens(LLM + rules), local-only | 0.859 [0.819, 0.895] | 0.762 | 0.141 | 0.124 |
| **LLM v3.1 alone (a priori s0)** | **0.857** [0.815, 0.893] | 0.735 | 0.147 | 0.133 |
| gpt56sol | 0.835 [0.804, 0.864] | 0.748 | 0.170 | 0.174 |
| Qwen2.5-7B (old pipeline) | 0.802 [0.759, 0.838] | 0.668 | 0.173 | 0.156 |
| Qwen2.5-14B-AWQ (old pipeline) | 0.795 [0.763, 0.822] | 0.678 | 0.173 | 0.164 |
| rules v1.1 | 0.791 [0.751, 0.827] | 0.673 | 0.175 | 0.138 |

Paired differences (same 57 studies; `paired_comparisons.csv`):

| Comparison | Δ macro AUC [95% CI] | Share of bootstrap draws ≤ 0 |
|---|---|---|
| LLM v3.1 − old Qwen-7B | **+0.055** [0.012, 0.093] | 0.003 |
| LLM v3.1 − gpt56sol | +0.021 [−0.002, 0.045] | 0.039 |
| LLM v3.1 − report_v2 | −0.014 [−0.043, 0.014] | 0.82 |
| LLM v3.1 − public ensemble | −0.026 [−0.051, −0.001] | 0.98 |
| ens(LLM + public) − public ensemble | +0.002 [−0.011, 0.016] | 0.38 |
| rules − old Qwen-7B | −0.012 [−0.053, 0.027] | 0.73 |

Per-class AUC: new LLM vs the final ensemble (full CIs in `per_class_metrics.csv`):

| Target (gold +) | LLM v3.1 | ens(LLM+gpt56+v2) | rules v1.1 | Main limitation |
|---|---|---|---|---|
| ACL (24) | 0.932 | 0.979 | 0.908 | |
| MCL (9) | 0.939 | 0.961 | 0.889 | few positives |
| Medial Meniscus (26) | 0.895 | 0.927 | 0.915 | |
| Lateral Meniscus (23) | 0.861 | 0.881 | 0.782 | |
| Medial OA (15) | 0.913 | 0.923 | 0.876 | |
| Lateral OA (11) | 0.871 | 0.817 | 0.799 | few positives, wide CI |
| PF OA (21) | 0.813 | 0.874 | 0.678 | 24% of report-"absent" knees are gold-positive |
| Effusion (35) | 0.781 | 0.871 | 0.629 | small effusions in reports are gold-negative |
| Synovitis (27) | 0.733 | 0.740 | 0.652 | usually unreported; 25% of report-"absent" are gold-positive |
| Baker's (12) | 0.920 | 0.931 | 0.826 | |
| Contusion (19) | 0.793 | 0.852 | 0.754 | many "possible" (non-traumatic edema) calls |
| Fracture (18) | 0.831 | 0.853 | 0.783 | 20% of "not mentioned" are gold-positive |

## 6. Calibration

- **Self-reported confidences in old outputs are not probabilities.** For example, gpt56sol's `__conf` has median 1.00. The old Qwen "probabilities" have ECE₅ 0.15–0.16 against gold.
- **The letter probabilities have a sensible ordinal structure.** P(gold = 1 | LLM category) is A 0.80 (n = 199 cells), B 0.39 (117), E 0.33 (3), C 0.10 (107) and D 0.09 (270). A pooled logistic fit gives weights pA 3.48, pB 1.56, pC −0.06, pE 0.69, close to the a-priori 1 / 0.5 / 0 / 0.25 ratios.
- **Out-of-fold calibration** (pooled LR with per-target intercepts) improves Brier (0.147 → 0.133) and ECE₅ (0.133 → 0.059). Under cross-fitting it *lowers* AUC (LOO 0.750; repeated 5-fold 0.759). That drop is an **evaluation artefact**: a single monotone per-target mapping cannot change within-target AUC. Cross-fitting scores each fold with a different intercept fitted on prevalence that is anti-correlated with the held-out fold. For the AUC metric, use the uncalibrated a-priori score. Use calibrated values only where probabilities matter (e.g. loss weighting). With 58 enriched gold studies, **calibrated probabilities transfer only to gold-like prevalence**, not to the population.

## 7. Question A vs Question B: report information vs extraction

Each gold cell that is wrong at 0.5 is categorised (`error_categories_by_target.csv`; per-cell detail with evidence spans in `data/labels_v3/gold_error_analysis.csv`, git-ignored). "Discordance" means both the LLM and the independent rules agree on what the report says, yet gold disagrees.

| Target | Correct @0.5 | FP, report explicitly positive (A) | FP, other (B) | FN, report explicitly negative (A) | FN, not mentioned (A) | FN, other (B) |
|---|---|---|---|---|---|---|
| ACL | 52 | 3 | 1 | 0 | 0 | 2 |
| MCL | 49 | 3 | 5 | 0 | 0 | 1 |
| Medial Meniscus | 49 | 3 | 3 | 0 | 0 | 3 |
| Lateral Meniscus | 47 | 1 | 6 | 2 | 0 | 2 |
| Medial OA | 51 | 2 | 4 | 0 | 0 | 1 |
| Lateral OA | 50 | 2 | 4 | 0 | 1 | 1 |
| PF OA | 43 | 0 | 9 | 0 | 1 | 5 |
| Effusion | 42 | **9** | 6 | 0 | 0 | 1 |
| Synovitis | 39 | 3 | 4 | 0 | 3 | **9** |
| Baker's | 52 | 2 | 2 | 0 | 0 | 2 |
| Contusion | 40 | 4 | **12** | 0 | 1 | 1 |
| Fracture | 47 | 3 | 2 | 0 | 3 | 3 |

(A = information/definition discordance between report and image read; B = extraction, uncertainty handling or definition mismatch.)

Interpretation:
- **Information availability limits Effusion, Synovitis, Fracture and PF OA.** 9 of 15 Effusion false positives are effusions *explicitly reported* in knees the gold readers called negative, which points to a size threshold. Synovitis and Fracture have gold positives that the report never mentions or explicitly denies. No report extractor can recover these.
- **Extraction and definitions limit Contusion, PF OA, MCL and Lateral meniscus.** Contusion "FP other" (12) is dominated by "possible" calls on non-traumatic or reactive marrow edema. PF OA "other" (14) comes from mild chondral change graded B. These are definition-threshold questions where the model followed the prompt's own (a-priori) definitions.
- **Reference labels inconsistent with their own report.** 33 cells are "explicitly positive in the report but gold-negative", and 2 are "explicitly negative but gold-positive". Under the image-read hypothesis these are expected disagreements, not label errors. **Report-vs-gold disagreement is not automatically the extractor's fault.**

## 8. Per-language performance (cell level; **2–28 gold studies per language, so very noisy**)

| Language (gold studies) | LLM v3.1 pooled cell AUC | Old Qwen-7B | gpt56sol | report_v2 |
|---|---|---|---|---|
| English (28) | 0.888 | 0.804 | 0.860 | 0.892 |
| Spanish (10) | 0.907 | 0.847 | 0.882 | 0.932 |
| Turkish (6) | 0.858 | 0.847 | 0.838 | 0.838 |
| Croatian (4) | 0.907 | 0.890 | 0.848 | 0.898 |
| Bulgarian (3) | 0.908 | 0.693 | 0.833 | 0.842 |
| Greek (3) | 0.680 | 0.769 | 0.717 | 0.828 |
| Dutch (2) | 0.848 | 0.830 | 0.778 | 0.885 |
| German (2) | 0.711 | 0.689 | 0.743 | 0.775 |

Greek and German are the weakest languages for every labeller. On 2–3 studies these differences are not interpretable per language. The Greek micro-sign normalisation was verified on text, but the 4B model still makes clear errors on long Greek reports (§10).

## 9. Final labels produced (`data/labels_v3/`, git-ignored)

| File | Content |
|---|---|
| `gold_labels_original.csv` | The 58 official labels, untouched |
| `pseudo_labels_v3.csv` | 4,349 non-gold studies. Per target: final soft label; assertion category; LLM pA–pE and s0; rules state and **evidence span**; gpt56sol and report_v2 values; n_sources; source SD; agreement flag; "report silent" flag. Per study: language, LLM status, model/prompt/ruleset versions. |
| `training_labels_v3.csv` | Gold rows (hard 0/1, `label_source=gold`, `is_human_verified=True`) plus pseudo rows (soft) |
| `training_labels_v3_masked.csv` | As above, but cells where both LLM and rules find no mention are NaN (masked variant, **not** recommended; see `baseline_comparison.md`) |
| `unresolved_studies.csv` | 113 flagged studies: 107 template reports, 4 empty-findings reports, 2 pixel-duplicate studies. No LLM failures. |
| `provenance.json` | Versions, SHA-1 hashes, counts |
| `llm_qwen3_4b_assert_v3.1_raw.csv`, `rules_v1_labels.csv` | Raw source outputs |

**Primary label.** The mean of {order-averaged LLM s0, gpt56sol, report_v2} (gold macro AUC 0.890, §11). The pre-specified 4-source mean (including rules) is kept as `{t}__ens4`. Excluding the measurably weaker rules was decided after seeing gold results and is logged in `data/labels_v3/CHANGELOG_after_gold.md`.

## 10. Robustness and sensitivity findings

- **Prompt sensitivity.** A Greek report that clearly states a medial meniscus tear was scored A (0.97) by prompt v3.0 and C ("not mentioned", 1.00) by v3.1. Controlled tests excluded prefix caching and Unicode normalisation as causes (same answer under all four combinations). A small model's answers can flip on prompt wording.
- **Order sensitivity.** Reversing the item order changes the argmax category in 26–28% of cells (dev 26%, dev+gold 28%), mostly between adjacent categories (mean |Δscore| 0.06). On gold, averaging both orders raised LLM macro AUC 0.857 → 0.874 (paired +0.017 [0.002, 0.033]). A full reversed pass was therefore run (full-run gain: +0.012 [0.001, 0.026], §11). Adoption is logged as post-hoc.
- **Prefix-cache equivalence** was verified on short and long prompts.

## 11. Order-averaged LLM (final)

The full reversed-order pass completed for all 4,407 studies: all `ok`, letter mass 1.0, no duplicates after one resume (the laptop slept at about 05:08 and the run resumed at 08:41). Gold metrics (`order_avg_macro.csv`, `order_avg_paired.csv`, `micro_metrics.csv`):

| Source | Macro AUC [95% CI] | Micro AUC (696 cells) |
|---|---|---|
| LLM canonical order (locked) | 0.857 [0.815, 0.893] | 0.865 |
| LLM reversed order | 0.862 [0.825, 0.896] | 0.869 |
| **LLM order-averaged** | **0.868** [0.830, 0.902] | 0.876 |
| ens(LLM canonical + gpt56sol + report_v2) | 0.884 [0.848, 0.915] | 0.895 |
| **ens(LLM order-averaged + gpt56sol + report_v2): final primary teacher** | **0.890** [0.853, 0.922] | **0.898** |
| ens(gpt56sol + report_v2), public only | 0.882 [0.849, 0.910] | 0.893 |
| report_v2 / gpt56sol / old Qwen-7B | 0.870 / 0.835 / 0.802 | 0.880 / 0.837 / 0.806 |

Paired deltas:
- Order-averaged − canonical LLM: **+0.012 [0.001, 0.026]**.
- Final teacher − canonical-LLM ensemble: +0.005 [−0.002, 0.014].
- Final teacher − public-only ensemble: +0.007 [−0.007, 0.022] (not resolved).
- Order-averaged LLM alone − public-only ensemble: −0.014 [−0.035, 0.008].

Per class, order averaging helps most where the canonical pass was weakest: Contusion 0.793 → 0.823, Fracture 0.831 → 0.869, ACL 0.932 → 0.962, Synovitis 0.733 → 0.751. Argmax agreement between the two orders on gold is only 0.724.

Gold informed the adoption of order averaging (`CHANGELOG_after_gold.md` #3), so these gains are optimistic. The motivation (order variance) was established on dev data.

## 12. Verdict

1. **Existing labels.** The consensus and the public label sets are usable training targets. `public_pseudo` contains no labels. The old Qwen labels are the weakest LLM labels in the repo (gold 0.80). The reported "honest" AUCs of the consensus and the v1–v4 models are **not** independent estimates.
2. **New local labeller.** It is a measurable improvement over the previous local pipeline (+0.055 vs old Qwen-7B on an independent, locked evaluation) and is compliant, reproducible and failure-transparent. It does **not** beat the best public label set on its own.
3. **Final teacher.** The mean of {order-averaged local LLM, gpt56sol, report_v2} has the highest gold agreement measured here (0.890 macro AUC), but its margin over the public-only ensemble is not statistically resolved on 58 studies.
4. **Ceiling.** Gold agreement of about 0.88–0.90 is consistent with the report→image ceiling estimated by other teams. The residual errors are dominated by report/image discordance (Effusion threshold, unreported Synovitis/Fracture) and definitional thresholds (OA grade, non-traumatic edema), not by language or parsing failures.
