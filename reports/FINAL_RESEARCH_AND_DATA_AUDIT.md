# Final research and data audit: RSNA knee label pipeline (labels_v3)

Date: 2026-10-17 (overnight autonomous run). Competition deadline: **2026-10-22 23:59 UTC**.
Companion reports: `dataset_inventory.md`, `competition_rules_audit.md`, `literature_review.md`, `label_validation_report.md`, `baseline_comparison.md`, `EDA_FINAL.md`.

Each statement is tagged as one of:
- **[measured]**: produced by a script in this run;
- **[verified]**: checked against data or primary sources;
- **[literature]** or **[claim]**: from external sources;
- **[hypothesis]**: not tested.

## A. Executive summary

**What was found.**
- The dataset is as described (4,407 studies; 58 gold; 4,349 report-only).
- The gold set is small, enriched and, by participant accounts, an independent *image* read. Report-vs-gold disagreement therefore has a ceiling of roughly 0.88–0.90 macro AUC that no extractor can pass.
- The existing label pipeline had real silent failures:
  - a "public pseudo-label" file that is **0.5 everywhere** but was used as a consensus voter;
  - a 0.5 failure fallback that a ≥ 0.5 threshold turns into positives;
  - 1,200-character truncation of 38% of reports;
  - Greek reports corrupted by micro-sign substitution (95%);
  - consensus weights and checkpoint selection fitted on the same gold rows used to report "honest" AUC.
- **Compliance.** The GitHub repo is **public and contains competition data**. Make it private now.

**What was executed.**
- Full DICOM header inventory (819k files) and pixel and report EDA.
- A new local, compliant labeller: Qwen3-4B forced-choice assertion probabilities on full, normalised reports, run twice over all 4,407 studies in two item orders with 0 failures.
- A 9-language rule extractor.
- A locked, independent gold evaluation with bootstrap CIs.
- A downstream test: 20+ label sets, trained on non-gold only, frozen-feature image model, 5 seeds, paired bootstrap.
- A tested Kaggle notebook.

**What worked.**
- The new local labeller beats the old pipeline (gold macro AUC 0.857 vs 0.802, paired +0.055 [0.012, 0.093]). Order averaging lifts it to 0.868.
- The final teacher, mean(order-averaged LLM, gpt56sol, report_v2), reaches **0.890** [0.853, 0.922] on gold, the best measured.
- Pseudo-labels are essential: an image model trained on them reaches 0.75–0.78 gold AUC versus 0.53–0.60 when trained on the 58 gold studies alone.

**What did not.**
- On their own, the local LLM and the rules are below the best public label set.
- Label improvements of this size are **invisible downstream** with frozen features (all label sets 0.75–0.78; only "rules are worse" is resolved).
- Masking "not mentioned" cells hurt.

**What remains unresolved.**
- Official confirmation of how gold labels were produced.
- Provenance of the public label sets.
- Whether label quality matters more with a fine-tuned image model.
- The leaderboard impact of any of this.

**Bottom line for the next 5 days.** Labels are no longer the bottleneck. The ranked plan is in section L:
1. Fix data exposure.
2. Make test decoding fail loudly.
3. Fine-tune a 2.5D model on `data/labels_v3/training_labels_v3.csv`.
4. Normalise laterality with the geometry table.



## B. Dataset audit

- **[verified] Counts.** 4,407 studies; 58 gold (strictly 0/1 on all 12 targets); 4,349 report-only; 24,371 series; 819,078 DICOMs (569 GB). Reports are present for every study. There are 0 unreadable headers and 0 missing series, and every study has all three planes. The walkthrough's 4,407 / 58 / 4,349 are confirmed.
- **[verified] Patient identifiers** are pseudonymised per study (4,407 unique), so patient-level grouping is impossible.
- **[verified] One pixel-identical duplicate study pair** (`…23376711` / `…31613209`) carries two **conflicting** reports (left vs right knee, different findings). It was found by an exhaustive embedding search plus pixel-hash verification of 42 candidate pairs.
- **[verified] Orientation conventions are uniform per plane.** Medial/lateral flips with knee side. **Knee side is recoverable from geometry for 100% of series**, with 97.8% agreement with the DICOM tag, which itself exists for only about 50%.
- **[verified] Report corruption and structure.**
  - 95% of Greek reports have micro-sign substitution.
  - English and French reports are long, so the old 1,200-character cut removed about 20% of their text.
  - Dutch clinical-question blocks precede the findings mid-sentence.
  - There are 4 empty-findings reports and 107 short templates.
- **[verified] Language = site = vendor confounding** (e.g. Greek, German and Dutch are Siemens-only). Intensity scale differs about 25× across vendors.
- **[verified] Hidden-test decoding risk.** The official data page lists JPEG-Lossless/JPEG-2000 transfer syntaxes. The local copy is uncompressed, and this machine lacks a JPEG-Lossless decoder.

## C. Competition compliance

- **[verified]** Open-weight pretrained models and public external data are allowed. Internet is off at submission. Reports are absent at test time, so they can only supervise training. The metric is macro ROC-AUC.
- **[verified]** The data-security clause forbids providing competition data to non-participants. **All report processing in this run was local.** Previous hosted-LLM (Gemini) processing of the 58 gold reports is a grey zone and should be disclosed or confirmed with the host.
- **[verified] Action required by the owner: the GitHub repository `Nishkarsh5108/RSNA` is public and tracks the competition `train.csv` (all reports and gold labels) and a raw DICOM.** This likely breaches the no-redistribution rule. Make the repository private and purge the data from history. No change was made to the remote or history. New data-derived outputs were added to `.gitignore`.

## D. Literature review (summary)

- **[literature]** Report labels and image labels disagree substantially even with expert readers (VisualCheXbert: κ ≈ 0.31–0.43 on CXR). Better report labellers produce better image classifiers.
- **[literature]** The uncertainty category matters. CheXpert's U-policies change F1 substantially per class. Labelling "not mentioned" as negative is a modelling choice, not a fact.
- **[literature]** Local LLM labellers are viable and privacy-preserving but weaker in non-English (Llama-2-70B: German MCC 0.66 vs English 0.75).
- **[claim]** Public solutions in this competition reach about 0.92–0.93 public LB with a single 2.5D fine-tuned reader on report-derived soft labels, and 0.944 with stacks. The best public label sets agree with gold at about 0.87–0.89 macro AUC. One author estimates a 0.88–0.92 ceiling for report-derived labels against gold.
- Full table with sources: `literature_review.md`.

## E. Current label-generation assessment (existing pipeline)

Verdict: **usable after correction, but not trustworthy as documented.**

| Issue | Evidence | Severity |
|---|---|---|
| The `public_pseudo` source holds **no labels**: exactly 0.5 on all 4,349 unlabeled rows × 12 targets | `label_audit/sources_summary.csv` [measured] | High. It was a voter in the consensus that votes "positive" everywhere (0.5 ≥ 0.5). Its "flat 0.98 sensitivity" in the aggregation log is this artefact. |
| A 0.5 fallback without a failure flag, combined with a ≥ 0.5 threshold, turns failures into positives | Code review; the MedGemma run (AUC 0.500, recall 1.0 on all classes) is the fallback signature | High (method); no total failures remain in the final Qwen files, but 1,744 / 1,509 cells are exactly 0.5 |
| 1,200-character truncation | Cuts 37.7% of reports and 50% of gold. Old Qwen false-negative rate on long gold reports is 0.30 vs 0.12 for an untruncated labeller [measured, suggestive] | Medium |
| No glossary for Turkish, Croatian, German, Bulgarian or French; Greek micro-sign corruption | [verified] | Medium |
| Consensus weights fitted on the same 58 gold rows used to report its "honest AUC" | Code review [verified] | High for **evaluation** claims (not honest); low for training value |
| Folds "grouped by patient" | Actually grouped by study (patient IDs per study) [verified] | Documentation error |
| Gold-based checkpoint selection in v1–v4 | Code review [verified] | High for reported metrics |

Measured quality of existing sources against gold:
- old Qwen-7B 0.802 and Qwen-14B 0.795 macro AUC;
- gpt56sol 0.835 and report_v2 0.870, probably optimistic (likely gold-tuned prompts).

Downstream, the current **consensus** labels train as well as any single public source (0.772–0.775). The corrupted voter did little damage in practice, because gpt56sol and report_v2 dominated the blend.

## F. Improved methodology (implemented and executed)

1. **Local multilingual LLM assertion scorer** (`src/labeling/llm_assertion_labeler.py`). Qwen3-4B-Instruct-2507 (nf4) with full reports, Unicode normalisation, and five assertion categories (present / possible / not mentioned / absent / prior only) read as **next-token probabilities under forced decoding**. There is no parsing, so there are no silent failures. A shared-prefix KV cache gives a 12× speedup, verified equivalent. Each row records explicit status, provenance (model revision, prompt hash) and a format-adherence check. It is resumable.
2. **Transparent rule extractor** (`src/labeling/rule_extractor.py`) for 9 languages, with pre- and post-negation, uncertainty, history-section removal and evidence spans. It is used for provenance and evidence, **not** as a training source (it was the weakest source).
3. **Item-order averaging.** The canonical and reversed answer orders are averaged to reduce the 4B model's order sensitivity (post-hoc, logged).
4. **Final teacher**: unweighted mean of {order-averaged LLM, gpt56sol, report_v2} (post-hoc exclusion of rules, logged). Gold labels are kept separately and untouched. Each assertion category, evidence span, source disagreement and "report silent" flag is preserved per cell.
5. **Validation protocol.** Development used non-gold studies only; the method was locked; there was a one-shot gold evaluation; calibration was out-of-fold only; and a downstream test trained on non-gold only with fixed epochs, 5 seeds and a paired bootstrap.

Alternatives considered:
- **Qwen3.5-4B/9B** use hybrid linear attention, which has unreliable kernels on Windows.
- **Qwen3-8B, EuroLLM-9B and Gemma-4-E4B** are 16–18 GB downloads with VRAM too tight for 6 GB.
- **Hosted APIs** were excluded by the compliance risk.
- **Dawid–Skene/CROWDLAB** don't fit: the sources are few and correlated.
- **CheXbert-style fine-tuning** would need hundreds of report-level expert labels.
- **Kaggle T4 execution**: unnecessary, because local execution was viable. The Kaggle notebook is provided for reproducibility or for running a larger model.

## G. Quantitative validation (gold, n = 58, image-read reference)

| Source | Macro AUC [95% CI] | Note |
|---|---|---|
| **Final teacher: mean(order-averaged LLM, gpt56sol, report_v2)** | **0.890** [0.853, 0.922] | gold-informed ensemble choice; canonical-LLM version 0.884 |
| mean(gpt56sol, report_v2) | 0.882 [0.849, 0.910] | |
| report_v2 | 0.870 [0.836, 0.903] | n = 57; probably gold-tuned |
| **Local LLM v3.1 (a priori, locked)** | **0.857** [0.815, 0.893] | independent evaluation |
| gpt56sol | 0.835 [0.804, 0.864] | probably gold-tuned |
| Old Qwen-7B / 14B | 0.802 / 0.795 | |
| Rules v1.1 | 0.791 [0.751, 0.827] | |

- Paired: local LLM − old Qwen-7B = **+0.055 [0.012, 0.093]**.
- Weakest class for every labeller: **Synovitis (≈0.67–0.74)**, because it is largely unreported.
- Effusion specificity is about 0.30 for all strong labellers, because reported small effusions are gold-negative.
- About 33 of the 696 gold cells are "explicitly positive in the report but gold-negative" and 2 are "explicitly negative but gold-positive". These are report/image discordance, not extraction error.
- Calibration: out-of-fold Brier 0.147 → 0.133 and ECE 0.133 → 0.059. AUC under cross-fitting is artefactually lower (see `label_validation_report.md` §6).

**Final (order-averaged) results.** Order-averaged local LLM: **0.868** [0.830, 0.902] (+0.012 [0.001, 0.026] vs canonical). **Final primary teacher**, mean(order-averaged LLM, gpt56sol, report_v2): **0.890** [0.853, 0.922]; micro AUC 0.898. Against the public-only ensemble: +0.007 [−0.007, 0.022] (not resolved). Downstream image model trained on the final teacher: 0.782 [0.735, 0.827], tied with the best alternatives.

## H. Baseline image-model assessment

Frozen EfficientNet-B0 v2 features, v2 head, trained on non-gold only, fixed 12 epochs, 5 seeds, evaluated on all 58 gold studies:
- Pseudo-label training reaches 0.747–0.784 macro AUC, depending on the label set.
- Gold-only training reaches 0.53–0.60.
- **Pseudo-labels are essential.**
- Better labels give small gains: final ensemble 0.784 vs old Qwen-7B 0.761 vs rules 0.747. Only "rules worse than the public mean" is statistically resolved (−0.028 [−0.053, −0.003]).
- The run-to-run noise floor (CPU vs GPU) is about 0.006–0.010.
- **Masking "not mentioned" cells hurt** in all three tests (−0.010 to −0.018).
- Soft labels were slightly better than hard ones (0.770 vs 0.766, within noise).
- The image representation, not the labels, is the binding constraint: public fine-tuned 2.5D readers reach about 0.92–0.93 public LB on similar report-derived labels [claim].

## I. Comprehensive EDA findings (top items; full detail in `EDA_FINAL.md`)

1. Gold is a small, enriched, probably image-read set with no normal knees. Report-derived prevalence is far lower for Synovitis, Fracture, ACL and Lateral meniscus.
2. Medial/lateral sides flip with knee side. Geometry recovers side for 100% of series (97.4% agreement with the tag), while v3/v4 mirror only tagged 'R' series (about 50%).
3. Report text had silent corruption (Greek micro sign 95%, truncation, mid-sentence section headers).
4. Language ≈ site ≈ vendor. Report-derived prevalence varies strongly by site, so labels carry site-specific reporting style.
5. One pixel-duplicated study pair has conflicting reports. Patient IDs can't link repeat patients.
6. Intensity scale ×25 across vendors; 97% of studies have multiple sagittal series; 3D series up to 320 slices; FOV 120–200 mm at the same matrix size.
7. Dedicated "ACL" sequences are a protocol-selection shortcut (50.6% vs 17.3% ACL-positive).
8. The hidden test may contain JPEG-Lossless data. Without decoders, v2/v3 zero-fill planes silently. Decoding was verified once pylibjpeg/gdcm are installed.

## J. Leakage and evaluation risks

| Risk | Status |
|---|---|
| Gold-based checkpoint selection in v1–v4 ("honest gold AUC") | **Invalidates the reported gold AUCs** (optimistic). Fixed in the downstream protocol here. |
| Consensus weights fitted on gold | Invalidates its reported "honest AUC"; training use is fine |
| Gold rows in training folds of existing checkpoints | Any gold evaluation of `output/v1`, `output/v3` checkpoints is contaminated |
| Report-derived features at inference | Reports are absent at test time; must never be inputs |
| Duplicate study pair across folds | Put it in one fold |
| Protocol shortcut (ACL-named series) | Don't use `SeriesDescription` or series counts as features |
| Public label sets tuned on gold | Their gold metrics are optimistic; treat as external teachers |
| Gold-informed choices in this run (ensemble membership, order averaging) | Logged in `data/labels_v3/CHANGELOG_after_gold.md`; the associated gold gains are optimistic |
| n = 58 gold | Differences below about 0.03 macro AUC are not resolvable |

## K. Hardware and runtime

| Item | Measured |
|---|---|
| Machine | i5-13420H (8C/12T), 32 GB RAM (about 16 GB free at start), RTX 4050 Laptop 6 GB (WDDM; CUDA 11.8 runtime, driver 610.62), NVMe SSDs (D: 305 GB free) |
| DICOM header inventory (24,371 series + 68,180 slice headers) | 3.7 min, 8 processes (about 110 series/s) |
| Report profiling (lingua, 4,407 reports) | about 1 min |
| Rule extractor (4,407 reports) | 2.6 ms/report (about 12 s total) |
| Qwen3-4B-Instruct-2507 download | 8.0 GB in 227 s |
| LLM labeller, 4,407 reports (nf4, prefix cache, batch ≤4, ≤8k tokens) | 0.55 studies/s while CPU-contended; **0.94–1.6 studies/s uncontended**; peak 4.4 GB VRAM. Canonical pass about 77 min including one deliberate stop; reversed pass about 75 min including one involuntary interruption (the laptop slept at about 05:08; resume at 08:41 skipped the 3,694 finished studies and lost nothing). |
| Bottlenecks found | (1) full-vocabulary logits for every prompt position (3.6 GB) caused spill to shared memory: fixed with `logits_to_keep=1`; (2) SDPA math-path memory for long masked prompts: avoided with a cached system prefix; (3) CPU contention from concurrent CPU training slowed the GPU job about 1.7× |
| Downstream head training (CPU, 5 seeds × 12 epochs per label set) | about 3.5 min per label set |
| Prefix-cache equivalence | Verified on short and long (2.5k-token) prompts: identical argmax except near-ties; no |Δscore| > 0.2 in 96 items |

## L. Recommended modelling strategy (ranked; deadline 2026-10-22 23:59 UTC)

| Priority | Action | Why (evidence) | Effort |
|---|---|---|---|
| 0 | **Make the GitHub repo private / purge competition data** | Public redistribution risk; possible disqualification | minutes |
| 1 | **Decoder safety in the submission notebook.** Attach pylibjpeg / pylibjpeg-libjpeg / openjpeg (or gdcm) wheels and call `assert_decoders()` (`src/audit/decode_selftest.py`). | Hidden test may contain JPEG-Lossless. v2/v3 silently zero-fill undecodable planes, and hidden-run logs are invisible. Decoding was verified locally once the packages were installed. | 1 h |
| 2 | **Fine-tune a 2.5D backbone instead of frozen features.** ConvNeXt-T or EfficientNetV2-S, 256–384 px, 3-slice windows, MIL/attention over slices and series, soft BCE on `training_labels_v3.csv` (pseudo rows), 4–5 folds. Never select on gold; select on pseudo-label validation and report gold once. | Frozen ImageNet features give about 0.78 gold / 0.775 LB. Public fine-tuned 2.5D readers on report labels give about 0.92–0.93 public LB [claim]. | 1–2 days (Kaggle 2×T4) |
| 3 | **Laterality normalisation for all studies.** Mirror coronal/axial images of right knees using the tag, or else the study-level geometric side (`reports/eda/metadata/laterality_by_series.csv`). | Medial/lateral targets (menisci, OA, MCL) need consistent sides. v3 mirrors only about 50% of right knees. | 1–2 h |
| 4 | **Use the labels_v3 primary teacher as soft targets; don't mask silent cells; drop the old Qwen files.** | Best gold agreement and best downstream result; masking hurt; old Qwen labels −0.014 to −0.024 downstream | done (files ready) |
| 5 | Per-series percentile normalisation (already done), plus intensity/gamma augmentation and fixed-mm in-plane resampling | ×25 vendor intensity range; FOV 120–200 mm | 2–4 h |
| 6 | Multi-series slots per (plane × fluid-sensitive) with masks; sample slices by physical position | 97% of studies have multiple sagittal series; 3D series up to 320 slices | 2–4 h |
| 7 | Validation hygiene: study-level folds stratified by language/site; duplicate pair in one fold; gold only as a final held-out check, with bootstrap CIs | n = 58; site-specific label style; one duplicate pair | 1 h |
| 8 | (Optional) Blend with a public high-scoring stack. Most teams at 0.943–0.944 fork one [claim]. | Strongest short-term LB lever if allowed by your goals | hours |
| 9 | (Optional, labels) Stronger local LLM pass on Kaggle T4s with the provided notebook (e.g. Qwen3-8B / 14B fp16), averaged into the teacher | The 4B model is order- and prompt-sensitive; ensemble members are complementary | half a day |

What **not** to spend time on, given the measurements: Dawid–Skene/CROWDLAB-style label models (sources are few and correlated); masking "not mentioned" cells; tuning label thresholds (AUC metric); any per-class decision based on gold with fewer than 15 positives.

## M. Unresolved limitations

1. **Gold is tiny and probably image-read.** Per-class gold AUCs have ±0.08–0.15 CIs, and macro differences below about 0.03 are unresolvable. Gold agreement measures a report-to-image proxy, not extraction accuracy alone.
2. **Provenance of gpt56sol and report_v2 is undocumented.** Their gold scores are likely optimistic because their prompts were probably tuned on gold.
3. **The local 4B LLM is prompt-sensitive on some items.** A clear Greek medial-meniscus tear flipped from A (v3.0 prompt) to "not mentioned" (v3.1), with cache and normalisation verified not to be responsible.
4. **The downstream test uses frozen ImageNet features.** Label-quality effects might be larger or smaller with a fine-tuned 2.5D model. That has not been tested here.
5. **Same-patient repeats** cannot be detected beyond pixel duplicates.
6. **Order and prompt sensitivity remain** even after averaging two orders (argmax agreement between orders is 0.72); a larger model was not tested.
7. **Not executed:** a fine-tuned 2.5D image model; a larger-LLM pass (only the 4B model, in two item orders); a decoding test on *actual* hidden-test files (only locally transcoded slices were tested); manual report-level annotation of a dev set to measure extraction accuracy independently of image–report discordance; any Kaggle submission.

## N. Artifact index

**Reports** (`reports/`; aggregate statistics only, no report text):
- `FINAL_RESEARCH_AND_DATA_AUDIT.md` (this file), `dataset_inventory.md`, `competition_rules_audit.md`, `literature_review.md`, `label_validation_report.md`, `baseline_comparison.md`, `EDA_FINAL.md`.

**EDA artefacts** (`reports/eda/`, git-ignored):
- `dicom/series_headers.csv`, `dicom/slice_headers_sample.csv`
- `metadata/`: checks, value counts, `series_table.csv`, `slice_geometry_sample.csv`, `laterality_by_series.csv`, `near_duplicate_pairs_featurecos.csv`, `duplicate_study_pairs_pixel_verified.csv`, `gold_vs_nongold_shift.csv`, figures
- `images/`: pixel statistics, orientation conventions, montage
- `reports_labels/`: language structure, cue expressions, duplicates, prevalence, correlations, by-language rates (rules and LLM), figures
- `label_audit/`: existing-source audit tables
- `label_validation/`: new-method validation tables, error categories, reliability plot
- `downstream/`: all downstream runs, including gold-only baselines
- `study_inventory.csv` (completeness matrix), `report_profile.csv`

**Code**:
- `src/audit/`: `dicom_inventory.py`, `dicom_eda.py`, `report_profile.py`, `audit_existing_labels.py`, `report_label_eda.py`, `image_eda.py`, `laterality_table.py`, `decode_selftest.py`
- `src/labeling/`: `llm_assertion_labeler.py` (labeller; `--reverse-items`), `rule_extractor.py` + `run_rules.py`, `metrics.py`, `validate_labels.py` (validation), `downstream_eval.py` (baseline evaluation), `gold_only_baseline.py`, `order_sensitivity.py`, `make_label_sets_v3.py`, `build_final_labels.py`
- `notebooks/rsna_knee_label_generation.ipynb` (Kaggle notebook, generated from the tested modules by `build_kaggle_notebook.py`; all cells executed locally by `smoke_test_notebook.py`)

**Configuration and environment**: `configs/label_pipeline.yaml`, `requirements-labels-v3.txt`.

**Labels and data** (`data/labels_v3/`, git-ignored because they contain report excerpts):
- `training_labels_v3.csv` (**use this**), `training_labels_v3_masked.csv`
- `pseudo_labels_v3.csv` (full provenance per cell), `gold_labels_original.csv`
- `unresolved_studies.csv` (failed/unresolved list), `provenance.json`
- `llm_qwen3_4b_assert_v3.1_raw.csv`, `llm_qwen3_4b_assert_v3.1_reversed_raw.csv`, `rules_v1_labels.csv`
- `gold_error_analysis.csv` (per-cell error analysis with evidence spans)
- `EVALUATION_PROTOCOL.md`, `CHANGELOG_after_gold.md`
- `dev_pilot_uids.csv`, `gold_uids_LOCKED.csv`
- run logs `llm_run_v3.1*.log`
- `label_sets/` (downstream training targets)

**Model weights**: Qwen3-4B-Instruct-2507 at `D:/rsna_work/hf_cache` (revision `cdbee75f…`). No image-model checkpoints were created; existing `output/v*/` were left untouched.

**Untouched**: `data/Raw_dicom`, `data/main_data` (including the `.bak` files), `output/`, the existing `src/` and `docs/`. Modified: `.gitignore` (data-derived outputs and `__pycache__` added).
