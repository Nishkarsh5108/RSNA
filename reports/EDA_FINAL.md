# EDA (final): RSNA Knee Abnormality Detection

Reference document for model development. Every number below comes from scripts in `src/audit/` and `src/labeling/`. Supporting tables and figures are in `reports/eda/`, which is git-ignored because it is derived from competition data.

**Coverage.** Full means all 4,407 studies, 24,371 series and 819,078 DICOM headers (one header per series, all CSVs, all reports). Sample means 2,000 series with all slice headers, or 360 series × 3 decoded slices for pixel statistics. Each finding below is marked [full] or [sample].

---

## 0. Executive summary: the findings that should change what you do next

| # | Finding | Evidence | Modelling decision |
|---|---|---|---|
| 1 | **Gold labels are a small, enriched, image-read reference.** Prevalence differs strongly from report-derived prevalence: Synovitis 46.6% vs ~12%, Fracture 31% vs 6.7%, ACL 41% vs 20%. | §2 [full] | Optimise for **ranking** (AUC), not calibration to report prevalence. Don't threshold or re-weight using report prevalence. Use gold only for final, held-out evaluation. |
| 2 | **Report-derived pseudo-labels are essential, and better labellers train somewhat better image models.** Trained on the 4,349 report-labelled studies, the frozen-feature image model reaches 0.73–0.78 gold AUC; trained on the 58 gold studies alone it reaches 0.51–0.59. Labellers with higher gold agreement (public sets) beat the old Qwen labels downstream by 0.02–0.04. | `baseline_comparison.md` | Use pseudo-labels for all non-gold studies. Averaging independent labellers helped downstream (0.781 vs 0.770). |
| 3 | **Medial/lateral orientation flips with knee side, and knee side is recoverable for 100% of series from geometry** (97.8% agreement with the DICOM tag, which exists for only 50%). In-plane orientation is otherwise uniform. | §4 [full + sample] | Mirror coronal/axial images of right knees, using `sign(patient-x of volume centre)` when the tag is missing. |
| 4 | **The report text had silent corruption.** 95% of Greek reports had micro-sign artefacts. The old 1,200-character cut removed 20% of English/French text (often the impression). Dutch history-section structure broke naive parsers. | §3 [full] | Normalise Unicode. Never truncate by characters. Detect sections. |
| 5 | **One pixel-duplicated study pair carries two conflicting reports** (left vs right knee, different findings). Patient IDs are pseudonymised per study, so repeat patients cannot be linked. | §6 [full] | Put the duplicate pair in the same fold (or drop it from training). Treat study-level CV as the best grouping available. |
| 6 | **Language = site = vendor.** Bulgarian is all Philips; German, Greek and Dutch are all Siemens. Report-derived positive rates differ wildly by site (Synovitis 0–18%; OA 27–72%). | §3, §8 [full] | Expect site-specific label bias. Stratify CV by site/language. Never use language or report metadata as a model input (not available at test time anyway). |
| 7 | **Intensity scales differ ~25× across vendors.** Multiple series per plane are the norm (97% of studies have >1 sagittal). 3D sequences reach 320 slices. | §4, §5 | Per-series percentile normalisation. Choose series by sequence flags and handle multiple series per plane. Sample slices by physical coverage, not by index. |
| 8 | **Protocol-selection shortcut.** Studies with a dedicated "ACL" series are 3× more often ACL-positive (50.6% vs 17.3% within the same scanner models). | §7 [full] | Don't feed `SeriesDescription` or series counts to the model. Check that performance holds on studies without such series. |

---

## 1. Dataset structure and completeness [full]

- 4,407 studies; 58 gold; 4,349 unlabeled; 24,371 series; 819,078 files (569 GB); 0 unreadable headers; 0 missing or extra series vs `train_series.csv`; 0 UID mismatches between folders, headers and CSVs.
- **Every study has all three planes.** Series per study: median 5 (3–14). Slices per study: median 162 (67–632).
- Fluid-sensitive (fat-suppressed) coverage: missing for sagittal in 257 studies, coronal in 159, axial in 0.
- `Fluid_Sensitive` and `Fat_Suppression` are identical in this release, so they carry one bit, not two.
- **Completeness matrix**: `reports/eda/study_inventory.csv`, one row per study with planes, fluid-sensitive coverage per plane, vendor, field, sex, laterality tag, language, report length, duplicate-report group and gold flag. Flagged studies: `data/labels_v3/unresolved_studies.csv`.
- Records that need attention (nothing has been removed silently):
  - the pixel-duplicate pair `…23376711` / `…31613209`;
  - 4 Dutch reports with empty findings;
  - 107 very short template reports (104 Spanish);
  - 278 series with implausible `BodyPartExamined` (WRIST/ANKLE/HEAD/BRAIN), which looks like a header-only problem;
  - 10 series with laterality written into `ContrastBolusAgent`.

## 2. Label distribution [full for gold; full for report-derived]

| Target | Gold + / 58 | Gold prevalence | gpt56sol (report) pos-rate, unlabeled | Rules "present" rate | Rules "not mentioned" rate |
|---|---|---|---|---|---|
| ACL | 24 | 41.4% | 20.2% | 14.7% | 33.5% |
| MCL | 9 | 15.5% | 14.8% | 9.8% | 45.6% |
| Medial Meniscus | 26 | 44.8% | 39.5% | 34.1% | 22.3% |
| Lateral Meniscus | 23 | 39.7% | 15.0% | 13.2% | 24.6% |
| Medial OA | 15 | 25.9% | 36.7% | 18.4% | 72.9% |
| Lateral OA | 11 | 19.0% | 26.6% | 12.7% | 77.3% |
| PF OA | 21 | 36.2% | 45.4% | 11.7% | 78.8% |
| Effusion | 35 | 60.3% | 59.0% | 55.4% | 18.7% |
| Synovitis | 27 | 46.6% | 12.4% | 10.6% | 79.8% |
| Baker's | 12 | 20.7% | 24.7% | 22.4% | 60.1% |
| Contusion | 19 | 32.8% | 16.8% | 20.6% | 61.1% |
| Fracture | 18 | 31.0% | 6.7% | 6.0% | 80.2% |

(Rules columns are from rules-v1.1, over the 4,349 unlabeled studies; see `reports/eda/reports_labels/prevalence_gold_vs_reports.csv`.)

- Gold has 4.1 positive findings per study (range 1–9). Every gold study has at least one positive, so the gold set contains **no normal knees**. In the reports, normal templates are common.
- **Insufficient evidence classes.** MCL (9 positives), Lateral OA (11) and Baker's (12) have so few gold positives that per-class gold AUC has a 95% CI half-width of roughly ±0.1 or more. Don't make per-class decisions from gold for these.
- **Co-occurrence (gold, φ).**
  - Degenerative cluster: Medial OA–Baker's 0.48, Medial meniscus–Medial OA 0.42, Lateral meniscus–Lateral OA 0.42, Effusion–Synovitis 0.40, Medial OA–PF OA 0.37.
  - Traumatic cluster: ACL–Contusion 0.38, Contusion–Fracture 0.33.
  - Between clusters: Medial OA–Contusion −0.33, ACL–PF OA −0.27.
  - Report-derived labels reproduce the pairwise pattern only partly (correlation of φ matrices r = 0.76), with weaker magnitudes.
  - These are statistical associations in an enriched sample, not anatomical rules.
- **Report-derived prevalence by language/site** varies a lot (`gpt56sol_pos_rate_by_language.csv`):
  - Synovitis: 0.00 Bulgarian, 0.04 Croatian, 0.05 Turkish, 0.18 English.
  - Medial OA: 0.27 Bulgarian/Greek vs 0.66 Croatian.
  - ACL: 0.08 Spanish vs 0.29 Croatian.

  Part of this is case mix and part is reporting style. The two cannot be separated without image reads.
- Existing label files: one contains no information (`public_pseudo` is all 0.5 on unlabeled rows), the Qwen files have about 3–4% of cells at exactly 0.5, and **`gpt56sol` and `report_v2` agree at κ ≈ 0.87**, so they are not independent voters. See `dataset_inventory.md` §4.2 and `label_validation_report.md`.

## 3. Multilingual reports [full]

| Language | n | Gold | Median chars | Truncated at 1,200 | Impression header | History header | Notes |
|---|---|---|---|---|---|---|---|
| English | 1,720 | 28 | 1,282 | 52.8% | 73% | 30% | multi-vendor; `[DATE]` in 22% |
| Spanish | 682 | 10 | 753 | 33.4% | 99.9% | 48% | 104 short templates |
| Turkish | 546 | 6 | 693 | 10.1% | 24% | 0% | "SOL/SAĞ DİZ" states the knee side; "grade III dejenerasyon" = tear-grade signal |
| Croatian/Bosnian | 406 | 4 | 1,173 | 47.8% | 0% | 0.5% | hard line wraps mid-sentence |
| Greek | 321 | 3 | 858 | 24.3% | 24% | 2.5% | **95% micro-sign corruption** |
| German | 262 | 2 | 807 | 13.4% | 3% | 0% | compound words ("Außenmeniskushinterhorn") |
| Bulgarian | 220 | 3 | 999 | 30.9% | 99% | 0% | "МР находка … Заключение" structure |
| Dutch | 154 | 2 | 916 | 21.4% | 35% | 99% | clinical question first; findings follow mid-sentence; 4 have empty findings |
| French | 96 | 0 | 1,361 | 63.5% | 99% | 0% | longest reports |

- **Negation and uncertainty expressions** (per-language frequency tables in `reports/eda/reports_labels/top_cue_expressions_by_language.csv`). Turkish and Greek often negate *after* the noun ("…yırtık izlenmedi", "…δεν παρατηρείται"), so negation detection limited to the preceding words fails. Croatian "uredan/održan", Bulgarian "запазен" and German "regelrecht/unauffällig" are the main "normal" forms.
- **Historical cues** (reconstruction, meniscectomy, "status post") are rare: <0.5% of target mentions are judged historical by the rules.
- **Encoding.** No mojibake apart from one English report. U+00B5/U+2206 substitution affects Greek.
- **Templates and duplicates.** 54 groups / 204 studies have identical text. All are templates (different images), except the pixel-duplicate pair, whose two reports differ.
- **Information availability (Question A)** depends on what counts as a statement:
  - *Explicit mention* (rules): there is no explicit mention of Synovitis in 80% of reports, Fracture 80%, OA 73–79%, Contusion 61%, Baker's 60%, MCL 46%.
  - *Inferable status* (LLM, which also reads global statements such as "bone marrow normal" or "cartilage preserved"): the "not mentioned" rate drops to Fracture 32%, Baker's 29%, Synovitis 24%, Contusion 24%, OA 12–19%, ligaments/menisci/effusion 5–8%.
  - Spanish template reports are the most silent (20–38% "not mentioned" for every target); Dutch and Turkish follow for some targets (`llm_not_mentioned_rate_by_language.csv`).
  - On gold, "not mentioned" cells were positive 10% of the time, close to "absent" (9%). The exceptions are Synovitis (36%) and Fracture (20%).

## 4. DICOM and imaging metadata [full unless noted]

- **Vendors (studies).** Siemens 1,951, Philips 1,301, GE 905, Canon/Toshiba 226, Fujifilm 16, Hitachi 8. Normalise the inconsistent vendor strings before stratifying.
- **Field strength.** 1.5 T 2,543, 3 T 1,601, 1.2 T 24, 1.0 T 1.
- **Geometry.** Median in-plane spacing 0.31–0.33 mm; FOV about 160 mm; matrix 256–1024 (most often 512, 640, 384, 320); slice thickness 3 mm. Slices per series: median 30 sagittal, 30 coronal, 32 axial; 95th percentile 51 / 38 / 144; maximum 320 (3D acquisitions).
- **Identical array shape ≠ identical physical geometry.** A 512² matrix spans 120–200 mm FOV across sites. Resizing to a fixed pixel grid changes physical scale by up to about 1.7×. Resampling to a fixed mm/pixel (e.g. 0.6 mm, as one public reader does) is the principled option.
- **Orientation [full].** Every axial series has rows +x / cols +y, every coronal series +x / −z, every sagittal series +y / −z. No per-series flipping is needed. In coronal and axial images, a **right knee shows its lateral side on image-left and a left knee on image-right**.
- **Laterality.**
  - The DICOM tag is present for 2,204 studies.
  - **Geometry (sign of the volume-centre x coordinate) agrees with the tag on 97.8%** of 1,001 sampled series and with the report-stated side on 95.4% [sample].
  - The v3/v4 notebooks mirror only when the tag equals 'R'. That leaves untagged right knees (about 50% of series) un-mirrored, and also misses the 23 series tagged 'RIGHT'.
  - Full table for all 24,371 series: `reports/eda/metadata/laterality_by_series.csv`. Geometry vs tag agreement is 97.4% (12,000 tagged series): 98.5% when |x| ≥ 20 mm and 58% when |x| < 20 mm (714 near-midline series). At study level (majority vote) agreement is 97.4%.
  - **Rule:** use the tag when present, otherwise the study-level geometric majority, and flag studies whose series all lie within 20 mm of the midline.
- **Slice geometry [sample of 2,000 series].** No duplicate positions, no irregular spacing, no gaps, and `InstanceNumber` monotonic with position. Sorting by position projected on the slice normal (as v2/v3 do) is correct.
- **Transfer syntax.** The local copy is 100% uncompressed. The **hidden test data may include JPEG Lossless and JPEG 2000**. Before this run, the `torch_gpu` environment had **no JPEG-Lossless decoder**. After installing `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg` and `python-gdcm`, a competition slice transcoded with `gdcmconv` to JPEG-Lossless SV1 and to JPEG-2000 decodes **bit-identically** [verified]. The submission notebook must bundle these wheels and should call `src/audit/decode_selftest.py`'s `assert_decoders()`. The v2/v3 preprocessing **silently zero-fills** a plane whose slices fail to decode.

## 5. MRI image characteristics [sample: 360 series, 3 slices each, stratified by plane × sequence]

| Plane / sequence | Median p99 intensity | Foreground fraction (Otsu) | Foreground touching border | SNR proxy (fg mean / bg SD) |
|---|---|---|---|---|
| Axial non-FS / FS | 1,132 / 526 | 0.28 / 0.30 | 0.00 / 0.00 | 7.7 / 7.3 |
| Coronal non-FS / FS | 1,463 / 490 | 0.43 / 0.48 | 0.01 / 0.08 | 6.0 / 6.0 |
| Sagittal non-FS / FS | 1,369 / 667 | 0.41 / 0.44 | 0.07 / 0.03 | 5.8 / 5.3 |

- **Intensity scale is vendor- and sequence-dependent.** Middle-slice p99 medians range from 283 (Siemens) to 7,272 (Canon). Fixed windowing is not viable; per-series percentile normalisation (1–99%) is required, as v2/v3 already do.
- Anatomy routinely touches the image edge in sagittal and coronal views (the thigh and calf extend beyond the FOV). Truncation along the long axis is normal and not a quality defect.
- Typical and unusual examples (lowest SNR, most edge contact): `reports/eda/images/montage_typical_unusual.png`. The low-SNR examples are noisy fat-suppressed slices, not corrupt files.
- Not measured: motion artefacts and anatomical-coverage completeness. These need visual review or a trained quality model.

## 6. Patient and study relationships [full]

- 4,407 PatientIDs for 4,407 studies: pseudonymised per study, so **patient-level grouping is impossible** with released identifiers.
- Near-duplicate search: frozen-feature cosine similarity over all study pairs (nearest-neighbour median 0.63, 99.9th percentile 0.91), plus exact oblique-orientation signatures with pixel-hash verification over 42 candidate pairs. Result: **exactly one pixel-identical pair** (`…23376711`, `…31613209`; 144 slices, 5 series), both non-gold, with **different reports** (one says left knee, the other right knee).
- No gold study has a duplicate.
- Implications: within-patient leakage across folds cannot be ruled out but is not detectable from images at the duplicate level. Keep the duplicate pair in one fold.

## 7. Leakage audit

| Channel | Finding | Risk | Action |
|---|---|---|---|
| Study / patient IDs | IDs are random UIDs; patient IDs are per study | none detectable | none |
| DICOM metadata | `SeriesDescription` contains "ACL" in 268 series. ACL-named series go with a 3× higher ACL report-positive rate (50.6% vs 17.3%). Laterality is in `ContrastBolusAgent` for 10 series. | shortcut learning (protocol selection), not label leakage | Don't use free-text metadata as features. Report performance on studies without an ACL series. |
| File/dir names | UIDs only | none | |
| Report-derived labels | Reports are absent at test time | **critical if any report-derived value is used as an input** | Training targets only |
| Duplicate studies | 1 pixel-identical pair | small; can straddle folds | same fold |
| Preprocessing caches | v2/v3 caches are deterministic, per-series normalised, and not fitted on labels | none | |
| Target-derived features | v3 "disagreement" weights come from labels only (allowed); consensus weights were fitted on gold | gold contamination of *evaluation* | Evaluate on gold never used for fitting |
| Validation-based selection | **v1–v4 save the best epoch by gold AUC on 12–15 gold studies and report that maximum** | optimistic bias in the reported "honest" AUC | Fixed epochs, or select on pseudo-label validation |
| Existing checkpoints | `output/v1/*.pt`, `output/v3/*.pt` were trained with gold rows in their training folds (GOLD_WEIGHT=8) | any later gold evaluation of these checkpoints is contaminated | Retrain without gold for evaluation |
| Precomputed public labels | `gpt56sol`/`report_v2` authors probably tuned prompts on gold | their gold agreement is optimistic | Treat as external teachers, not as validated |

## 8. Distribution shift

| Comparison | Result | Real shift or small-sample? |
|---|---|---|
| Gold vs non-gold: vendor, language, sex | χ² p = 0.75, 0.84, 0.28 | no detectable shift |
| Gold vs non-gold: field strength | median 2.25 T vs 1.5 T, p = 0.077 | weak evidence; small n |
| Gold vs non-gold: report length | median 1,206 vs 974 chars, p = 0.03 | consistent with enrichment (more findings → longer report) |
| Gold vs non-gold: label prevalence | gold much higher for Synovitis/Fracture/ACL/Lateral meniscus | **real** (enrichment + image read vs report) |
| Language (site) vs report-derived prevalence | large differences (Synovitis 0–18%) | partly reporting style, partly case mix |
| Vendor vs intensity | 25× scale differences | real; handled by normalisation |
| Public vs private LB | host warns prevalence may differ | unknown |

## 9. Model-oriented recommendations (evidence → action → verification)

| # | Evidence | Risk / opportunity | Action | Expected benefit | Trade-offs | Verifying experiment |
|---|---|---|---|---|---|---|
| R1 | v2 frozen ImageNet features: 0.775 LB. Public 2.5D fine-tuned readers on report labels: about 0.92–0.93 LB alone [A]. | Biggest gap is the image model, not the labels | Fine-tune a 2.5D CNN (ConvNeXt-T / EffNetV2-S) end-to-end at 256–384 px with MIL pooling over slices and series | Large (≈+0.1 AUC, per public evidence) | GPU time (Kaggle T4 ×2) | 4-fold CV on pseudo-labels plus gold-held-out AUC with fixed epochs |
| R2 | Laterality recoverable for 100% of series (97.4% agreement with the tag on 12,000 tagged series; 98.5% when off-midline) | v3 mirrors only the ~50% tagged series | Mirror right knees using the geometric side | Cleaner medial/lateral features for meniscus, OA and MCL | none | Ablation: mirror-by-tag vs mirror-by-geometry on gold AUC for medial/lateral targets |
| R3 | Averaging independent labellers improved downstream (0.781 vs 0.770) | Single-teacher noise | Train on a rank/mean ensemble of independent teachers | Small but consistent | Correlated teachers add little | Downstream protocol in `baseline_comparison.md` |
| R4 | Silence is common (explicit-mention rate 20–40% for Synovitis/Fracture/OA), but on gold "not mentioned" ≈ "absent" (10% vs 9% positive), except Synovitis and Fracture | Masking silent cells throws away mostly-correct negatives | **Keep silent cells as low soft scores (don't mask).** Masking lowered downstream AUC by 0.010–0.018 in all three tests. A per-target exception (a slightly higher prior for Synovitis/Fracture) is an untested option. | Avoids a measured loss | — | Done: masked vs unmasked (`baseline_comparison.md` §4) |
| R5 | Soft labels exist with calibrated letter probabilities | Hard thresholds discard uncertainty | Use soft BCE targets (already supported) | Smooths noise | — | Soft vs hard downstream comparison |
| R6 | Gold is enriched, image-read and tiny (58); per-class SE about 0.06–0.1 | Over-steering on gold noise | Use gold only as the final held-out check (all 58, never trained on). Select on pseudo-label CV. | Honest model selection | Can't fine-tune on gold | Track gold AUC with bootstrap CIs; ignore deltas <0.03 |
| R7 | Class imbalance in report labels (Fracture 6.7%, Synovitis 12%) | Rare heads under-trained | `pos_weight` clipped (already in v2/v3) or focal loss for rare heads | Moderate | Calibration shifts (irrelevant for AUC) | Per-class downstream AUC with and without |
| R8 | 97% of studies have multiple sagittal series; fluid-sensitive is missing in 257 sagittal / 159 coronal | Single-series selection discards information or picks inconsistent sequences | Fixed slots per (plane, FS flag) with a mask for missing series (v3 bag pooling is a good start) | Robustness | Memory | Ablation over slot set |
| R9 | 3D series up to 320 slices; FOV 120–200 mm | Index-based sampling covers different anatomy | Sample slices by physical position; resample in-plane to fixed mm/pixel | Consistent anatomy | Preprocessing time | Compare gold AUC with fixed-mm vs fixed-px |
| R10 | Intensity scale ×25 across vendors | Domain shift | Per-series percentile normalisation plus intensity augmentation (gamma, contrast) | Robustness across vendors | — | Per-vendor gold/CV AUC breakdown |
| R11 | Protocol-selection shortcut (ACL series) | Shortcut learning | Don't use series descriptions or counts as features | Generalisation | — | AUC on studies without ACL-named series |
| R12 | Hidden test may include JPEG-Lossless/J2K DICOMs; v2/v3 zero-fill planes that fail to decode | **Silent degradation** of compressed test studies (hidden-run logs are invisible) | Bundle decoder wheels; call `assert_decoders()` so a missing decoder fails loudly | Avoids silently blank planes | — | `src/audit/decode_selftest.py` (verified locally on transcoded slices) |
| R13 | Patient grouping impossible; 1 duplicate pair | Mild optimism in CV | Study-level stratified folds; duplicate pair in one fold | Honest CV | — | — |
| R14 | Site-specific labelling style | Labels encode site priors | Stratify folds by language/site; monitor per-site CV | Detects site overfit | — | Per-language CV AUC |

Metric-specific notes: the competition metric is macro AUC, which is threshold-free and prevalence-invariant, so calibration and class thresholds don't matter for the score. Focal loss and class weights matter only through their effect on ranking.
