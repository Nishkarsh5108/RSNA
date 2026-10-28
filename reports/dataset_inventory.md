# Dataset inventory: RSNA Knee Abnormality Detection (local copy)

Generated 2026-10-17 from a full scan: one DICOM header per series for all 24,371 series, every slice header for a random 2,000-series sample, and all CSVs. No pixel data was decoded for this inventory.

Machine-readable outputs (git-ignored, under `reports/eda/`):
- `dicom/series_headers.csv`: one row per series, 38 tags
- `dicom/slice_headers_sample.csv`: 68,180 slice headers from 2,000 series
- `metadata/metadata_checks.csv`, `metadata/series_table.csv`, `metadata/vc_*.csv`, `metadata/slice_geometry_sample.csv`
- `study_inventory.csv`: one row per study (planes, sequences, vendor, field strength, PatientID, language, duplicate-report group, gold flag)
- `report_profile.csv`: language, length, section markers, duplicate hash, truncation flags
- `label_audit/*.csv`: audit of the existing label files

Scripts: `src/audit/dicom_inventory.py`, `src/audit/dicom_eda.py`, `src/audit/report_profile.py`, `src/audit/audit_existing_labels.py`, `src/audit/report_label_eda.py`, `src/audit/image_eda.py`.

## 1. Counts (verified)

| Item | Count | Note |
|---|---|---|
| Training studies (`train.csv`) | **4,407** | All unique `StudyInstanceUID`s |
| Studies with all 12 gold labels | **58** | Values strictly 0/1; no partially labelled study |
| Studies without labels | **4,349** | |
| Studies with a report | **4,407** (100%) | No null or empty report; min length 52 chars |
| Series (`train_series.csv` = on disk) | **24,371** | No series missing on disk, none extra |
| DICOM files | **819,078** | 569.2 GB; all are `.dcm`; **0 unreadable headers** |
| Series per study | median 5 (range 3–14) | 3:1, 4:675, 5:2,299, 6:698, 7:310, 8:145, 9:176, 10+:103 |
| Slices per study | median 162 (IQR 130–207, max 632) | |
| Slices per series | median 30 (sag), 30 (cor), 32 (ax) | 5–95%: sag 19–51, cor 16–38, ax 19–144. Max 320 (3D sequences) |
| Test studies (local stub) | 3 | Hidden test set about 1,300 studies, no reports |
| Unique PatientIDs | **4,407** (one per study) | IDs are pseudonymised **per study**, so repeat patients cannot be linked |

The walkthrough's counts (4,407 / 58 / 4,349) are **confirmed**.

## 2. Imaging structure

| Property | Finding |
|---|---|
| Planes | **Every study has sagittal, coronal and axial** series. CSV plane labels agree with the plane derived from `ImageOrientationPatient` for 24,371/24,371 series. 704 series are oblique (dominant-axis cosine < 0.9). |
| Sequence flags (`Fluid_Sensitive`, `Fat_Suppression`) | Always equal in this release (0/0 or 1/1): sag 5,197 non-FS / 4,667 FS; cor 3,985 / 4,624; ax 1,179 / 4,719 |
| Missing fluid-sensitive plane | 257 studies have no fluid-sensitive sagittal, 159 none coronal, 0 none axial |
| Multiple series per plane | 4,288 studies (97%) have >1 sagittal, 3,283 have >1 coronal, 1,094 have >1 axial |
| In-plane orientation | **Uniform**: axial rows +x / cols +y (5,898); coronal +x / −z (8,609); sagittal +y / −z (9,864). No per-series flipping is needed, but **medial/lateral sides are mirrored between left and right knees** in coronal and axial views. |
| Laterality | DICOM `Laterality` tag present in 2,204 / 4,407 studies. The sign of the volume centre's patient-x coordinate predicts knee side with **97.8%** agreement with the tag (n=1,001 series) and 95.4% with the side stated in the report (n=710). Tag and report agree 95.7%. **Geometry gives a complete laterality signal.** |
| Pixel spacing / FOV | Median in-plane spacing 0.31–0.33 mm; median FOV 160 mm; matrix 256–1024 (most common 512, 640, 384, 320) |
| Slice thickness | Median 3.0 mm in all planes |
| Slice geometry (2,000-series sample) | 0 duplicate positions, 0 irregular spacing (CV > 0.1), 0 gaps > 1.5× median, `InstanceNumber` monotonic with position in 100%. 225 series show more than one `ImageOrientationPatient` string (float jitter; not yet quantified as a true angle change). |
| Transfer syntax (local copy) | 100% Explicit VR Little Endian (uncompressed). The **hidden test set may contain JPEG Lossless / JPEG 2000** (official data page). The `torch_gpu` env initially had **no JPEG-Lossless decoder**. After installing pylibjpeg(-libjpeg,-openjpeg) and gdcm, transcoded competition slices decode bit-identically (`src/audit/decode_selftest.py`). |
| Vendors (studies) | Siemens 1,951 · Philips 1,301 · GE 905 · Canon/Toshiba 226 · Fujifilm 16 · Hitachi 8. Vendor name strings are inconsistent (e.g. "Philips", "Philips Healthcare", "Philips Medical Systems") and need normalising. |
| Field strength (studies) | 1.5 T 2,543 · 3 T 1,601 · 1.2 T 24 · 1.0 T 1 |
| Sex tag | M 2,076 · F 1,894 · O 199 · missing 238 |
| Intensity scale | Raw p99 differs about 25× across vendors (median middle-slice p99 from 283 to 7,272), so per-series normalisation is mandatory |

## 3. Reports

| Property | Finding |
|---|---|
| Languages (lingua; 40 terse English reports mis-detected as "Latin" re-mapped) | English 1,720 · Spanish 682 · Turkish 546 · Croatian/Bosnian 406 · Greek 321 · German 262 · Bulgarian (Cyrillic) 220 · Dutch 154 · French 96. The public write-up's counts (e.g. Turkish 831) **did not reproduce**. |
| Language vs vendor | Strongly confounded: Bulgarian = Philips only; German, Greek, Dutch = Siemens only. **Language is effectively a site proxy.** |
| Gold studies by language | English 28, Spanish 10, Turkish 6, Croatian 4, Greek 3, Bulgarian 3, Dutch 2, German 2, French 0 |
| Length | Median 977 chars; English median 1,282, French 1,361, Turkish 693, Spanish 753 |
| Old 1,200-char truncation | Would truncate **37.7%** of all reports, **50%** of gold reports, 52.8% of English and 63.5% of French reports (≈20% of English/French text lost, usually the impression) |
| Encoding artefact | **305 / 321 Greek reports spell μ with the MICRO SIGN (U+00B5)** and Δ with U+2206, giving 8,252 mixed-script words (103 after normalisation). Every previous extractor saw corrupted Greek. |
| Duplicate reports | 54 groups / 204 studies with identical text (largest 37). These are **template reports** (e.g. Turkish "all normal", Spanish "Sin anomalías"). Group members differ in vendor, series count and slice count, so there is **no evidence of duplicated studies**. One group contains a gold study. |
| Very short reports (<150 chars) | 107 (104 Spanish templates) |
| Empty findings | 4 Dutch reports end right after "Bevindingen:" with no findings. These must be treated as missing information, not negative. |
| De-identification | `[DATE]` placeholders in 21.9% of English and 99.4% of Dutch reports |

## 4. Labels and label files

### 4.1 Official labels
- `data/Raw_dicom/train.csv` is byte-identical to `data/main_data/train (1).csv` (md5 `3868da29af…`).
- 58 gold studies, binary. Prevalence: ACL 41.4%, MCL 15.5%, Med. meniscus 44.8%, Lat. meniscus 39.7%, Med. OA 25.9%, Lat. OA 19.0%, PF OA 36.2%, Effusion 60.3%, Synovitis 46.6%, Baker's 20.7%, Contusion 32.8%, Fracture 31.0%. 4.1 positives per study on average (range 1–9).
- Gold prevalence is far above report-derived prevalence for several findings (e.g. Synovitis 46.6% vs about 12%; Fracture 31% vs 6.7%; ACL 41% vs 20%). This is consistent with a **deliberately enriched, image-read gold set** [participant claim; not stated officially].

### 4.2 Existing derived label files (provenance verified from code and content)

| File | What it actually contains | Status |
|---|---|---|
| `main_data/train_folds.csv` | Gold labels for the 58 rows (NaN elsewhere); folds only for the 58 (−1 for others) | Gold copy |
| `main_data/without_labels.csv` | 58 gold reports with **hosted-LLM (Gemini Flash) hard predictions** | Genuine predictions; compliance note |
| `main_data/extracted_clinical_findings.csv` | Qwen stage-1 English summaries and JSON. Its 12 label columns are gold for the 58 rows and NaN for the rest. | Labels are a gold copy |
| `to_use/available_labels.csv` | The 58 gold rows | Gold copy |
| `to_use/train_folds_with_pseudo.csv` ("public_pseudo") | **Gold copied into the 58 rows; exactly 0.5 for all 12 targets on all 4,349 unlabeled rows** | **Contains no pseudo-labels.** It was used as a 5th "annotator" in the consensus, where 0.5 ≥ 0.5 votes **positive everywhere**. |
| `to_use/train_folds_sota_qwen25.csv(.bak)` | Qwen2.5-7B (1,200-char truncation). `.bak` has gold copied into the 58 rows; the current file has regenerated genuine predictions there. 1,744 cells exactly 0.5; 1 value outside [0,1]. | Genuine on unlabeled rows |
| `to_use/train_folds_qwen14b_awq_v2.csv(.bak)` | Qwen2.5-14B-AWQ, same scheme. 1,509 cells exactly 0.5; 2 NaN cells. | Genuine on unlabeled rows |
| `to_use/gt58_regenerated_qwen_predictions.csv` | Regenerated 7B/14B predictions for the 58 gold reports | Genuine |
| `to_use/report_labels_gpt56sol.csv` = `labels_llm_gpt56sol.csv` | Public LLM label set (most likely the public Kaggle dataset "RSNA Knee LLM Report Labels (GPT-5.6-Sol)"); values plus `__conf` | Provenance **not recorded** in the repo. 6/58 gold rows identical to gold. |
| `to_use/report_labels_v2.csv` | 5-level verdict/confidence scorer (YES/NO/UNK). **Missing exactly 1 study, which is a gold study.** | Provenance **not recorded** |
| `to_use/train_folds_consensus.csv` = `final/...` | 50/50 blend of a gold-AUC-weighted mean and CROWDLAB over 5 sources **including the all-0.5 public_pseudo** | Gold rows copied; weights fitted on the same gold rows used to report its "honest" AUC |

## 5. Splits and leakage-relevant structure
- Existing folds use `StratifiedGroupKFold` grouped by `StudyInstanceUID`. Since PatientIDs are unique per study, this is effectively study-level stratified K-fold. Patient-level grouping is **impossible** with the released identifiers.
- No gold study shares a PatientID or an image-level duplicate with a non-gold study. One template-report group contains a gold study, which is harmless for image models.
- v1/v2/v3 training notebooks select checkpoints on gold AUC within the validation fold and report that maximum as "honest". See `baseline_comparison.md`.

## 6. Unresolved uncertainties
1. The official origin of the gold labels (image read vs report) is unconfirmed by the host.
2. The provenance of `gpt56sol` and `report_v2` (authors, prompts, whether tuned on gold) is not documented.
3. Whether the hidden test set mirrors the training site/vendor mix is unknown; prevalence may differ (official warning).
4. Same-patient repeat studies cannot be ruled out, because IDs are pseudonymised per study. Image-hash near-duplicate detection was not run.
