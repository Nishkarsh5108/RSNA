# Image-model experiments

Status: **final** (2026-10-18 04:50 IST).
- **Executed:** 11 fold-0 runs (E0, E1, E1s2, E1p, E2a, E2c, E4, E6, E8, E1L, E5w), 3 final all-fold models (F1–F3), TTA evaluation, and one gold-58 scoring.
- **Not executed:** runs marked *deferred* (§4) have no results.

## 1. Common protocol (all E-runs unless stated)

| Item | Setting |
|---|---|
| Split | 5 folds over the 4,349 non-gold studies, stratified by report language × report-positive burden (`study_index.add_folds`, seed 42). The pixel-identical duplicate pair is in one fold (fold 1). PatientID is pseudonymised per study, so **no patient-level grouping is possible**. |
| Train / validate | Train on folds 1–4 (3,479 studies). Validate on fold 0 (870 studies). Gold-58 is never trained on (`--gold-weight 0`). |
| Selection metric | **Pseudo-val macro ROC-AUC**: fold 0 scored against the labels_v3 teacher binarised at 0.5. The checkpoint is the EMA weights at the epoch with the best pseudo-val AUC. |
| Gold-58 | Predictions saved **sealed** at the selected epoch of every run (`gold_preds_SEALED.npz`). They are scored **once**, after the final configuration is locked (`evaluate_sealed.py`). |
| Labels | `data/labels_v3/training_labels_v3.csv`: soft teacher = mean of 3 report-derived sources; about 0.89 macro AUC vs gold (label study) |
| Index / cache | Index v1 (`study_index_train.csv`) for all ablations; final models use index v2 (laterality fix for 27 studies; slot UIDs identical, verified). Cache v2 (224 px squash) unless stated. |
| Model | `KneeMIL`: timm backbone on 3-slice 2.5D windows (step about 3.5 mm) → 384-d projection, plus slot and slice-position embeddings → aggregation → 12 logits |
| Training | 8 epochs; AdamW (LR 1e-4 backbone, 5e-4 head; weight decay 0.05); 0.5-epoch warm-up then cosine; EMA 0.999; bf16; batch 2 studies × accumulation 2; gradient clip 1.0; GPU augmentation (affine ±10°, ±10% scale, ±6% shift; gamma and gain) per study and series; seed 42 |
| Hardware | RTX 4050 Laptop 6 GB, 12 CPU threads, Windows 11; single GPU job at a time (sequential queue `run_queue.py`) |

Uncertainty: 95% CIs from 1,000 study-level bootstrap resamples. Deltas are **paired** (same resamples for both runs). `p(Δ≤0)` is the fraction of bootstrap draws in which the run is not better than the reference.

Secondary metric, **confident-label AUC**: the same per-class AUC restricted to label cells where the teacher is confident (soft label ≤ 0.2 or ≥ 0.8; 85% of cells). It shows whether a gain holds where the pseudo-labels are reliable, rather than coming from fitting teacher noise. For E1 it is 0.923, vs 0.882 on all cells.

**Limitation (stated up front).** Pseudo-val measures agreement with *report-derived* labels, not image ground truth. A model can gain pseudo-val AUC by learning report or site correlates, and comparisons involving label transformations (E5) are biased toward the teacher. Gold-58 is the only image-read check, and it is tiny (macro-AUC SE ≈ 0.02–0.025).

### Pre-registered selection rule (written 2026-10-17 14:10 IST, before E8, E2c, E4, E6, E1L or E5w finished)

1. **Adopt a change** (resolution, aggregation, series, schedule, labels) only if its paired Δ vs E1 is **> +0.004**, above the seed-to-seed difference of 0.0026 measured by E1s2, **and** its confident-label AUC does not drop.
2. **No evidence:** if Δ is between −0.004 and +0.004, keep the simpler or cheaper option.
3. **Combined configuration.** If two or more changes are adopted, validate them together as one run on fold 0 (E9) before training final models. Interactions are not assumed to be additive.
4. **Backbone (E6).** If it is within noise of E1 but less correlated with it than a second seed is (r < 0.95), and the E1 + E6 average beats the E1 + E1s2 seed average, use it as a diverse ensemble member. Otherwise prefer seed ensembles of the chosen configuration.
5. **Final models.** Train on all 5 non-gold folds (index v2), never on gold, for the selected number of epochs, and use the last-epoch EMA weights. Ensemble size is justified by the pseudo-val pair-ensemble evidence. Gold-58 is scored once afterwards and does not change the selection.
6. *Added about 15:28 IST (E1p started 15:25), before E1p's result:* a **correctness fix** (E1p, the sagittal position input, defect D15) is adopted unless it is demonstrably harmful (Δ < −0.004). The default is the version whose inputs mean what the model is told they mean.

## 2. Results summary (pseudo-val fold 0, n = 870)

| Run | Question | Pseudo-val macro AUC [95% CI] | Δ vs E1 [95% CI] | Best epoch | Train min | Peak GB |
|---|---|---|---|---|---|---|
| E0_frozen_effb0_v2head | Reproduce v2 frozen baseline under the new protocol | 0.7923 [0.7800, 0.8043] | −0.0901 [−0.1010, −0.0787] | 7 | — | — |
| **E1_cnxtT_s3_k8_attn12** | Fine-tuned 2.5D ConvNeXt-T, 3 FS-first planes, K = 8, per-finding attention | **0.8823 [0.8733, 0.8911]** | reference | 7 | 56.6 | 3.81 |
| E1s2_cnxtT_s3_k8_attn12_seed2 | Seed replicate of E1 (seed 2) | 0.8849 [0.8766, 0.8935] | +0.0026 [−0.0009, +0.0065] | 7 | 56.9 | 3.81 |
| E2a_cnxtT_s3_k8_mean | Aggregation: masked mean pooling instead of attn12 | 0.8745 [0.8653, 0.8839] | −0.0078 [−0.0121, −0.0032] | 7 | 60.7 | 3.81 |
| E8_cnxtT_s3_k8_attn12_v6r256 | Preprocessing: 0.6 mm/px aspect-preserving, 256 px crop (v6) | 0.8799 [0.8709, 0.8892] | −0.0024 [−0.0067, +0.0015] | 7 | 75.4 | 4.73 |
| E1p_cnxtT_s3_k8_attn12_posfix | Pipeline fix D15: correct sagittal slice-position input for right knees | 0.8791 [0.8703, 0.8880] | −0.0032 [−0.0065, −0.0002] | 7 | 58.9 | 3.81 |
| E2c_cnxtT_s3_k8_tfm | Aggregation: 2-layer transformer over window tokens, then attn12 | 0.8845 [0.8754, 0.8937] | +0.0021 [−0.0014, +0.0063] | 7 | 66.3 | 3.86 |
| E4_cnxtT_s6_k8_attn12 | Series: 6 slots (FS + non-FS per plane) instead of 3 FS-first | 0.8797 [0.8711, 0.8884] | −0.0027 [−0.0064, +0.0010] | 7 | 91.1 | 3.78 |
| E6_effv2s_s3_k8_attn12 | Backbone: EfficientNetV2-S (in21k → 1k) instead of ConvNeXt-T | 0.8676 [0.8578, 0.8775] | −0.0147 [−0.0209, −0.0086] | 7 | 53.9 | 4.37 |
| E1L_cnxtT_s3_k8_attn12_ep12 | Schedule: 12 epochs (cosine) instead of 8 | 0.8836 [0.8750, 0.8923] (epoch 8; last 0.8815) | +0.0013 [−0.0022, +0.0047] | 8 of 12 | 85.1 | 3.81 |
| E5w_cnxtT_s3_k8_attn12_confw1 | Labels: soft BCE weighted per label by teacher confidence \|2y−1\| | 0.8771 [0.8685, 0.8865] | −0.0053 [−0.0085, −0.0020] | 7 | 57.0 | 3.81 |

## 3. Per-run details

### E0: frozen-feature baseline (reproduction of v2)
- **Question.** Where does the previous pipeline stand under a valid protocol?
- **Config.** v2 frozen EfficientNet-B0 ImageNet features (12 slices, 1 series per plane) with the v2 attention-pool head. Retrained on folds 1–4; epoch chosen on pseudo-val fold 0 (`src/imaging/exp0_frozen_baseline.py`).
- **Result.** Pseudo-val 0.7923. Per class: ACL 0.785, MCL 0.724, MM 0.764, LM 0.718, MOA 0.831, LOA 0.815, PFOA 0.769, Effusion 0.822, Synovitis 0.873, Baker's 0.795, Contusion 0.816, Fracture 0.795.
- **Interpretation.** It reproduces the frozen-feature regime (the earlier LB 0.775 used gold-selected checkpoints, a different protocol).

### E1: fine-tuned 2.5D ConvNeXt-T (primary experiment)
- **Question.** Does end-to-end fine-tuning of a 2.5D encoder, with the corrected pipeline, beat the frozen baseline?
- **Config.** `convnext_tiny.fb_in22k_ft_in1k`, slots 0, 1, 2 (sagittal / coronal / axial FS, non-FS fallback), K = 8 windows, `attn12` per-finding attention, soft labels, orientation normalisation on.
- **Runtime.** 56.6 min training plus about 5 min validation for 8 epochs (about 7 min/epoch; 8.4 studies/s); peak 3.81 GB; checkpoint 113 MB (EMA weights).
- **Curve (pseudo-val by epoch).** 0.762, 0.821, 0.849, 0.864, 0.874, 0.879, 0.882, 0.882. Training loss 0.563 → 0.403. Still creeping up at the end of the cosine schedule.
- **Result.** **0.8823 [0.8733, 0.8911]**; **+0.090 [0.079, 0.101] over E0**, and better in all 12 classes. Largest gains: medial meniscus +0.156, lateral meniscus +0.134, Baker's +0.123, MCL +0.112, ACL +0.105. Smallest: synovitis +0.029, lateral OA +0.045.
- **Subgroups (pseudo-val).** English 0.881, Spanish 0.892, Turkish 0.876, **Croatian 0.827 (n = 81, the weakest site group)**, Greek 0.876, German 0.881, Bulgarian 0.891, Dutch 0.871, French 0.929 (n = 19). Left 0.878 / right 0.886. Side from tag 0.873 / from geometry 0.891. **FS fallback (sagittal or coronal FS missing) 0.859, n = 83**, vs 0.884 with all three FS series present.
- **Label-noise context for the subgroups.** Teacher disagreement by report language, over all 4,349 pseudo-labelled studies:

  | Language | Mean between-source SD | Label cells in (0.2, 0.8) | Fraction of cells where all sources agree |
  |---|---|---|---|
  | Croatian | **0.144** | **0.30** | **0.48** |
  | French | 0.139 | 0.18 | 0.59 |
  | Turkish | 0.137 | 0.20 | 0.63 |
  | German | 0.131 | 0.21 | 0.63 |
  | Greek | 0.129 | 0.18 | 0.65 |
  | Dutch | 0.127 | 0.13 | 0.66 |
  | Bulgarian | 0.126 | 0.13 | 0.69 |
  | Spanish | 0.114 | 0.10 | 0.74 |
  | English | 0.101 | 0.10 | 0.67 |

  The Croatian group has the noisiest teacher. Its targets are noisier for both training and evaluation, so its lower pseudo-val AUC is at least partly label noise rather than an image-domain failure. Gold-58 is too small to separate the two.
- **Where the model looks** (`attention_map.py`, 300 val studies; `reports/eda/imaging/attention_E1_*.png`). The per-finding attention follows the anatomy, with no supervision beyond study-level labels.

  | Finding | Attention pattern | Plane share |
  |---|---|---|
  | MCL | Coronal | coronal 0.50, sagittal 0.10 |
  | Lateral meniscus | Sagittal, **lateral** half | sagittal 0.61; position 0.39, peak 0.30 |
  | Medial meniscus | Sagittal **medial** half and coronal | sagittal position 0.64, peak 0.70–0.83 |
  | Lateral vs medial OA | Sagittal lateral vs medial | positions 0.45 vs 0.57 |
  | Baker's cyst | **Posterior** coronal slices, plus axial | coronal position 0.75 |
  | Effusion | **Superior** axial slices (suprapatellar recess) | axial 0.48; position 0.67 |
  | PF OA | **Anterior** coronal slices and axial | coronal position 0.33 |
  | ACL | Central sagittal and axial | sagittal 0.43, axial 0.41 |

  The medial/lateral separation along the sagittal axis is evidence that the sagittal laterality normalisation works. A model without it would see the two compartments swapped in about half the knees. The same analysis exposed defect **D15** (mirrored slice-position *input* for right-knee sagittal series), which E1p tests.
- **Interpretation.** Fine-tuning is the main lever, as predicted by defect D9 in the pipeline audit. The weakest classes (MCL 0.836, PF OA 0.846, lateral meniscus 0.852) are the next targets.

### E1s2: seed replicate (noise floor)
- **Question.** How much do results move with the training seed alone (initialisation of new layers, data order, augmentation and window draws)?
- **Config.** E1 with `--seed 2`.
- **Runtime.** 56.9 min; 3.81 GB.
- **Curve.** 0.763, 0.819, 0.845, 0.863, 0.874, 0.881, 0.884, 0.885 (E1: 0.762, 0.821, 0.849, 0.864, 0.874, 0.879, 0.882, 0.882).
- **Result.** 0.8849 [0.8766, 0.8935]; paired Δ vs E1 **+0.0026 [−0.0009, +0.0065]** (p(Δ≤0) = 0.08). Confident-label AUC 0.928 vs 0.923.
- **Per-class seed spread is large.** Lateral meniscus 0.883 vs 0.852 (0.031); MCL 0.824 vs 0.836; fracture 0.888 vs 0.894.
- **Interpretation.** Seed-to-seed differences of **about 0.003 in macro AUC and up to about 0.03 in a single class**. Macro deltas below about 0.005, or single-class deltas below about 0.03, between single runs are not evidence of a real difference.
- **Ensemble.** E1 + E1s2 averaged: **0.8882** (+0.0033 over the better seed; prediction correlation 0.95). Seed ensembling is a cheap, reliable gain.

### E2a: mean pooling (aggregation comparison, part 1)
- **Question.** Does per-finding attention earn its complexity over plain masked mean pooling of the S × K window tokens?
- **Config.** E1 with `--agg mean`; everything else identical.
- **Runtime.** 60.7 min training (8 epochs; epoch 2 ran alongside an inference-time evaluation on the same GPU). Peak 3.81 GB.
- **Curve.** 0.750, 0.809, 0.835, 0.853, 0.864, 0.871, 0.874, 0.8745. Below E1 at every epoch.
- **Result.** 0.8745 [0.8653, 0.8839]; **paired Δ vs E1 −0.0078 [−0.0121, −0.0032]**, worse in all 1,000 bootstrap draws.
  - Largest losses: MCL −0.027, PF OA −0.018, ACL −0.013, fracture −0.011, medial meniscus −0.011.
  - Mean pooling is marginally better for synovitis (+0.008) and Baker's (+0.003).
  - FS-fallback subgroup: 0.867 vs 0.859 (n = 83).
- **Interpretation.** Localised findings (a ligament or meniscal tear visible on a few slices) benefit from attention; diffuse findings do not. This matches the MIL literature (Ilse et al. 2018) and the public 2.5D reader. The deficit (−0.008 vs E1, −0.010 vs E1s2) is about 3× the seed-to-seed difference, so the conclusion survives seed noise. Attention stays the default.
- **Ensemble check.** The E1 + E2a probability average reaches 0.8839 (+0.0016 over E1; prediction correlation 0.93). The gain is small because the models are similar.

### E8: physical-resolution preprocessing (v6: 0.6 mm/px, 256 px)
- **Question.** Does aspect-preserving resampling to a common physical scale (0.6 mm/px, centred 256 px = 153.6 mm crop or pad) beat the v2 224 px squash? Public readers report gains from 256–384 px, but their changes were confounded.
- **Config.** E1 with `--cache D:/rsna_work/cache_v6_256 --preproc v6`. Same index, series, windows and training.
- **Runtime.** 75.4 min training (1.33× E1); peak 4.73 GB (vs 3.81).
- **Curve.** 0.757, 0.812, 0.839, 0.855, 0.866, 0.875, 0.879, 0.880. Below E1 at every epoch, with the gap closing at the end.
- **Result.** 0.8799 [0.8709, 0.8892]; **paired Δ vs E1 −0.0024 [−0.0067, +0.0015]** (p(Δ≤0) = 0.88). Confident-label AUC 0.921 vs 0.923.
  - Per class, everything is within seed noise: lateral meniscus −0.020 (seed spread 0.031), contusion +0.014, fracture −0.011.
- **Ensemble.** E8 + E1s2 = 0.8882, the same as the E1 + E1s2 seed ensemble (0.8882) despite lower correlation (0.936 vs 0.952). No diversity benefit.
- **Interpretation / decision.** No evidence of a gain at 1.33× training cost, 1.24× memory and higher decode cost. **Not adopted** (pre-registered rule 2).
  - Likely reasons, measured on the 12,805 FS-slot series:
    - **Resolution.** The largest field-of-view side has quantiles (5/25/50/75/95%) of 140/160/160/170/192 mm, so v2's 224 px squash already gives 0.62–0.86 mm/px (median 0.71). v6's 0.6 mm/px is only about 15% finer at the median.
    - **Crop.** The 153.6 mm crop trims **81%** of series: about 3 mm per side at the median, 19 mm at the 95th percentile, which can cut peripheral anatomy.
    - **Aspect ratio.** Only the 7% non-square series gain geometric fidelity.
  - Higher resolution (e.g. 384 px at 0.4 mm/px) was not tested: it needs a new cache and about 3× compute.

### E1p: sagittal slice-position input fix (pipeline defect D15)
- **Question.** Does giving right-knee sagittal windows the correct anatomical position input (p instead of 1 − p) change performance?
- **Config.** E1 with `--sag-pos-fix`; same seed (42), so data order and augmentation draws match E1.
- **Runtime.** 58.9 min; 3.81 GB.
- **Curve.** 0.767, 0.822, 0.846, 0.860, 0.870, 0.876, 0.878, 0.879.
- **Result.** 0.8791 [0.8703, 0.8880]; paired Δ vs E1 **−0.0032 [−0.0065, −0.0002]**. Confident-label AUC 0.921 vs 0.923. Prediction correlation with E1 0.973, higher than between two seeds (0.952).
- **Interpretation.**
  - The fix barely changes the model: the image content, not the position input, carries the anatomy (see the E1 attention analysis).
  - The −0.003 is the size of the seed-to-seed difference. The bootstrap CI only covers validation-sample noise, not training noise, so it is not evidence of harm.
  - There is also **no evidence of benefit**.
- **Decision.** Adopted under pre-registered rule 6 (correctness fix, Δ inside the −0.004 tolerance), so that inputs mean what they claim for both knees. The report states that this rests on correctness, not on a measured gain.

### E2c: transformer + per-finding attention (aggregation comparison, part 2)
- **Question.** Does contextualising the window tokens across slices and planes (2-layer pre-norm transformer, d = 384, 8 heads, FFN 768) before the per-finding attention help, as in the public 2.5D ConvNeXt reader?
- **Config.** E1 with `--agg tfm`.
- **Runtime.** 66.3 min (1.17× E1); peak 3.86 GB.
- **Curve.** 0.760, 0.818, 0.842, 0.859, 0.873, 0.880, 0.883, 0.8845. Slower start than E1; overtakes it from epoch 6.
- **Result.** 0.8845 [0.8754, 0.8937]; **paired Δ vs E1 +0.0021 [−0.0014, +0.0063]** (p(Δ≤0) = 0.12). Confident-label AUC 0.928 (E1 0.923, E1s2 0.928).
  - Per class, within seed noise: MCL +0.009, lateral meniscus +0.021 (seed spread 0.031), synovitis +0.010, PF OA −0.008.
- **Ensemble.** E2c + E1 = 0.8884 and E2c + E1s2 = 0.8888 (correlation 0.94–0.95), vs 0.8882 for the E1 + E1s2 seed pair. A negligible diversity advantage.
- **Decision.** Δ is below the +0.004 adoption bar and about the size of the seed difference (+0.0026). **Not adopted** (rule 2: keep the simpler, 17% cheaper attn12). The transformer is not worse, so it remains a reasonable alternative.

### E4: six series per study (series-selection comparison)
- **Question.** Does adding each plane's non-fluid-sensitive series (usually PD/T1-weighted, showing anatomy and marrow) as 3 extra slots beat the minimal FS-first selection?
- **Config.** E1 with `--slots 0,1,2,3,4,5 --bs 1 --accum 4`. Same effective batch of 4 studies and the same number of optimiser steps; 96 images per step would not fit in 6 GB. FS slots no longer fall back to non-FS series, because those have their own slots.
- **Cost.**
  - 4.84 series / 165 slices decoded per study, vs 3.0 / 106 (1.56×).
  - 91.1 min training (1.61× E1) plus 4.4 min per validation (disk-bound reads of the extra series).
  - Peak 3.78 GB.
- **Curve.** 0.758, 0.823, 0.851, 0.866, 0.873, 0.877, 0.879, 0.880. Slightly ahead of E1 until epoch 3, behind from epoch 4.
- **Result.** 0.8797 [0.8711, 0.8884]; **paired Δ vs E1 −0.0027 [−0.0064, +0.0010]**. Confident-label AUC 0.923 (= E1).
  - Per class, within seed noise, with no consistent pattern: Baker's +0.011, lateral OA +0.006, medial meniscus +0.005; lateral meniscus −0.023 (seed spread 0.031), PF OA −0.016, MCL −0.011.
- **Ensemble.** E4 + E1s2 = 0.8874 (correlation 0.941), below the E1 + E1s2 seed pair (0.8882). No diversity benefit.
- **Decision.** **Not adopted.** No gain at 1.6× training cost and 1.56× decode cost. The FS series (plus the non-FS fallback when an FS series is missing) already carry the information the teacher labels reward.

### E6: EfficientNetV2-S backbone (backbone comparison)
- **Question.** Is the second strong public backbone (EfficientNetV2-S, used by a 0.923-LB reader) better than ConvNeXt-T, or complementary to it?
- **Config.** E1 with `--backbone tf_efficientnetv2_s.in21k_ft_in1k`. Same LRs, drop-path and recipe; not re-tuned for this backbone.
- **Runtime.** 53.9 min (0.95× E1); peak 4.37 GB.
- **Curve.** 0.733, 0.810, 0.838, 0.853, 0.861, 0.864, 0.867, 0.868. Below E1 at every epoch.
- **Result.** 0.8676 [0.8578, 0.8775]; **paired Δ vs E1 −0.0147 [−0.0209, −0.0086]**. Confident-label AUC 0.910 vs 0.923.
  - Worse in 11 of 12 classes; fracture −0.058, PF OA −0.027, contusion −0.023.
- **Ensemble (rule 4).** It is the least correlated model tested (r = 0.89 with E1 vs 0.95 between seeds), but E1 + E6 = 0.8845 < E1 + E1s2 = 0.8882. **Not used as an ensemble member.**
- **Interpretation.** ConvNeXt is the better encoder for this recipe. A likely factor is normalisation: EfficientNetV2 uses BatchNorm, whose batch statistics come from 48 highly correlated windows of only 2 studies per step. ConvNeXt's LayerNorm is batch-independent. A BN-frozen or GroupNorm variant, or backbone-specific LR tuning, was not tested.

### E1L: 12-epoch schedule
- **Question.** E1 was still improving when its cosine schedule reached LR 0. Does a longer schedule (12 epochs, same peak LRs) help?
- **Config.** E1 with `--epochs 12`.
- **Runtime.** 85.1 min (1.5×).
- **Curve.** 0.762, 0.816, 0.844, 0.859, 0.869, 0.876, 0.880, 0.8825, **0.8836**, 0.8829, 0.8820, 0.8815.
  - Validation BCE bottoms at epoch 7 (0.4393) and then rises (0.4440 at epoch 11) while training loss keeps falling (0.405 → 0.378). That is the onset of overfitting the teacher labels.
- **Result.** Best-epoch (epoch 8) checkpoint 0.8836 [0.8750, 0.8923]; **paired Δ vs E1 +0.0013 [−0.0022, +0.0047]**. Confident-label AUC 0.926. Last epoch 0.8815.
- **Decision.** **Not adopted** (below the +0.004 bar).
  - It also matters for the final models: they are trained on all folds without a holdout and use the **last** epoch.
  - In every 8-epoch ConvNeXt/EfficientNet run (E1, E1s2, E1p, E2a, E2c, E4, E6, E8) the best epoch *was* the last (epoch 7). With 12 epochs the last epoch is past the optimum (−0.002).
  - The 8-epoch schedule is therefore the right one for last-epoch, all-fold training.

### E5w: confidence-weighted soft labels (training-target comparison, part 1)
- **Question.** 14.7% of label cells are uncertain (0.2 < y < 0.8), mostly cases where the three report-derived sources disagree. Does down-weighting them (BCE weight |2y−1|) reduce label-noise damage?
- **Config.** E1 with `--conf-weight 1.0`. Training loss is not comparable to the other runs because it is weighted.
- **Runtime.** 57.0 min; 3.81 GB.
- **Curve.** 0.755, 0.813, 0.839, 0.854, 0.865, 0.872, 0.876, 0.877. Below E1 throughout.
- **Result.** 0.8771 [0.8685, 0.8865]; **paired Δ vs E1 −0.0053 [−0.0085, −0.0020]**. Confident-label AUC 0.922 vs 0.923, so not better even where the teacher is sure.
- **Interpretation / decision.** Uncertain teacher labels still carry ranking signal: a 0.67 cell is more often positive than a 0.33 cell. Down-weighting them removes signal rather than noise. Plain soft BCE already expresses the uncertainty through its targets. **Not adopted.**

### Selection locked: 2026-10-17 23:32 IST (when F1 started; before any gold metric)

- **Final recipe.** The E1p configuration:
  - ConvNeXt-T 2.5D MIL; slots 0, 1, 2 (FS-first with non-FS fallback); K = 8; attn12; v2 224 px;
  - laterality normalisation with the correct sagittal position input; soft labels; 8 epochs.
- **Final models.** F1, F2, F3 (seeds 42 / 7 / 11), each trained on all 4,349 non-gold studies (index v2), last-epoch EMA weights.
- **Inference.** Equal-weight average with shift TTA.
- **What drove the selection.** No single-factor change cleared the pre-registered +0.004 bar:

  | Change | Δ vs E1 |
  |---|---|
  | Mean pooling | −0.008 |
  | Transformer aggregation | +0.002 |
  | 0.6 mm/px 256 px | −0.002 |
  | 6 series | −0.003 |
  | EfficientNetV2-S | −0.015 |
  | 12 epochs | +0.001 |
  | Confidence weights | −0.005 |

  The two measured gains are both ensemble-type: seed averaging (+0.003) and shift TTA (+0.0012 / +0.0017). The position fix was adopted as a correctness fix (rule 6).
- **Gold.** Gold-58 is scored once after all queued runs finish and **cannot change this selection**.
- **Runs after this point** (E1o, E5h, E2b, E3s/c/a) are analysis only.

### Gold-58: scored ONCE (2026-10-18 04:15:43 IST, after the 23:32 lock; `GOLD_UNSEALED.json`)

The sealed predictions of every run were scored together by `evaluate_sealed.py --final F1,F2,F3`. They are the training-setting window bag, **without TTA**. Bootstrap: 2,000 study resamples, paired against E1.

| Run | Gold macro AUC [95% CI] | Δ vs E1 on gold [95% CI] | Δ vs E1 on pseudo-val |
|---|---|---|---|
| **FINAL ensemble F1 + F2 + F3** (all-fold, 3 seeds) | **0.9065 [0.878, 0.931]** | **+0.0145 [+0.002, +0.027]** | — |
| F1 / F2 / F3 (single all-fold models) | 0.900 / 0.900 / 0.898 | +0.008 / +0.008 / +0.006 (n.s.) | — |
| E1s2 (seed 2) | 0.905 [0.877, 0.931] | +0.013 [+0.002, +0.026] | +0.003 |
| E4 (6 series) | 0.902 [0.873, 0.928] | +0.010 [−0.003, +0.025] | −0.003 |
| E2c (transformer) | 0.901 [0.870, 0.928] | +0.009 [−0.007, +0.026] | +0.002 |
| **E1** (reference) | 0.892 [0.862, 0.919] | — | — |
| E1p (position fix) | 0.891 [0.861, 0.918] | −0.001 [−0.014, +0.011] | −0.003 |
| E1L (12 epochs) | 0.890 [0.862, 0.916] | −0.002 [−0.014, +0.011] | +0.001 |
| E8 (0.6 mm/px, 256 px) | 0.889 [0.859, 0.916] | −0.003 [−0.019, +0.013] | −0.002 |
| E5w (confidence weights) | 0.884 [0.855, 0.911] | −0.008 [−0.019, +0.003] | −0.005 |
| E2a (mean pooling) | 0.866 [0.831, 0.899] | −0.026 [−0.046, −0.006] | −0.008 |
| E6 (EfficientNetV2-S) | 0.857 [0.827, 0.886] | −0.035 [−0.060, −0.010] | −0.015 |
| E0 (previous frozen design) | 0.753 [0.691, 0.808] | −0.139 [−0.194, −0.087] | −0.090 |

**Reading it.**
- **Gold agrees with pseudo-val on everything pseudo-val called clearly worse:** E0, E6, E2a, and E5w directionally.
- **Gold cannot separate the top group.** The two E1 seeds alone differ by **0.013** on gold, 5× their pseudo-val difference. Gold-58 is that noisy, so the E4 and E2c point estimates above E1 are not evidence (and they cannot change the locked selection).
- **The ensemble's gain is significant on gold:** +0.015 [+0.002, +0.027] vs E1. That gain combines more data (all folds) and seed averaging.
- **Overall:** the fine-tuned pipeline moves gold from 0.753 (E0) to 0.907.

**Per class on gold** (final ensemble; number of gold positives in parentheses):

| Finding | Gold AUC | Finding | Gold AUC |
|---|---|---|---|
| ACL (24) | 0.951 | MCL (9) | 0.946 |
| Medial meniscus (26) | 0.936 | Lateral meniscus (23) | 0.850 |
| Medial OA (15) | 0.977 | Lateral OA (11) | 0.896 |
| PF OA (21) | 0.855 | Effusion (35) | 0.939 |
| **Synovitis (27)** | **0.680** | Baker's (12) | 0.967 |
| Contusion (19) | 0.961 | Fracture (18) | 0.921 |

- **Synovitis is the outlier:** 0.90 on pseudo-val but 0.68 on gold.
  - The label study found that about half of gold-positive synovitis is *not mentioned in the reports*. Every report-derived labeller has gold AUC 0.68–0.74 for synovitis.
  - The image model learned "what the report says" and inherits that blind spot. E0's synovitis is also 0.673.
  - This is a label-definition problem (image read vs report), not an image-model failure.
- **The next weakest classes** (lateral meniscus 0.850, PF OA 0.855) are also the noisiest in the label study.

### Inference-time variants (TTA, `eval_tta.py`; no retraining)

Two checkpoints, pseudo-val fold 0. Paired bootstrap vs the training setting (K = 8 evenly spaced windows).

| Variant | E1 | E1p | Δ vs training setting (E1p) [95% CI], p(Δ≤0) |
|---|---|---|---|
| K = 8 (training setting) | 0.8824 | 0.8791 | — |
| K = 12 | 0.8825 | 0.8802 | +0.0011 [−0.0010, +0.0031], 0.14 |
| K = 16 | 0.8826 | 0.8793 | +0.0002 [−0.0020, +0.0021], 0.42 |
| **K = 8 + K = 8 shifted by half a window step (2 bags averaged)** | **0.8836** | **0.8808** | **+0.0017 [+0.0001, +0.0033], 0.02** |
| K = 8 + K = 16 | 0.8836 | 0.8801 | +0.0010 [−0.0001, +0.0021], 0.04 |

- A single bigger bag does not help; averaging two shifted bags does, consistently on both checkpoints.
- It is deterministic, so only validation-sample noise applies, and the paired CI excludes 0.
- Cost: one extra prediction pass per model (about 4 min for 1,300 studies). **Adopted** (`tta_shift=True` in the inference notebook).

## 4. Queued and deferred runs (NOT executed)

**Queued:**
1. F1 (running since 23:32)
2. F2
3. F3 — final models, locked recipe, about 75–80 min each.

**Deferred** at 00:05 IST on 2026-10-18, after the owner raised concern about 10+ hours of sustained load:
- **Reason.** The GPU had run at its thermal limit (86 °C, clocks throttled 2.0 → 1.4 GHz) for over 10 h.
- **Runs.** E1o (orientation normalisation off), E5h (hard labels), E2b (max pooling), E3s / E3c / E3a (single planes).
- **Status.** Analysis only: they cannot change the locked selection. Their questions remain open in this report.
- **To run them later** (about 4.5 GPU-hours; resumable):

```
cd D:/Coding/kaggle/RSNA
PYTHONIOENCODING=utf-8 HF_HOME=D:/rsna_work/hf_cache python src/imaging/run_queue.py --queue output/imaging_v5/queue_deferred.json
python src/imaging/analyze_pseudo_val.py
```
