# RSNA Knee MRI: final image-model report

Status: **final** (2026-10-18 04:45 IST). Every number is measured. Six analysis-only runs were deferred and are marked "not executed". Model selection was locked at 2026-10-17 23:32 IST; gold-58 was scored once at 2026-10-18 04:15 IST. Companion reports:
- `image_model_literature_review.md`
- `image_pipeline_audit.md`
- `image_model_experiments.md` (every run, in detail)
- `repo_remediation_plan.md`

## 1. Executive summary

- **Final model.** An ensemble of 3 fine-tuned **2.5D ConvNeXt-Tiny multi-slot MIL** models, one per seed, each trained on all 4,349 non-gold studies.
  - Inputs: the sagittal, coronal and axial fluid-sensitive series; 8 three-slice windows each (about 3.5 mm apart); 224 px; laterality-normalised.
  - Head: per-finding attention pooling.
  - Inference: shift TTA.
- **Gold-58, scored once after the selection was locked: 0.9065 [0.878, 0.931] macro AUC.**
  - The previous frozen-feature design (E0) scores 0.753 [0.691, 0.808] on the same 58 studies.
  - On held-out pseudo-labelled studies, the recipe scores 0.879–0.885 per single model and 0.888 for 2 seeds averaged, vs 0.792 for E0.
  - **No leaderboard score exists:** no submission was made from this machine (no Kaggle credentials).
- **Fine-tuning was the decisive change** (+0.090 pseudo-val, +0.139 gold vs the frozen design).
  - Seven single-factor changes were tested against a pre-registered rule. None beat the base recipe beyond seed noise (0.003): transformer aggregation, 6 series, 0.6 mm/px 256 px, 12 epochs, EfficientNetV2-S, mean pooling, confidence-weighted labels.
  - The reliable gains were variance reduction: seed ensembles and window-shift TTA.
- **Pipeline defects fixed**, several of which would have hurt silently. Details in §4 and `image_pipeline_audit.md`.
  - **Laterality.** Right-knee sagittal slice order was not normalised, which attention maps confirmed now works. The slice-position input was mirrored for right knees, found by the same attention analysis. The side policy is tag-first, then coronal/axial geometry.
  - **Silent decode skipping.** Two truncated training DICOMs are now handled by an explicit, reported end-slice policy.
  - **Kaggle runtime.** The Kaggle image is Python 3.13 and had no matching decoder wheels: compressed hidden-test DICOMs would have aborted the run. Fixed with wheels for 3.10–3.14 and install-before-import.
  - **T4 precision.** fp16 on pre-Ampere GPUs, verified against fp32.
- **Feasibility.**
  - All training ran **locally** (RTX 4050 6 GB): about 57 min per 8-epoch fold run, about 85 min per all-fold run, thermally throttled. Kaggle training was not required.
  - Inference of the final 3-model, TTA package measured **1.9 s/study → about 42 min for 1,300 studies** (9 h limit).
- **Biggest remaining weakness: synovitis** (gold 0.680). About half of gold-positive synovitis is absent from the reports, so report-derived labels cannot teach it. Every label source shares this ceiling.
- **Owner action needed.** The GitHub repository is **public** and contains competition data from earlier commits. A remediation plan is prepared (`repo_remediation_plan.md`) but not executed.

## 2. Literature findings (details: `image_model_literature_review.md`)

- **Strong public readers for this competition** (0.92–0.94 LB) share a recipe: a fine-tuned 2D/2.5D CNN encoder (ConvNeXt, EfficientNetV2, CoAtNet) on 3-slice windows from fluid-sensitive-first series, multi-instance pooling with per-finding attention, 256–384 px input, EMA weights, and report-derived soft labels. Reported numbers are participant claims [A], not reproduced.
- **Knee-MRI literature.** MRNet (Bien et al. 2018) established slice-encoder + pooling over series with report-derived labels, and showed a large cross-site drop (ACL 0.965 → 0.824). Attention MIL (Ilse et al. 2018) suits study-level labels with findings on few slices.
- **Frozen foundation features** (DINOv2, the user's v4) did not beat fine-tuned CNNs in any source found. 3D CNNs were not pursued: few pretrained 3D weights, anisotropic 3–4 mm slices, and the 6 GB VRAM limit.

## 3. Model candidates considered

| Candidate | Status | Outcome |
|---|---|---|
| Frozen EfficientNet-B0 features + attention head (previous v2) | E0, reproduced under the new protocol | 0.792 pseudo-val |
| **Fine-tuned 2.5D ConvNeXt-T, 3 FS-first slots, per-finding attention** | E1 (+ seed replicate E1s2) | 0.882 / 0.885; seed spread 0.003 |
| Aggregation: mean / max / transformer + attention | Mean −0.008 [−0.012, −0.003]; transformer +0.002 [−0.001, +0.006] (noise, 1.17× cost); max: **not executed** (deferred) | **attn12 kept** |
| Sagittal position-input fix (defect D15) | E1p done: −0.003 [−0.007, −0.000] (same seed; within seed noise) | **adopted** as a correctness fix (pre-registered rule 6) |
| Inference: two shifted window bags (TTA) | done on E1 and E1p: +0.0012 / +0.0017 [+0.0001, +0.0033] | **adopted** |
| Planes: sagittal / coronal / axial alone | E3s / E3c / E3a: **not executed** (deferred; see experiments report §4). Plane usage was measured instead with attention maps. | — |
| Series: 6 slots (FS + non-FS) | E4: −0.003 [−0.006, +0.001] at 1.6× training and 1.56× decode cost | not adopted |
| Labels: hard / confidence-weighted soft | E5w: −0.005 [−0.009, −0.002] (confident-label AUC not better); hard labels (E5h): **not executed** (deferred) | soft BCE kept |
| Backbone: EfficientNetV2-S | E6: −0.015 [−0.021, −0.009]; diverse (r = 0.89) but E1 + E6 = 0.8845 < E1 + E1s2 = 0.8882 | not adopted, not an ensemble member |
| Resolution: 0.6 mm/px, 256 px (v6) | E8 done: −0.002 [−0.007, +0.002] at 1.33× training cost | not adopted |
| Schedule: 12 epochs | E1L: +0.001 [−0.002, +0.005] at best epoch 8; overfits after epoch 7 (last epoch 0.8815) | not adopted; 8 epochs = best-is-last, needed for all-fold training |
| Orientation normalisation off | E1o: **not executed** (deferred). Orientation evidence: mean-image correlation and attention maps (§4, experiments report). | — |
| CoAtNet, RadImageNet, DINOv2 fine-tuning, 3D CNNs | not run (time; see literature review) | — |

## 4. Image-pipeline corrections (details and verification: `image_pipeline_audit.md`)

1. **Laterality.**
   - Study side: majority DICOM tag, else majority geometry of coronal/axial series (≥ 20 mm off-midline). Tag/geometry conflicts fell 57 → 28.
   - Right knees: coronal/axial columns mirrored and **sagittal slice order reversed** (lateral → medial for every knee).
   - Mean-image correlation, left vs right knees, improves after normalisation: coronal 0.869 → 0.988; sagittal 0.950 → 0.988.
2. **Decoding.**
   - Fail-loud: a failed series falls back to the next candidate or is masked; > 2% of studies failing aborts.
   - One undecodable end slice is dropped and **reported**. Two such truncated files exist in training.
   - Missing decoders abort.
3. **Train/test preprocessing parity.** Bit-exact on 51 local series (v2). v6 is produced by the same function in training and inference.
4. **Series slots.** Plane × {FS, non-FS}; the series with most slices wins; empty FS slots fall back to the plane's non-FS series. Missing slots are masked, never zero images.
5. **Physical 2.5D step.** About 3.5 mm between window slices, from SpacingBetweenSlices / SliceThickness.
6. **Validation hygiene.**
   - Pixel-identical duplicate pair kept in one fold.
   - Gold-58 sealed until the selection is locked.
   - Checkpoint selection on pseudo-val only.
7. **Kaggle runtime** (Python 3.13 image):
   - offline wheels for cp310–cp314;
   - dependencies installed before pydicom is imported;
   - `fork` process pools;
   - fp16 on pre-Ampere GPUs (precision checked).

## 5. Final architecture — LOCKED 2026-10-17 23:32 IST (before any gold metric; see experiments report)

```
per study (side from DICOM tag, else coronal/axial geometry)
 └─ 3 series slots: sagittal / coronal / axial fluid-sensitive (most slices; non-FS series of that plane if no FS)
     └─ v2 preprocessing: slice order along the normal, per-series 1–99 percentile → uint8 → 224 × 224
     └─ laterality: right knee → coronal/axial mirrored, sagittal slice order reversed (lateral → medial)
     └─ K = 8 windows over 4–96% of the slab, each = 3 slices [c−s, c, c+s], s ≈ 3.5 mm / spacing
 └─ ConvNeXt-Tiny (ImageNet-22k → 1k, fine-tuned) on each 3-slice window → 768-d
 └─ Dropout → Linear 384 → LayerNorm, + slot embedding + slice-position MLP (correct anatomical position for both knees)
 └─ per-finding attention pooling (12 queries over the 24 window tokens) → 12 logits
inference: 2 window bags (training setting + shifted half a step), sigmoid, mean over bags and over the 3 seeds
```

- **Training recipe** (E1p configuration, validated on fold 0: pseudo-val 0.879; recipe family 0.879–0.885 over seeds and variants): see §6.
- **Final models.** F1, F2, F3: seeds 42, 7, 11, trained on **all 4,349 non-gold studies** (index v2), 8 epochs, last-epoch EMA weights. Equal-weight average with shift TTA.
- **Evidence for the ensemble.** A 2-seed average gains +0.003 on pseudo-val; shift TTA gains +0.0012 to +0.0017.
- **What was tested and not adopted, with reasons** (`image_model_experiments.md`):
  - mean pooling (−0.008);
  - 0.6 mm/px 256 px (−0.002 at 1.33× cost);
  - transformer aggregation (+0.002, noise, 1.17× cost);
  - six series (−0.003 at 1.6× cost).

## 6. Training methodology (common to all runs)

| Item | Setting |
|---|---|
| Labels | `labels_v3` teacher (soft, mean of 3 report-derived sources), about 0.89 macro AUC vs gold in the label study |
| Optimiser | AdamW: backbone LR 1e-4, head LR 5e-4, weight decay 0.05; 0.5-epoch warm-up then cosine to 0 |
| Batch | 2 studies × gradient accumulation 2 (effective 4 studies). Six-slot runs: 1 × 4. |
| Precision | bf16 autocast (RTX 4050); fp16 + GradScaler on pre-Ampere GPUs (`--amp auto`) |
| Regularisation | EMA 0.999 (timm ModelEmaV3; EMA weights evaluated and shipped); drop-path 0.1; dropout 0.2 in the projection; gradient clip 1.0 |
| Augmentation (GPU, per study and series) | Affine (±10° rotation, ±10% scale, ±6% shift), gamma ±20%, gain ±10%. **No left–right flips**: they would undo laterality normalisation (MCL has no lateral counterpart). |
| Sampling | Training: one random window centre per bin (K bins over 4–96% of the slab). Evaluation: evenly spaced centres. |
| Restartability | `ckpt_last.pt` every epoch (model, optimiser, scheduler, EMA, scaler, history); `--resume`; `--time-limit-min` for Kaggle sessions; Windows sleep prevented during training |

## 7. Validation design

- **Folds.** 5 folds over the 4,349 non-gold studies, stratified by report language × report-positive burden. PatientID is pseudonymised per study, so **patient-level grouping is impossible** and was not claimed. The pixel-identical duplicate pair shares a fold.
- **Selection.** Pseudo-labelled fold 0 (870 studies; teacher binarised at 0.5). Paired bootstrap deltas; secondary confident-label AUC (teacher ≤ 0.2 or ≥ 0.8).
- **Gold-58.** Never trained on. Predictions saved sealed per run, scored **once** after the selection is locked (`evaluate_sealed.py` writes a lock file before computing any metric, and refuses a second run).
- **Limitations.**
  - Pseudo-val measures agreement with report-derived labels, not image truth.
  - Gold-58 is tiny (macro-AUC SE ≈ 0.02–0.025) and probably enriched; it cannot separate candidates that differ by less than about 0.03.
  - The hidden test's site and prevalence mix is unknown.

## 8. Experimental comparisons (pseudo-val fold 0, n = 870; details in `image_model_experiments.md`)

All runs share the split, labels, evaluation code and (unless stated) recipe. Δ is paired vs E1. Seed-to-seed difference (E1 vs E1s2): 0.0026.

| Run | Change vs E1 | Macro AUC [95% CI] | Δ vs E1 [95% CI] | Confident-label AUC | Train min | Decision |
|---|---|---|---|---|---|---|
| E0 | previous frozen-feature v2 design | 0.7923 [0.780, 0.804] | −0.090 [−0.101, −0.079] | 0.832 | — | baseline |
| **E1** | — (fine-tuned ConvNeXt-T, S3, K8, attn12) | **0.8823** [0.873, 0.891] | — | 0.923 | 56.6 | reference |
| E1s2 | seed 2 | 0.8849 [0.877, 0.894] | +0.003 [−0.001, +0.007] | 0.928 | 56.9 | noise floor |
| E1p | sagittal position-input fix | 0.8791 [0.870, 0.888] | −0.003 [−0.007, −0.000] | 0.921 | 58.9 | adopted (correctness) |
| E2a | mean pooling | 0.8745 [0.865, 0.884] | −0.008 [−0.012, −0.003] | 0.916 | 60.7 | rejected |
| E2c | transformer + attn12 | 0.8845 [0.875, 0.894] | +0.002 [−0.001, +0.006] | 0.928 | 66.3 | rejected (noise, cost) |
| E4 | 6 series (FS + non-FS) | 0.8797 [0.871, 0.888] | −0.003 [−0.006, +0.001] | 0.923 | 91.1 | rejected |
| E6 | EfficientNetV2-S | 0.8676 [0.858, 0.878] | −0.015 [−0.021, −0.009] | 0.910 | 53.9 | rejected |
| E8 | 0.6 mm/px, 256 px | 0.8799 [0.871, 0.889] | −0.002 [−0.007, +0.002] | 0.921 | 75.4 | rejected |
| E1L | 12 epochs | 0.8836 [0.875, 0.892] | +0.001 [−0.002, +0.005] | 0.926 | 85.1 | rejected |
| E5w | confidence-weighted labels | 0.8771 [0.869, 0.887] | −0.005 [−0.009, −0.002] | 0.922 | 57.0 | rejected |
| E1 + E1s2 average | 2-seed ensemble | 0.8882 | +0.006 vs E1, +0.003 vs E1s2 | — | — | ensemble adopted |
| E1p + shift TTA | 2 window bags | 0.8808 | +0.0017 [+0.0001, +0.0033] vs E1p | — | — | TTA adopted |

**Reading the table.**
- Around E1 the recipe is on a plateau: seven single-factor changes moved macro AUC by −0.015 to +0.002, and only mean pooling, EfficientNetV2-S and confidence weighting were clearly worse.
- What reliably helped is variance reduction: seed averaging and TTA.
- This agrees with the label study: with report-derived teacher labels at about 0.89 AUC vs gold, the student is near what the labels can teach. Confident-label AUC is 0.92–0.93 against 0.88 on all cells.
- Analysis-only runs E1o (orientation off), E5h (hard labels), E2b (max pooling) and E3s/c/a (single planes) ran after the lock. Their results are in `image_model_experiments.md`.

## 9. Per-class results

**Pseudo-val** (fold 0, n = 870, single fold-held-out models of the final recipe family) and **gold-58** (final ensemble, scored once). Seed-to-seed spread per class on pseudo-val is up to 0.03 (lateral meniscus).

| Finding (gold positives) | Pseudo-val E1p (final config) | Pseudo-val E1 / E1s2 | Pseudo-val E0 (old) | **Gold: final ensemble** | Gold: E0 (old) |
|---|---|---|---|---|---|
| ACL (24) | 0.880 | 0.890 / 0.895 | 0.785 | **0.951** | 0.695 |
| MCL (9) | 0.837 | 0.836 / 0.824 | 0.724 | **0.946** | 0.594 |
| Medial meniscus (26) | 0.914 | 0.920 / 0.922 | 0.764 | **0.936** | 0.653 |
| Lateral meniscus (23) | 0.844 | 0.852 / 0.883 | 0.718 | **0.850** | 0.783 |
| Medial OA (15) | 0.902 | 0.905 / 0.900 | 0.831 | **0.977** | 0.902 |
| Lateral OA (11) | 0.861 | 0.860 / 0.866 | 0.815 | **0.896** | 0.799 |
| PF OA (21) | 0.841 | 0.846 / 0.851 | 0.769 | **0.855** | 0.784 |
| Effusion (35) | 0.875 | 0.875 / 0.869 | 0.822 | **0.939** | 0.845 |
| Synovitis (27) | 0.898 | 0.902 / 0.901 | 0.873 | **0.680** | 0.673 |
| Baker's (12) | 0.922 | 0.918 / 0.924 | 0.795 | **0.967** | 0.846 |
| Contusion (19) | 0.891 | 0.889 / 0.898 | 0.816 | **0.961** | 0.771 |
| Fracture (18) | 0.884 | 0.894 / 0.888 | 0.795 | **0.921** | 0.696 |
| **Macro** | 0.879 | 0.882 / 0.885 | 0.792 | **0.9065 [0.878, 0.931]** | 0.753 |

- **Gold is higher than pseudo-val for most classes.** Gold is image-read and enriched: every gold study has at least one positive, so positives are clearer. Pseudo-val uses noisy report-derived targets, and confident-label AUC is 0.92–0.93 against 0.88 overall.
- **Synovitis is the exception** (0.90 pseudo-val vs 0.68 gold). Reports omit about half of the gold-positive synovitis, and the model learned what reports say.
- **Lateral meniscus and PF OA** are the weakest classes on both metrics.
- **Prediction distributions** per class: `reports/eda/imaging/pred_distributions_*.png`.


## 10. Generalisation limitations

- **Site / language.** On pseudo-val, Croatian studies score lowest (E1 0.827, n = 81). Croatian reports also have the noisiest teacher (30% uncertain label cells vs 10% for English), so the measured gap mixes label noise with any image-domain effect.
- **Missing FS series.** Studies without a sagittal or coronal FS series score lower (E1 0.859 vs 0.884; n = 83). These are the FS-fallback cases.
- **Laterality.** 70 untagged near-midline studies rely on weak geometry and may be mis-oriented.
- **Label definitions vs image reads.** Synovitis (gold 0.68), the effusion size threshold and unreported fractures are report/image discordances (label study). A model trained on report-derived labels inherits them. The hidden test is presumably image-read like gold, so expect the per-class pattern above rather than the pseudo-val one.
- **Gold-58 is small and enriched.** CI ±0.03, and seeds alone differ by 0.013 on it. Leaderboard behaviour (about 1,300 studies, unknown prevalence; the host warns prevalence may differ) is the real test.
- **Site shift.** Pseudo-val and gold come from the training distribution. If the hidden test contains other sites or vendors, the MRNet experience (ACL 0.965 → 0.824 external) suggests drops are possible. Per-series percentile normalisation and intensity augmentation are the only mitigations implemented.

## 11. Runtime and hardware analysis

| Item | Measured |
|---|---|
| Local training (ConvNeXt-T, S = 3, K = 8, 224 px) | about 7 min/epoch (8.4 studies/s), 56.6 min for 8 epochs, peak 3.81 GB, checkpoint 113 MB |
| Validation (870 studies) | 0.6 min |
| Sustained thermal state (RTX 4050 Laptop, after 5 h of training) | 80 °C, SM clock 2.0 GHz of 3.1 GHz max, about 55 W, throttle reason "SW thermal slowdown". Throughput was stable across runs (about 8.4 studies/s for S = 3, 224 px), so all timings above are at this thermally limited steady state. |
| v6 cache build (13,221 series, 2 CPU processes) | 118 min (about 1.1 s per series per process; uncompressed DICOM) |
| Decode cost per study | 3 slots: 3.0 series / 106 slices; 6 slots: 4.8 series / 165 slices |
| Pixel decode, 512 × 512 16-bit slice (pydicom 3.0.2 + pylibjpeg) | uncompressed 1.3 ms; RLE 8.2 ms; **JPEG-2000 lossless 17.0 ms** (bit-identical round trip); JPEG-Lossless could not be timed (pydicom has no encoder) but uses the libjpeg decoder |
| **Measured end to end** (`bench_inference.py`: 50 fold-0 training studies posed as a test set; 4 decode workers; E1 checkpoint; a training job running concurrently on the same machine) | 97 s total = 87.5 s header index + decode + 9.7 s prediction → **1.95 s/study → about 42 min for 1,300 studies** with one model. Sanity check: AUC vs teacher on these 50 studies 0.898. |
| **Final package measured** (`bench_inference.py`, 3 checkpoints × 2 TTA bags, 4 decode workers, 50 studies, machine otherwise idle) | 95.7 s = 76.3 s index + decode + 19.4 s prediction → **1.91 s/study → about 42 min for 1,300 studies**. The in-sample AUC on these 50 studies (0.951) is only a sanity check: they were in the final models' training data. |
| Final model training (all folds, 4,349 studies, 8 epochs, thermally throttled GPU) | F1 84.0 / F2 85.7 / F3 85.9 min training + about 6.8 min of (in-sample) validation each; peak 3.81 GB; checkpoint 113.6 MB |
| Hidden-test estimate (1,300 studies, 4 CPUs, 3 slots) | Series preprocessing ≈ 1,300 × 3 × 1.1 s / 4 ≈ 18 min (uncompressed); worst-case extra J2K decode ≈ 1,300 × 106 × 17 ms / 4 ≈ 10 min. Model: about 0.9 min per model per window bag (RTX 4050) → **< 1 h end to end** vs the 9 h limit. |

## 12. Competition-rule compliance

- **Inputs at test time.** DICOM pixels plus the provided `test_series.csv` (plane, FS flag). **No report text, no labels, no hidden-test tuning.** Reports are used only to derive training labels, which the Data tab permits.
- **External resources.** ImageNet-pretrained timm backbones: ConvNeXt-T `fb_in22k_ft_in1k` and EfficientNetV2-S `in21k_ft_in1k`, both public. The rules allow freely available pretrained models. No external medical data.
- **Offline.** Internet off at submission. Weights and wheels come from an attached **private** dataset.
- **Runtime.** ≤ 9 h budget; the estimate is in §11.
- **Data security.**
  - No competition data or derived artefacts were uploaded anywhere.
  - All derived outputs (`data/labels_v3/`, `data/imaging_v5/`, `reports/eda/`, `output/`) are git-ignored.
  - **The GitHub repository is public and tracks competition data from earlier commits.** The remediation plan (`repo_remediation_plan.md`) is prepared but **not executed**, because it needs the owner's decision.
- **Winning-code licence (CC-BY-NC)** and the 2 final submissions are the owner's decisions.

## 13. Final inference pipeline

`notebooks/rsna_knee_image_model_inference.ipynb` is generated from `src/imaging/*.py` by `notebooks/build_image_notebooks.py`, so it runs exactly the tested code. Steps:
1. **Offline dependencies.** Install the wheels before pydicom is first imported, then assert decoder availability (JPEG-Lossless SV1 and process 14, JPEG-2000 lossless and lossy).
2. **Header-only study index.** Slots, side, 2.5D step; same rules as training.
3. **Decode the selected series** for each preprocessing version the attached checkpoints need, applying the failure policy above.
4. **Predict** with every `knee_*.pt` checkpoint: equal-weight probability average; window-shift TTA **on by default** (`TTA_SHIFT = True`).
5. **Write and validate** `submission.csv`: columns, row count, IDs, finite values in [0, 1]. Also write `inference_report.json` (failures, dropped slices, side sources, timings).

Tests executed locally (all pass):
- **Inference tests 1–6** (`src/imaging/test_inference_local.py`):
  1. normal run;
  2. truncated FS series → fallback;
  3. missing plane → masked;
  4. missing decoders → abort;
  5. mixed v2 + v6 ensemble = mean of the single models;
  6. truncated end slice → dropped and reported; truncated middle slice → series rejected.
- **End-to-end notebook tests** (`notebooks/test_image_notebooks_local.py`, each notebook in a fresh interpreter): v2; v2 with TTA; v6. Plus the **release test**: the inference notebook run on the exact upload folder (F1–F3, TTA on), giving a valid `submission.csv`.

**How to run it on Kaggle (step by step).**
1. **Upload the weights.** Kaggle → Datasets → New Dataset: upload the folder `D:/rsna_work/kaggle_upload/rsna-knee-image-model` (3 `knee_F*.pt`, `wheels/`, `README.md`). Set visibility to **Private**.
2. **Import the notebook.** On the competition page → Code → New Notebook → File → Import Notebook → `notebooks/rsna_knee_image_model_inference.ipynb`.
3. **Attach inputs.** The competition data (attached automatically from the competition page) plus the private dataset from step 1.
   - Attach **only one** dataset that contains `knee_*.pt` files: the notebook ensembles every checkpoint in the first such folder it finds.
   - Alternatively, set `MODEL_DIR` explicitly in the CONFIG cell.
4. **Settings.** Accelerator **GPU T4**; **Internet off**.
5. **Dry run.** *Run All* runs on the 3 example test studies. Check that the dependency cell prints all four decoders `True`, and that the last cell prints `submission OK`.
6. **Submit.** *Save Version* (Save & Run All), then submit that version to the competition. The scoring run processes the hidden test (about 1,300 studies; expected about 1 h on Kaggle; measured 42 min locally).
7. **Outputs.** `/kaggle/working/submission.csv` and `/kaggle/working/inference_report.json` (decode failures, dropped end slices, side sources, timings). Check the report after the first scoring run.

## 14. Artifact index (everything below is git-ignored or local; derived from competition data, so keep it private)

| What | Where |
|---|---|
| **Kaggle upload folder** (make it a PRIVATE Kaggle dataset) | `D:/rsna_work/kaggle_upload/rsna-knee-image-model/`: `knee_F1_*.pt`, `knee_F2_*.pt`, `knee_F3_*.pt` (113 MB each, EMA weights + args) and `wheels/` (pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0 and pylibjpeg-openjpeg 2.6.0 for CPython 3.10–3.14, pydicom 3.0.2, timm 1.0.30) |
| Inference notebook (submission) | `notebooks/rsna_knee_image_model_inference.ipynb` |
| Training notebook (Kaggle-capable, resumable) | `notebooks/rsna_knee_image_model_training.ipynb` |
| Notebook generator and local notebook test | `notebooks/build_image_notebooks.py`, `notebooks/test_image_notebooks_local.py` |
| Source (tested modules) | `src/imaging/`: `study_index.py`, `infer_lib.py`, `data.py`, `model.py`, `train.py`, `predict.py`, `run_queue.py`, analysis scripts |
| Full archive (weights, configs, metrics, predictions, code, reports) | `output/imaging_v5/rsna_model_artifacts.zip` |
| Light archive (no weights) | `output/imaging_v5/rsna_artifacts_light.zip` |
| Structured artefact folder | `output/imaging_v5/rsna_artifacts/` (`models/`, `experiments/<run>/`, `analysis/`, `code/`, `MODEL_CARD.md`, `README.md`) |
| Per-run outputs | `output/imaging_v5/runs/<run>/`: config, history, summary, `val_preds_best.csv`, `gold_preds_SEALED.npz`, `ckpt_best.pt`, `ckpt_last.pt` |
| Run logs, queue | `output/imaging_v5/logs/`, `output/imaging_v5/queue.json`, `queue_deferred.json` |
| Analysis tables and figures | `reports/eda/imaging/`: pseudo-val results, per-class, subgroups, correlations, pair ensembles, TTA, attention maps, prediction distributions, gold results |
| Caches | v2 224 px: `output/v2/preprocessed/` (legacy, verified); v6 256 px: `D:/rsna_work/cache_v6_256/` (47.7 GB, 21,334 series = 13,221 FS-first + 8,113 further non-FS; not used by the final model, can be deleted to free space) |
| Study index (folds, sides, slots) | `data/imaging_v5/study_index_train_v2.csv` (final), `study_index_train.csv` (ablations) |

## 15. Instructions for local analysis

- Per-run outputs in `output/imaging_v5/runs/<run>/`:
  - `config.json`, `history.csv` (per-epoch loss, AUC and per-class AUC), `summary.json`;
  - `val_preds_best.csv` (fold-0 probabilities per StudyInstanceUID);
  - `gold_preds_SEALED.npz`.
- `python src/imaging/analyze_pseudo_val.py` regenerates every pseudo-val comparison: CIs, paired deltas, per-class, subgroups, between-run correlation, pair ensembles.
- `python src/imaging/eval_tta.py <run>` scores inference-time window variants.
- Join predictions to `data/labels_v3/training_labels_v3.csv` on `StudyInstanceUID` for custom analyses. Keep everything private.

## 16. Recommended next steps (ranked by expected value per hour)

1. **Submit and calibrate.** Submit the final 3-seed ensemble. Optionally also submit a single all-fold model, or the interim E1p release, to measure the ensemble's LB gain. That calibrates pseudo-val and gold-58 against the leaderboard, which is unknown so far.
2. **More members of the same recipe.** About 80 min each locally. Seed averaging is the only measured, reliable gain (+0.003 for the 2nd model; diminishing after). Fold models (5 × 80 % data) instead of all-fold seeds would add data diversity and give out-of-fold predictions.
3. **Label noise is the ceiling, not the architecture.** No architectural change improved on E1 by more than seed noise (best +0.002; several were clearly worse), and confident-label AUC is 0.92–0.93 vs 0.88 overall. Next lever: knowledge distillation / noisy-student. Mix the teacher with out-of-fold image-model predictions, and re-train. This needs the 5 fold models above.
4. **Synovitis** (gold 0.68, the largest single loss). Reports omit about half of the image-visible synovitis, so a better report labeller cannot fix it.
   - **Option A.** Derive synovitis targets partly from image-visible correlates (effusion-synovitis appearance, effusion predictions).
   - **Option B.** Accept the ceiling.
   - Either way, judge any synovitis-specific change on an independent criterion, not by tuning on gold-58 (that would be selection on the test proxy).
5. **Higher input resolution done properly.** For example 384 px at 0.4 mm/px with a field of view that is not cropped. The v6 test (0.6 mm/px, 153.6 mm crop) cropped 81% of series and gave no gain. This needs a new cache (about 2–3 h CPU) and about 3× GPU per run (Kaggle T4×2 or longer local runs).
6. **Backbones without BatchNorm, or with frozen BN**, for ensemble diversity (CoAtNet, ConvNeXt-S). EfficientNetV2-S with BN on correlated mini-batches was −0.015.
7. **Robustness.**
   - An image-based side classifier for the 70 untagged near-midline studies.
   - A closer look at the Croatian subgroup (noisiest labels) and the studies with missing FS series (0.859).
8. **Deferred analysis runs** (about 4.5 GPU-hours; they do not affect the submission): orientation off (E1o), hard labels (E5h), max pooling (E2b), single planes (E3s / E3c / E3a). Run them with:

   ```
   python src/imaging/run_queue.py --queue output/imaging_v5/queue_deferred.json
   ```

9. **Owner actions.**
   - Make the GitHub repository private, and optionally purge history (`repo_remediation_plan.md`).
   - Keep the Kaggle dataset with weights and wheels **private**.
