# The current model, mathematically (the 0.927 submission)

Derived from the code that produced the submitted checkpoints:
- `src/imaging/study_index.py`, `infer_lib.py`, `data.py`, `model.py`, `train.py`, `predict.py`;
- frozen copies in `D:/rsna_work/fallback_0927/src_imaging/`.

Shapes and parameter counts were printed from the code on 2026-10-18. Coverage statistics were computed over all 4,407 training studies.

Notation:
- B studies per batch, S = 3 series slots, K = 8 windows per slot;
- H = W = 224 px, D = 384, C = 12 findings.

## 1. From DICOM study to series volumes

**Series selection** (`study_index.choose_slots`). Each study is split into 6 slots: {sagittal, coronal, axial} × {fluid-sensitive (FS), non-FS}.
- The model uses slots 0–2: the FS series of each plane, falling back to the same plane's non-FS series when no FS series exists (`fallback=True`).
- A slot with no series at all is masked: m_s = 0.

**Slice order** (`order_series_files`). Slices are sorted by the projection of ImagePositionPatient onto the slice normal, n = r × c (the cross product of ImageOrientationPatient's row and column vectors). The fallbacks are InstanceNumber, then file name.

**Intensity** (`preprocess_series`). Per series, with v the stacked slices after RescaleSlope/Intercept and MONOCHROME1 inversion:

  u = clip((v − q₀.₀₁) / (q₀.₉₉ − q₀.₀₁), 0, 1) · 255, then uint8

q are the series-wide 1st/99th percentiles. The window is per series, not per slice, so relative brightness across slices is kept.

**Resizing.** Every slice is resized to 224 × 224 with PIL bilinear, a **squash** that ignores aspect ratio and physical pixel size.
- The largest field-of-view side has quantiles 140/160/160/170/192 mm (5/25/50/75/95%).
- The model therefore sees **0.62–0.86 mm/px (median 0.71)**. Physical scale varies by ±15% between studies.
- This is the main geometric difference from the strongest public models, which see 0.365 mm/px over a fixed 140 mm field.

**Decode failures** (`decode_series`).
- Exactly one undecodable slice at either end of a series of 10 or more slices is dropped and reported.
- Any other failure raises an error. At inference the next candidate series is tried, and otherwise the slot is masked.
- More than 2% of studies failing aborts the run. Nothing is ever zero-filled.

## 2. Windows, coverage and physical spacing (`data.py`)

For a series with T slices (span limits lo = 0.04, hi = 0.96):
- **Evaluation:** centres c_k = round(lo(T−1) + k·(hi−lo)(T−1)/(K−1)), for k = 0…7.
- **Training:** one centre drawn uniformly in each of K equal bins of the same span (stratified jitter).
- **Window:** the 3 slices {c−s, c, c+s}, clipped to [0, T−1], with s = max(1, round(3.5 mm / spacing)) giving about 3.5 mm between neighbours.
- **Result:** each window is a 3-channel image x_{s,k} ∈ [0,1]^{3×224×224}, and each study is a tensor X ∈ [0,1]^{3×8×3×224×224}.

Measured over the 4,407 training studies:

| Plane | Slices T (5/50/95%) | Spacing mm (median) | Slab extent mm (median) | Slices covered, K = 8 (median; 5th pct) | Covered with the deployed shift bag |
|---|---|---|---|---|---|
| Sagittal | 19 / 30 / 48 | 3.3 | 104 | 80%; 50% | 100% |
| Coronal | 18 / 30 / 38 | 3.5 | 106 | 80%; 63% | 100% |
| Axial | 20 / 32 / 144 | 3.9 | 126 | 75%; **17%** | 100% (5th pct 28%) |

Reading:
- One study is about 106 decoded slices; the model encodes 24 windows (72 slice-channels), or 48 windows with the shift bag.
- Coverage is adequate except for thin-slice 3D axial series (about 5–10% of studies, 0.6 mm spacing, T up to 192), which K = 8 samples sparsely.
- **Continuity between slices** is modelled only *inside* a window: 3 channels spanning about 7 mm, mixed by the first convolution.
- Windows never interact before pooling (§4). The only cross-window information is the slice-position input p.

## 3. Laterality normalisation

The side is taken from the DICOM Laterality tag when present. Otherwise it comes from the sign of the volume-centre patient-x (97.8% agreement with the tag where both exist). 70 studies are flagged ambiguous.

For right knees:
- **Coronal and axial:** columns are mirrored (W axis).
- **Sagittal:** slice order is reversed. In the cache, sagittal slices are ordered patient-left to patient-right; after reversal, slice 0 is lateral for both sides.

The slice-position feature is p = c/(T−1) for all knees in the submitted models (`sag_pos_fix=True`). Earlier models received 1 − p for right-knee sagittal series (defect D15).

## 4. Network (`model.KneeMIL`, 28.37 M parameters: 27.82 M encoder, 0.55 M head)

| Step | Operation | Shape (per batch) |
|---|---|---|
| Input | uint8 → float/255, valid windows only | X: [B, 3, 8, 3, 224, 224] → [N = Σ valid slots × 8, 3, 224, 224] |
| Normalise | (x − μ_ImageNet) / σ_ImageNet (the three slices fill the RGB slots) | [N, 3, 224, 224] |
| Encoder | ConvNeXt-T (`convnext_tiny.fb_in22k_ft_in1k`): 4 stages, LayerNorm, final map 768 × 7 × 7, global average pool + LayerNorm | [N, 768] |
| Projection | Dropout(0.2) → Linear(768 → 384) → LayerNorm | f: [B, 3, 8, 384] |
| Token embeddings | t_{s,k} = f_{s,k} + e_slot[s] + MLP_pos(p_{s,k}), with MLP_pos: R → 384 → GELU → 384 | T: [B, 24, 384] |
| Mask | token valid ⟺ slot present | [B, 24] |
| Per-finding attention | a_{c,j} = (W₂ tanh(W₁ t_j + b₁) + b₂)_c, with W₁ ∈ R^{256×384}, W₂ ∈ R^{12×256}; α_{c,·} = softmax over valid tokens j | α: [B, 12, 24] |
| Pooling | v_c = Σ_j α_{c,j} t_j | V: [B, 12, 384] |
| Heads | z_c = ⟨v_c, w_c⟩ + b_c, with w_c a row of cls_w ∈ R^{12×384} | logits [B, 12] |
| Output | p̂_c = σ(z_c) | [B, 12] |

**Cross-plane fusion.** Because z_c is linear in v_c:

  z_c = Σ_j α_{c,j} · ⟨t_j, w_c⟩ + b_c = Σ_j α_{c,j} · σ_{c,j} + b_c

where σ_{c,j} is a **per-window score** for finding c. The study logit is therefore an attention-weighted *average* of window scores:
- It lies between the smallest and largest window score: min_j σ_{c,j} ≤ z_c − b_c ≤ max_j σ_{c,j}.
- Planes are combined only through the softmax weights. Evidence on sagittal and coronal windows is averaged, not conjoined.
- A finding visible in 1 of 24 windows raises z_c only if attention concentrates on that window. This is why attention beat mean pooling by 0.008 (E2a) and a transformer over tokens added only +0.002 (E2c).
- The 12 heads share the encoder, projection and tokens. Only the 12 attention distributions and the 12 vectors w_c are finding-specific.

## 5. Training objective and optimisation (`train.py`)

**Targets.** y ∈ [0,1]^{12} is the labels_v3 soft teacher: the mean of three report-derived sources (order-averaged Qwen3-4B forced-choice score s₀ = p_A + 0.5 p_B + 0.25 p_E, gpt56sol, report_v2).
- Gold agreement is 0.890 macro AUC.
- No cell is missing, so there is **no missing-label mask**. A report that is silent on a finding gets whatever value the sources give (usually low).
- Gold studies are excluded from training (`--gold-weight 0`).

**Loss.** Soft binary cross-entropy averaged over the 12 findings, then over the batch:

  L = (1/B) Σ_b (1/12) Σ_c −[y_{bc} log σ(z_{bc}) + (1 − y_{bc}) log(1 − σ(z_{bc}))]

- No positive-class weights.
- No confidence weighting: weighting by |2y−1| was tested in E5w and lost 0.005.

**Augmentation.** One affine transform per (study, slot), shared by all windows and channels of the series so the slices stay aligned: rotation ±10°, scale ±10%, translation ±6%. Then intensity x ← clip(x^γ · g, 0, 1), with log γ ~ U(−0.2, 0.2) and g ~ U(0.9, 1.1). Window centres are also jittered in training (§2).

**Optimisation.**
- AdamW: backbone LR 1e-4, head LR 5e-4, weight decay 0.05.
- Batch of 2 studies, 2 accumulation steps (effective batch 4); 870 optimiser steps per epoch on 3,479 studies (fold runs), or 1,087 on 4,349 (final models).
- Linear warm-up over 0.5 epoch, then cosine decay to 0 at epoch 8; gradient-norm clip 1.0.
- bf16 autocast locally; fp16 + GradScaler on pre-Ampere GPUs.
- EMA of the weights with decay 0.999 (timm ModelEmaV3, warm-up schedule).

**Checkpoint.**
- Fold runs: the epoch with the best pseudo-val macro AUC, which was the last epoch in every 8-epoch run.
- Final models (F1–F3, all 4,349 non-gold studies): the last-epoch EMA weights.

## 6. Inference, test-time averaging and ensembling (`predict.py`)

Per model m:
- **Two bags:** the training setting and a bag shifted by Δ = (hi − lo)/(2(K − 1)) = 0.0657 of the slab.
- p̂_m = ½(σ(z_m^{bag1}) + σ(z_m^{bag2})).

The submission is p̂ = ⅓ Σ_{m ∈ {F1, F2, F3}} p̂_m, with seeds 42, 7 and 11.

Precision: fp16 on T4. The measured max |Δp| vs fp32 is 0.0014, with Spearman ≥ 0.9998.

## 7. Audit questions

| Question | Answer | Evidence |
|---|---|---|
| Does it sample enough anatomy? | Mostly. K = 8 covers 75–80% of slices; with the shift bag about 100%. Thin-slice axial is the exception. **Not** in-plane: 0.71 mm/px is coarse for meniscal tears (1–3 mm) and cartilage (2–3 mm). | §2 table; public 0.949 model uses 0.365 mm/px |
| Can it learn continuity between slices? | Only within a 3-slice window of about 7 mm, through the first convolution. No 3D context and no inter-window interaction. | §4 |
| Can it combine evidence across planes? | Only as an attention-weighted average of per-window scores; no AND-type conjunction. A 2-layer transformer over tokens (E2c) added +0.002, inside noise. | §4, E2c |
| Imaging features or report-writing correlations? | Mainly imaging features. Attention follows the anatomy without localisation labels: medial vs lateral sagittal halves, posterior coronal for Baker's, suprapatellar axial for effusion. The model **beats its own report labels on gold** for medial OA (0.977 vs 0.923), lateral OA (0.896 vs 0.817), effusion (0.939 vs 0.871), contusion (0.961 vs 0.852) and fracture (0.921 vs 0.853). Where the report definition differs from the image read (synovitis: model 0.680, labels 0.74), it learns the report's notion. | attention analysis (E1); label report §5 |
| Limited by architecture, data representation, labels or validation? | Labels and validation first, then representation (resolution); architecture last. Every architectural variant (mean, transformer, 6 series, EfficientNetV2-S, 12 epochs) landed within ±0.003 or below, while seeds and data volume moved gold by 0.013–0.015. | `image_model_experiments.md` |

### Leakage and confounding audit

| Risk | Status |
|---|---|
| Duplicate studies | One pixel-identical pair (…23376711 / …31613209), placed in one fold (fold 1). No other duplicates by pixel hash. |
| Patient overlap | Not identifiable: PatientID is pseudonymised per study (4,407 unique). Repeat scans of one patient could straddle folds; the effect cannot be measured. |
| Site or scanner confounding | Folds are stratified by report language (a site proxy) × positive burden, **not grouped by site**. A third-party repository reports that random folds inflate AUC by about 0.05 through scanner memorisation. Our pseudo-val may therefore be optimistic, but by the same amount for every compared run, so paired deltas are less affected. The test set is believed to come from the same sites. |
| Report templates | Reports are never model inputs. Templates affect only the labels: Croatian reports have the noisiest teacher, with source SD 0.144 and 30% uncertain cells. |
| Train/validation leakage | None found. Gold is excluded from training. The 5-fold split is fixed (seed 42, identical in index v1 and v2). Pseudo-labels (planned) use out-of-fold predictions only. |
| Selection on gold | Avoided since the v5 protocol. Gold was scored once (2026-10-18 04:15) and once more for the pre-registered synovitis check H-SYN1. |

## 8. The three most consequential bottlenecks

1. **Labels: report-derived targets vs image-read test labels.**
   - The teacher agrees with gold at 0.890 macro.
   - The model learns the report's definitions, including the 0.68–0.74 synovitis ceiling and report thresholds for effusion, OA grade and "possible" contusion.
   - Evidence: the image model already beats its labels on 6 of 12 findings on gold. Strong third-party single models (0.95+) attribute most of their gain to labels and out-of-fold pseudo-labels.
2. **In-plane resolution and fixed physical scale.**
   - 224 px squash gives 0.71 mm/px with ±15% scale variation, half the resolution of the strongest public models (0.365 mm/px, fixed 140 mm field).
   - Evidence: public score histories (nartaa: 336 → 384 native plus all windows, +0.001–0.006); our 0.6 mm/px test (E8) was too small a step to settle it. Experiment HR tests the full step.
3. **Evaluation blind spot.**
   - Pseudo-val measures agreement with report labels, so it cannot reward label improvements or report-blind findings, and it may be inflated by site memorisation.
   - Gold-58 is too small (two seeds differ by 0.013).
   - Consequence: the levers that matter most (labels, synovitis) can only be judged on the public LB, about 390 studies with 5 submissions per day.
