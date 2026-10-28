# Image pipeline audit and corrections (v1–v4 → v5)

Date: 2026-10-17. Every check below was executed, with code under `src/imaging/` and outputs in `reports/eda/imaging/` (git-ignored). Unexecuted items are marked.

## 1. The original pipeline (traced from code)

| Stage | v1 | v2 (LB 0.775) | v3 | v4 | Where |
|---|---|---|---|---|---|
| DICOM → array | third-party `.npz` (24×224×224; missing hidden-test studies → LB 0.500) | `_order_series_files` (sort by IPP·normal, fallback InstanceNumber, filename); `_decode_slice` (RescaleSlope/Intercept, MONOCHROME1, middle frame); `preprocess_series` (per-series 1–99 percentile → uint8 → PIL bilinear 224×224) | same, plus mirroring by DICOM `Laterality=='R'` for coronal/axial | v3 preprocessing | `src/model/knee_mri_v2_training.ipynb` §A; v3 §A |
| Series selection | 1 per plane | `build_priority_series`: FS first, else any series of the plane; ties broken by UID order | up to 3 per plane, FS first | same as v3 | §1 |
| Slices | 12, `linspace` by index | 12, `linspace` by index | 3 × 12 slots + mask | 12 / 8 | §B |
| Backbone | EfficientNet-B0 ImageNet, **frozen** | frozen | frozen | DINOv2-S frozen (fine-tune optional) | §B |
| Aggregation | attention pool → MLP | attention pool → MLP | region-guided plane attention | same | §B |
| Missing planes | zero-filled | **zero-filled** (`_fallback_count`) | masked | masked | §B |
| Checkpoint selection | best gold AUC | **best gold AUC on the val fold's 12–15 gold studies** | same | same | §E |

## 2. Defects found (with evidence)

| # | Defect | Evidence | Impact |
|---|---|---|---|
| D1 | **Undecodable slices are skipped silently.** A plane with no decodable slices is zero-filled. | `preprocess_series`: `except Exception: continue`; Section B fallback to zeros | Hidden-test JPEG-Lossless/J2K (official data page) without decoders → silently blank planes and invisible degradation |
| D2 | No JPEG-Lossless decoder in the original environment | `get_decoder(JPEGLosslessSV1).is_available == False` before installing pylibjpeg-libjpeg/gdcm [measured] | as D1 |
| D3 | Laterality: v2 none; v3/v4 mirror only `Laterality=='R'` | Tag present in 2,203/4,407 studies; 23 series tagged 'RIGHT' are missed | About half of right knees keep reversed medial/lateral anatomy in coronal and axial views |
| D4 | **Sagittal slice order is not laterality-normalised** in any version (and in no public solution inspected) | All sagittal series have rows +y, cols −z → slice normal −x, so ascending order runs patient-left → right: lateral→medial for left knees, medial→lateral for right knees [derived from the uniform orientation table, verified numerically below] | Position-aware or ordered aggregation sees medial and lateral compartments swapped for about half the studies |
| D5 | Geometric side from an arbitrary sagittal slice is noisy | Sagittal slices span about ±50 mm in x. Using coronal/axial only cut tag/geometry conflicts 57 → 28 and raised report agreement 0.955 → 0.963 on geometry-only studies [measured] | 27 studies (0.6%) changed side |
| D6 | Index-based slice sampling ignores physical spacing | Spacing 0.6–6.2 mm (median 3.5); 3–7% of sagittal/axial series are thin 3D | 3-slice windows would span 1.2 mm on 3D series vs 7 mm on 2D |
| D7 | Aspect ratio ignored in the 224×224 resize | 1,769 series (7.3%) are non-square (aspect 0.5–2.0) [measured] | Geometric distortion; consistent between train and test, so not a leak |
| D8 | No physical in-plane resampling | FOV 120–200 mm at the same matrix size [EDA] | Physical scale varies up to about 1.7× |
| D9 | Frozen backbone | Gold 0.73–0.78; LB 0.775 | Main performance limiter (confirmed by E1 vs E0) |
| D10 | Checkpoint selected on gold | Notebook code | Reported "honest" gold AUCs are optimistic |
| D11 | **Two training DICOM files are truncated** (half the pixel data missing); v2 silently dropped those slices | Found by the strict v6 cache build: `ValueError: number of bytes of pixel data is less than expected (409600 vs 819200)` in one series (slice 36 of 36) and `(173056 vs 346112)` in another (slice 1 of 38). The legacy v2 cache holds 35 and 37 slices for them [measured] | Benign here (end slices), but the same silent skip would hide systematic decoder failures (D1) |
| D12 | **Kaggle runtime mismatch.** The Kaggle image is Python 3.13 (`Kaggle/docker-python` `Dockerfile.tmpl`: Colab base 2026-09-17, `python3.13` paths), but the bundled decoder wheels covered only cp310–cp312. | Wheel tags in the upload folder vs Dockerfile [verified] | On the hidden test (JPEG-Lossless/J2K), decoders would fail to install → abort → **no submission** |
| D13 | The decoder-install branch of the notebook was never exercised: decoders were installed after `pydicom` had been imported, relying on a module reload to discover the plugins | Code reading; locally the branch never ran because decoders were preinstalled | Possible "decoders unavailable" abort on Kaggle |
| D14 | Hard-coded bf16 autocast in training and inference | `train.py` / `predict.py` | Kaggle T4 (pre-Ampere) has no native bf16: emulated, slow, or failing |
| D15 | **Slice-position input mirrored for right-knee sagittal series** (a bug in *my* v5 dataset code). The windows themselves are correctly mirrored, so window k sits at anatomical position p (0 = lateral) for both knees. But the position fed to the model was 1 − p for right knees, so the position signal contradicts itself between sides (55% of studies are right knees). | Found by `attention_map.py`: attention grouped by window index separates the menisci sharply (lateral meniscus peaks at sagittal position 0.30, medial at 0.70–0.83), yet attention-weighted mean *input* position was ≈ 0.5 for both. Unit check: right-knee input positions 0.97 → 0.03 vs left 0.04 → 0.96; images identical. | Noisy position cue for sagittal tokens in all E-runs so far (E1o unaffected). The image content still carried the anatomy. |

## 3. Corrections implemented (v5) and how each was verified

| Fix | Implementation | Verification |
|---|---|---|
| **Decode failures are loud** (D1, D2) | `infer_lib.preprocess_series` raises `DecodeError`. `predict.assert_decoders` aborts if JPEG-Lossless/J2K decoders are missing. A failed series falls back to the next candidate of the same slot, or the slot is masked; more than 2% of studies failing aborts. | Test 2: truncated FS series → logged, fallback used. Test 4: fake missing decoders → abort. Transcoded JPEG-Lossless and J2K slices decode bit-identically once the wheels are installed (`decode_selftest.py`). |
| **Exact train/test preprocessing parity** | Inference re-implements v2 semantics (so the verified 40 GB v2 cache can be reused) | **Bit-exact on 51/51 series** (15 local test, 36 train; max abs diff 0). The rebuilt study index equals the training index (`check_infer_exactness.py`). |
| **Laterality policy** (D3, D5) | Study side = majority DICOM tag; otherwise majority geometric side of coronal/axial series ≥ 20 mm off-midline; flags `side_conflict` / `side_ambiguous` | Tag vs geometry conflicts adjudicated with the side written in reports: **tag 29 vs geometry 10** → tag-first. Sources: tag 2,203, geometry 2,204; conflicts 28; ambiguous 131 (70 untagged; these rely on weak geometry). |
| **Orientation normalisation** (D3, D4) | Coronal/axial right knees: mirror columns. **Sagittal right knees: reverse slice order** (and channel order). Lateral is then image-right (cor/ax) and slice 0 (sag) for every study. | Pearson r between mean left-knee and mean right-knee images (250 + 250 studies): coronal **0.869 → 0.988**, axial 0.968 → 0.989, sagittal at 20% depth 0.950 → 0.988, at 80% 0.972 → 0.990. Visual montage `reports/eda/imaging/orientation_mean_images.png`. |
| **Series selection** | 6 slots = plane × {FS, non-FS}; within a slot the series with most slices; ≥ 3 slices; FS slot falls back to the plane's non-FS series when missing (416 studies lack a sagittal or coronal FS series) | `study_index.py` counts. Fallback exercised in inference Test 2. |
| **Missing series** | Explicit slot mask; model attention ignores masked tokens; never zero images | Inference Test 3: coronal removed → slot masked; only that study's predictions change |
| **Physical 2.5D step** (D6) | Window = [c−s, c, c+s], s = max(1, round(3.5 mm / spacing)); windows spread over 4–96% of the slab | 285 sagittal and 314 axial FS series use s > 1 |
| **Duplicate pair** | Forced into one CV fold | Both studies in fold 1 |
| **Gold sealed** (D10) | Gold-58 predictions saved at the pseudo-val-selected epoch; scored once by `evaluate_sealed.py` after model selection. The script refuses a second run (lock file `GOLD_UNSEALED.json`, written with the locked selection *before* any metric is computed). | — |
| **Explicit end-slice policy** (D11) | `infer_lib.decode_series`: exactly one undecodable slice at either end of the slice order, in a series of ≥ 10 slices, is dropped and **reported** (`dropped_slices`; inference report stage `dropped_end_slice`). Any other decode failure raises → next candidate series or masked slot. | Both affected training series reproduce the legacy v2 cache **bit-exactly** (35 and 37 slices). Fully truncated series still fail and fall back (inference test 2). |
| **Kaggle Python 3.13** (D12) | Wheels for pylibjpeg-libjpeg 2.4.0 and pylibjpeg-openjpeg 2.6.0 now cover cp310–cp314 (manylinux x86_64), plus pydicom 3.0.2 and timm 1.0.30 | Wheel tags listed; offline install path in the notebook |
| **Install before import** (D13) | The dependency cell installs pydicom / decoders / timm from attached wheels **before pydicom is first imported**, asserts that pydicom was not imported earlier, then asserts JPEG-Lossless (both SVs) and JPEG-2000 (both) decoders are available | Local notebook run (each notebook in a fresh interpreter) |
| **Process pools** | `infer_lib.process_pool` uses the `fork` context where available (Python 3.14 changed the POSIX default to `forkserver`, which cannot pickle notebook-defined functions) | Local tests (Windows falls back to the default context) |
| **Sagittal position fix** (D15) | `KneeStudyDataset(sag_pos_fix=True)` / `train.py --sag-pos-fix` gives p for both sides. The flag is stored in the checkpoint args; inference and evaluation read it, and default to the old behaviour for older checkpoints (the interim E1 release keeps working exactly as trained). | Unit check above; effect measured by run E1p (E1 + fix) |
| **Precision by GPU** (D14) | bf16 on compute capability ≥ 8, otherwise fp16 (inference) / fp16 + GradScaler (training; `--amp auto`) | `check_amp_precision.py` on 120 val studies: fp16 max \|Δp\| vs fp32 = **0.0014**, bf16 = 0.013; per-class Spearman ≥ 0.9998. fp16 training path smoke-tested, including resume with scaler state. |

## 4. Not changed (documented trade-offs)

- **D7 / D8 (aspect ratio, physical resampling).** The v2 cache was kept so experiments could start immediately; re-caching 24k series would take hours. The distortion is identical in training and inference (bit-exact), so it costs accuracy at most and is not a validity problem.
  - **v6 preprocessing implemented** (`infer_lib.preprocess_series_v6`): same decode, order and percentile steps, then aspect-preserving PIL bilinear resampling to 0.6 mm/px, then a centred 256 × 256 crop or zero-pad (153.6 mm field of view, as in a public 0.923-LB reader). Visual check: `reports/eda/imaging/v2_vs_v6_preprocessing.png`.
  - **v6 cache built and verified** for the 13,221 FS-first series (`build_cache_v6.py`, 118 min on 2 CPU processes, 31 GB):
    - The first pass failed on exactly the two D11 files, and `finalize_cache_v6.py` refused to mark the cache ready.
    - After the explicit end-slice policy, a re-run filled both series (dropped slices recorded in `_dropped_slices_*.csv`).
    - Verification at 13:50 IST: 13,221/13,221 files present, 0 failures, 300 sampled arrays valid (uint8, T × 256 × 256, non-constant).
  - Tested in E8. Inference reads each checkpoint's `preproc` and decodes per version, so v2 and v6 models can be ensembled (inference test 5).
- **70 untagged near-midline studies** may have wrong orientation (geometry ≈ chance there). An image-based side classifier (e.g. fibula position) could resolve them. Not built.
- **Resolution** fixed at 224 px by the cache.
