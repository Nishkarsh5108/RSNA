# Knee MRI workstation (local demo)

A website and an interactive workstation for the knee MRI model. Drop a study (DICOM folder, files or a .zip). It is
parsed, rebuilt in 3D in patient coordinates and read for 12 findings by the model, on this machine's GPU. Every
step of the inference is streamed to the browser as it happens and can be replayed. The explanations are the
model's own exact per-cell attributions.

```
powershell -ExecutionPolicy Bypass -File demo\start.ps1      # build once, serve on http://127.0.0.1:8000, open the browser
```

Development, with hot reload on http://127.0.0.1:5173:

```
python demo/server/app.py                                     # API on 127.0.0.1:8000 (torch_gpu env)
npm --prefix demo/web run dev                                 # site; /api is proxied to the server
```

## What runs

`server/engine.py` imports the public model modules unchanged (`side.py`, `detector.py`, `classifier.py`, generated
by `src/postcomp/build_infer_kernel.py --local-dir`). It runs the same three passes as `src/postcomp/infer_bp.py`,
the pipeline behind the 0.958 private-leaderboard late submissions:

1. **Knee side**: ResNet-18d on every coronal and axial series gives p(right). Left knees are mirrored.
2. **ROI detector**: ConvNeXt-T per slice with 3D top-down fusion. It predicts one 3D box per finding per series and is
   averaged with a mirrored pass.
3. **Box-pooling classifiers**: every checkpoint in `KNEE_CLS_DIR` runs on every series (112 mm crop at 256 px,
   8×8 ConvNeXt-T feature cells per slice). Each finding pools only the cells in its own box.
4. **Ensemble**: the study probability is the mean over its series, then the mean over the models.

The engine runs one series at a time and streams each result (newline-delimited JSON):

- the side probability per series;
- both detector passes;
- the crop;
- every model's logits.

**Fidelity check**: `python demo/server/tests/test_fidelity.py` runs the engine and `infer_bp.py` on the 3 public
test studies with the same weights. On 15 Nov 2026 the largest difference over 60 series × model pairs was 1.6e-4
(fp16 noise).

**Explanations**: the classifier's head is linear over a weighted mean of feature cells, so
`logit_k = Σ_cells share_kc · (h_k · f_c) + b_k` holds exactly. The encoder output is captured from the model's own
forward pass and decomposed into two maps:

- **Attention** (`share`): where the model pooled.
- **Evidence** (`share × h·f`): what it found there.

The UI reports the largest gap between Σ contributions + bias and the forward's logit. In fp32 the gap is 5e-7; in
fp16 it is at most 0.05 logits.

**Lesion** overlay: the classifier's auxiliary lesion head is a 1×1 convolution, trained only during training on
per-slice lesion boxes. It gives each cell's probability of being covered by a lesion of the finding. It reads the
same captured features but does not feed the probability, so it is an independent localisation.

The evidence map shows what drove the score, and that need not be the most visible lesion. For each map the finding
panel lists the series' top slices, and selecting a finding jumps to the peak of the map currently shown.

**Hardware**: RTX 4050 Laptop, 6 GB. Weights stay fp32 and inference runs under fp16 autocast, as on the Kaggle T4.
Peak VRAM is 1.2 GB for a typical study and 3.1 GB for a 160-slice 640×1280 series. A study takes 4–11 s after the
first run warms up. No quantization is needed, so there is no accuracy cost.

## Privacy

- The server binds to 127.0.0.1 only.
- The browser zips dropped files and posts them to that local server.
- The server writes them to `KNEE_CACHE` (default `D:/rsna_work/demo_cache`, outside the repo) so the model's own
  DICOM reader can read them.
- Purging a study deletes them; the cache is also cleared on start and stop.
- Fonts and libraries are bundled, so the site makes no external requests.

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `KNEE_MODULES` | `D:/rsna_work/bp_modules` (side.py, detector.py, classifier.py) |
| `KNEE_PUBLIC` | `D:/rsna_work/public_code/tascj` (side and detector weights) |
| `KNEE_CLS_DIR` | `D:/rsna_work/kaggle_runs/knee-bp-weights-ds` (every `*.pth` is loaded; c2_f2 / c3_f3 join automatically) |
| `KNEE_RAW` | `data/Raw_dicom` (local sample studies: 3 public test, 4 gold-58 reference studies) |
| `KNEE_SAMPLES` | `D:/rsna_work/demo_samples.json` (which gold-58 studies to offer as reference samples; kept outside the repo) |
| `KNEE_CACHE` | `D:/rsna_work/demo_cache` |
| `KNEE_PYTHON` | the torch_gpu interpreter (used by start.ps1) |

## Limits

- One study at a time. A drop that contains several studies uses the one with the most images.
- Multi-frame (enhanced) DICOM is not read. Localizers and series with fewer than 6 slices are skipped, and the
  workstation lists them with the reason.
- Fat suppression is inferred from the header (99% agreement with the dataset's labels on 600 training series). It
  only affects the detector and can be changed per series.
- Research prototype. Not a medical device; not for diagnosis. Probabilities are trained against image-read labels
  produced by language-model agents (1st- and 7th-place public label sets), not against radiologist reads.

## Credit

- The model architecture and training scripts (side classifier, ROI detector, box-pooling classifier) are the public
  1st-place solution by tascj0 (Team Kangaroo).
- Labels: tascj0 (CC BY 4.0) and songqizhou (7th place, CC BY-NC-SA 4.0).
- Data: RSNA 2026 knee MRI competition.

Our work is the label fusion, the fold models, the inference pipeline, the evaluation, and this workstation (see
`reports/POSTCOMP_DECISION_LOG.md`).
