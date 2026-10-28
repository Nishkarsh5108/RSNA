# Plan for the optional professor GPU sessions (RTX PRO 6000 Blackwell; Tuesday 2026-10-20 and/or Wednesday 2026-10-21)

Status: **access not confirmed** (updated 2026-10-19 01:15 IST). The owner calls the professor on Monday and asks for Tuesday and Wednesday: the college is a 2-hour commute each way, so Monday would be too short. Nothing in the main plan depends on these sessions. The local RTX 4050 queue continues either way.

**Since 2026-10-19 the main pick is the public OAI-trained pipeline blended with our models** (owner's decision). These sessions now improve our own members (their share of the blend, and the clean second pick); they no longer decide the main score.

**Recommendation (2026-10-19 04:36 IST): skip the sessions unless the owner wants them for learning.** Plan §10c dropped our models from both final picks:
- the picks are two fastMRI-free variants of the public pipeline;
- a 0.93-level model adds nothing to a 0.954 blend;
- a ConvNeXt-S trained by Wednesday cannot close a 0.026 gap.

Four hours of commuting buy no expected leaderboard gain.

## 0. Before the session

**Ask the professor:**
1. Tuesday all day, Wednesday if possible; may jobs run overnight Tuesday → Wednesday?
2. Operating system; a personal login or folder that other users cannot read (competition rule 4, §2).
3. About 150 GB free disk; permission to install Python packages (Blackwell needs a PyTorch build for CUDA 12.8) with internet access for the install.
4. Can the laptop reach the machine by SSH on the college network? Then the session is driven from the laptop; otherwise the owner runs the commands in §4.

**Bring:** the laptop and an external USB 3 SSD of at least 128 GB holding the hr cache (54 GB; copied Monday night) plus the code, index and labels (§2).

## 1. Jobs, in priority order

HR was adopted on Sunday (E9, +0.0048 pseudo-val, §8 of the experiment plan), so every job uses the hr cache. Times on the big GPU are **estimates** (5–10× the 4050; measured by job 1).

| # | Job | Runs | Est. time | Feeds |
|---|---|---|---|---|
| 0 | Environment check + smoke test (§3–4) | 1 | 0.5 h | go / no-go |
| 1 | Speed calibration: E9 recipe, 1 epoch, fold 0 | 1 | 0.2 h | how many runs fit |
| 2 | Backbone test, fold 0, labels_v3, index v1 (identical to E9 except the backbone): ConvNeXt-S; ConvNeXt-B if job 1 shows headroom | 1–2 | 1–2 h | paired pseudo-val vs E9 (K = 12 + shift); adopt if Δ > +0.004, the same bar HR passed |
| 3 | Round-2 teachers: 5 folds, best backbone, hr | 5 | 2–4 h | better pseudo-labels, used only if the Monday LB A/B shows pseudo-labels help (§9 of the plan) |
| 4 | Final students: all folds, best backbone, hr, the labels the LB favours | 3–5 seeds | 2–5 h | candidate A |
| 5 | v2 (224 px) students with the best backbone, for ensemble diversity | 2–3 | 1–2 h | candidate A |

- **Two days:** jobs 0–3 on Tuesday (job 4 overnight if the machine may stay on), 4–5 on Wednesday.
- **One day (Tuesday):** jobs 0, 1, 2, then 4 (both label variants if the LB verdict is not in yet).
- **Hard cut-off:** checkpoints must be back on the laptop by Wednesday about 21:00 IST. Thursday 2026-10-22 is for packaging, upload, LB scoring and the final selection (deadline Friday 05:29 IST). Anything still training then is dropped.
- **Safety copy:** bring Tuesday's finished checkpoints home on Tuesday evening as well.

Why it matters:
- Laptop (measured): one ConvNeXt-T hr all-fold model takes about 2.5 h (4.2 studies/s), so 6–7 models per day.
- ConvNeXt-S hr fits the laptop only at batch 1 (4.94 GB peak, measured) at an estimated 5 h per model, so it is out of reach locally.
- Third-party: one competitor's single larger CoAtNet at 0.365 mm/px with every slice at inference scored 0.940 public LB on competition data alone (self-reported), versus our 0.927. Model size, resolution and windows are the levers this GPU opens (hypothesis: labels may also differ).
- All code paths are tested locally. Only the backbone name, batch size and paths change.

## 2. What to bring (prepared on the laptop)

| Item | Size | Location |
|---|---|---|
| Code: `src/imaging/*.py`, `src/labeling/metrics.py`, `src/emergency/*.py` | < 1 MB | repo |
| Index | 3 MB | `data/imaging_v5/study_index_train.csv`, `study_index_train_v2.csv` |
| Labels | 2 MB | `data/labels_v3/training_labels_v3.csv` (+ `pl_*.csv` when built) |
| Cache | hr: **54 GB (measured)**; v2: about 30 GB (only for job 5) | `D:/rsna_work/cache_hr352/`, `output/v2/preprocessed/` |
| E9 prediction set (for the paired comparison) | < 1 MB | `output/emergency_research/preds/E9_hr_cnxtT_s3_k6_attn12_c320_pf__shift.npz` |

Transfer: an external USB SSD (about 10–30 min for 85 GB, estimate).

**Competition-data rule 4.** The caches, labels and index are derived from competition data. The rules require "reasonable and suitable measures to prevent persons who have not formally agreed to these Rules from gaining access".
- Keep them in your own user account or folder, not readable by other users.
- Do not upload them anywhere.
- Delete them at the end of the session.
- Bring back only checkpoints and logs.

## 3. Environment (inspect first; do not assume)

```bash
nvidia-smi                                   # GPU model, VRAM, driver, CUDA version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
df -h .                                      # need about 70 GB free for the cache + checkpoints
nproc; free -g                               # data-loader workers: about nproc / 2
```

Blackwell GPUs report compute capability 12.x and need a PyTorch build with CUDA ≥ 12.8. If the machine's torch lacks it:

```bash
python -m venv knee && source knee/bin/activate        # or conda create -n knee python=3.11
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install timm==1.0.30 pandas numpy scikit-learn scipy
```

The code paths are relative to a root `R` in `train.py` and the queue (`R = "D:/Coding/kaggle/RSNA"`). On another machine, pass every path explicitly: `--index`, `--labels`, `--val-labels`, `--cache` and `--out-root`. Do not edit the code.

## 4. Commands (tested on the laptop; see §6)

Smoke test (2 min, must finish with `DONE`):

```bash
python src/imaging/train.py --run-id BW_smoke --index <idx_v2> --labels <labels> --val-labels <labels_v3> \
  --cache <cache_hr352> --preproc hr --crop 320 --K 6 --K-eval 12 --sag-pos-fix \
  --backbone convnext_small.fb_in22k_ft_in1k --bs 2 --accum 2 --epochs 1 --max-train 64 --workers 8 --out-root runs
```

Main runs (HR branch), one at a time. They are resumable: re-run the same command with `--resume`, which also restarts cleanly after a time limit.

```bash
for S in 42 7 11; do
python src/imaging/train.py --run-id BW_hr_cnxtS_s${S}_all --index <idx_v2> --labels <pl_g.csv> --val-labels <labels_v3> \
  --cache <cache_hr352> --preproc hr --crop 320 --K 6 --K-eval 12 --sag-pos-fix --train-all-folds \
  --backbone convnext_small.fb_in22k_ft_in1k --bs 2 --accum 2 --epochs 8 --workers 8 --seed $S --out-root runs \
  --time-limit-min 110 > runs/BW_hr_cnxtS_s${S}_all.log 2>&1
done
```

- **Batch:** `--bs 2 --accum 2` keeps the effective batch of 4 studies used locally. With more than 40 GB VRAM, `--bs 4 --accum 1` gives the same effective batch with fewer steps.
- **Precision:** `--amp auto` selects bf16 on Blackwell.

## 5. Validation and export

- **Pseudo-val is not a holdout** for all-fold students; only their training curves are logged.
- **Fold-0 sanity check:** if time allows before the all-fold seeds, run one fold-0 ConvNeXt-S (drop `--train-all-folds`). It gives a paired pseudo-val comparison against E9/E1p using `src/emergency/eval_windows.py` + `compare_preds.py`.
- **Gold:** gold predictions are saved sealed by every run. They are not opened during the session.
- **Export:** copy each `runs/<run_id>/ckpt_best.pt` (= last epoch for all-fold runs; about 200 MB for ConvNeXt-S) and `config.json`/`history.csv` back to the laptop. Verify SHA-256 on both ends:
  - Linux: `sha256sum`
  - Windows: `certutil -hashfile ckpt_best.pt SHA256`
- **Kaggle compatibility:** the inference notebook reads the backbone name, preprocessing and crop from the checkpoint args. A ConvNeXt-S checkpoint needs only timm (already bundled), so no extra wheel.

## 6. Pre-session test log

| Test | Result |
|---|---|
| HR data path + forward/backward on CPU (ConvNeXt-T, crop 320, K = 6 train; K = 12 and K = 0 eval) | passed 2026-10-18 15:00 |
| Equivalence of the updated code with the frozen 0.927 code (dataset train/eval, logits) | passed: identical tensors, max logit diff 0.0 |
| HR training on GPU (E9, ConvNeXt-T) | passed 2026-10-18 18:31: 8 epochs, 4.2 studies/s, 3.1 GB peak, pseudo-val 0.8848 (K = 12) |
| ConvNeXt-S smoke test on the 4050 (hr, crop 320, K = 6, batch 1, 32 studies, 1 epoch) | passed 2026-10-18 18:55: 4.94 GB peak; the full run-to-checkpoint path works |
| Path audit for another machine | `train.py`: every input and output has a flag (`--index --labels --val-labels --cache --out-root`); the one other D: path is read only with `--random-series` (unused). `eval_windows.py`, `compare_preds.py` and `build_pseudo_labels.py` use a fixed root: either evaluate on the laptop after copying the run folder back, or add a `RSNA_ROOT` override **after** the overnight queue ends (the PL watcher runs `build_pseudo_labels.py` tonight) |
| Local library versions to match | torch 2.7.1 (cu118 locally; cu128 build needed on Blackwell), timm 1.0.30, numpy 2.4.4, pandas 2.3.3, scikit-learn 1.8.0 |

## 7. If access is not granted

The same commands run on the laptop at 1.5–2.5 h per model (ConvNeXt-T). ConvNeXt-S is not attempted locally: it would need gradient checkpointing and about 4 h per model.
