# Emergency experiment results (2026-10-18 →)

Status: **updated 2026-10-20 (final-strict 0.956, seen 05:37 IST; gold-58 follow-up finished 08:21 IST, nothing adopted)**. The machine-readable table is `output/emergency_research/experiments.csv`, rebuilt by `src/emergency/update_experiments_csv.py` from run folders and saved predictions; metrics are recomputed, never typed in. Every comparison below is paired on the same 870 pseudo-val studies (fold 0) unless stated.

Seed-to-seed noise reference (previous protocol): macro +0.0026 between two seeds; up to 0.03 in a single finding.

## 1. Completed

| ID | Question | Result | Decision |
|---|---|---|---|
| H-SYN1 | Synovitis output := rank-mean(synovitis, effusion)? (gold, pre-registered) | 0.680 → 0.702, Δ +0.022 [−0.043, +0.093] | Not adopted (expected macro gain ≤ 0.002) |
| H-SYN2 | Synovitis label := effusion label where the report is silent? (label level, gold, pre-registered) | 0.756 → 0.738 | Not adopted (our labeller is rarely silent on synovitis) |
| ALLW (E1p) | Every slice as a window at inference, for a K = 8-trained v2 model | 0.8791 → 0.8797, Δ +0.0006 [−0.0016, +0.0027]; 3.5× windows | Not adopted; the deployed shift bag stays (Δ +0.0017 [+0.0001, +0.0033]) |
| Cache hr | 13,221 series at 0.365 mm/px (140 mm field, centre 352 px) | 50.6 min, 0 failures, 2 end slices dropped and logged | Ready for E9 |
| EFF windows | 4 / 6 windows per slot instead of 8 (E1p, one bag) | 0.8585 (Δ −0.0206 [−0.0252, −0.0161]) / 0.8743 (Δ −0.0048 [−0.0083, −0.0015]) | Rejected: the network costs about 0.03 s/study; the AUC loss is worth 6–24 min under the efficiency formula |
| EFF index | Fast index vs full header pass (50 studies, 4 workers) | 0.97 → 0.28 s/study; preprocessing 1.65 → 0.96 s/study | Adopted (see §3) |
| Seed reference | E1s2 (seed 2) in the deployed setting | 0.8870 vs E1p 0.8808 (seed spread 0.006); E1p + E1s2 = 0.8880 | Reference for E9 |
| S1 (public LB) | Does the HR member add to the 0.927 ensemble? (F1–F3 1/6 each + E9 1/2) | **0.928** vs 0.927, Δ +0.001 (inside the ±0.002 noise band) | HR kept (pre-registered middle band; its pseudo-val rule passed). No measurable LB gain |
| A-strict (public LB) | Does the owner's own run reproduce heliosli's fastMRI-free public recipe (nartaa CoAtNet 384 + S6, 70/30)? | **0.954** (submission 57384567), the same as the recipe's page | Final pick (second) |
| A-strict+ (public LB) | Do v27's extra parts (nartaa 224 checkpoint, goodpjw2008 reader) add to that recipe? | **0.954** (submission 57384570); vs A-strict, \|Δ\| < 0.001 at three decimals | Final pick (main, by the plan §10c rule). No measurable gain. Since 2026-10-20: second final pick (plan §11) |
| final-strict (public LB) | Does nartaa's all-data S6 (s6fall, fastMRI-free) add to A-strict as a third leg (Raptor 0.5 / S6 0.25 / s6fall 0.25)? | **0.956** (submission 57406523, seen 2026-10-20 05:37 IST); vs A-strict +0.002 at three decimals (roughly 0.001–0.003). Predicted 0.955–0.956 | **Final pick (main, by the plan §11 rule)**; A-strict becomes the fallback |
| G58 upgrades (gold, pre-registered) | Do any of 6 blend upgrades (A1, B, C1, C2, D1, D2) lift final-strict's blend (0.9361 on gold_58)? | None passes. Largest point gains: C2 (standardised logits) +0.0028 [−0.0011, +0.0071]; D1 (compartment weights) +0.0021 [+0.0000, +0.0047], 5/12 findings up | Not adopted (plan §12b) |
| B on oof_300 (S6 held-out folds) | Does mirror TTA hold up when each study is scored by the S6 fold that never trained on it? | 0.9180 → 0.9113, Δ −0.0066 [−0.0107, −0.0026] (300 studies, labels_v3 confident cells) | Condition not met; B rejected (plan §12c) |
| E adapter (gold, 5-fold out-of-fold) | Does a gold-calibrated residual adapter on the S6 and s6fall class tokens help? | 0.9364 → 0.9364, Δ +0.0000 [−0.0003, +0.0004]. The fixed L2 weight kept it at the identity (largest weight 0.0005) | Not adopted, and not re-run with other settings (no second look at gold). No final-strict-plus (plan §12c) |

### E9: high-resolution input (the genuinely different representation)

**Configuration.** E1p with:
- the hr cache (140 mm field on a 384 px grid = 0.365 mm/px; centre 352 px stored);
- random 320 px crops in training and the centre 320 px at inference;
- K = 6 random windows per slot in training (18 per study);
- batch 1 × accumulation 4 (same effective batch of 4 studies);
- 8 epochs.

**Cost.** 105 min training + 18 min validation on the RTX 4050; peak 3.09 GB.

**Curve vs E1p** (pseudo-val; E9 at K = 12, E1p at K = 8):

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| E9 | 0.748 | 0.804 | 0.842 | 0.862 | 0.874 | 0.881 | 0.884 | 0.885 |
| E1p | 0.767 | 0.822 | 0.846 | 0.860 | 0.870 | 0.876 | 0.878 | 0.879 |

E9 starts behind and overtakes E1p from epoch 3.

**Inference setting** (pre-registered rule §7 of the plan):

| Setting | Macro AUC |
|---|---|
| K = 12 | 0.8848 |
| **K = 12 + shift** | **0.8856** (+0.0008) |
| Every slice | 0.8846 (−0.0002) |
| Full 352 px field (128 mm) | 0.8848 (−0.0007 [−0.0016, +0.0001]) |

**Chosen:** K = 12 + shift, 320 px.

**Paired vs E1p, deployed setting (0.8808):**
- E9 is **+0.0048 [+0.0005, +0.0090]**.
- Confident-label AUC 0.9290 vs 0.9231.
- Gold guard: 0.8962 vs 0.8908.
- The pre-registered HR rule (> +0.004) **passes narrowly**.
- A second v2 seed alone (E1s2) scores 0.8870, so as a *single* model HR is within seed noise.

**Ensemble value:**
- E1p + E9: **0.8889**.
- E1p + E1s2: 0.8880.
- Prediction correlation with E1p: 0.934 for E9 vs 0.958 for the second seed.
- HR is a genuinely diverse member.

**Per finding (pseudo-val, E9 − E1p):**

| Gains | Losses |
|---|---|
| Lateral meniscus **+0.033** | ACL −0.011 |
| Synovitis +0.020 | Fracture −0.009 |
| Medial meniscus +0.017 | Baker's −0.005 |
| MCL +0.014 | PF OA −0.004 |
| Medial OA +0.008 | |

The gains are thin structures that need detail. The losses fit the visual QA: the 117 mm field clips the patella and posterior soft tissue in some series.

**Decision.** HR is adopted as an **ensemble family next to v2**. Overnight HR students then ran (§2). Release S1 (F1–F3 1/6 each + E9 1/2) scored **0.928** on the public LB (reference 0.927; see the S1 row above).

## 2. Overnight runs (all finished by 2026-10-19 10:13 IST)

- OOF teachers done. Pseudo-val on each model's own held-out fold:

  | Fold | Model | Pseudo-val |
  |---|---|---|
  | 0 | E1p | 0.8791 |
  | 1 | O1 | 0.8757 |
  | 2 | O2 | 0.8820 |
  | 3 | O3 | 0.8823 |
  | 4 | O4 | 0.8779 |

- **Pseudo-labels built 23:49** (`pl_g_r1.csv`, control `pl_u_r1.csv`). Sanity checks passed:
  - 4,407 rows, no NaN;
  - the five teacher-only columns are identical to labels_v3;
  - gold rows are unchanged (fold −1, never trained on);
  - OOF macro AUC vs teacher 0.8789;
  - 1.6–6.3% of cells per mixed finding cross 0.5 (71–273 studies).
- Students, all completed (start → finish, from their run folders):
  1. SHR_PLg_all_s42 (HR): 23:49 → 02:03;
  2. S224_PLg_all_s42 (v2): 02:04 → 03:55;
  3. SHR_rep_all_s42 (HR on report labels, the control): 03:55 → 07:10;
  4. SHR_PLg_all_s7 (HR): 07:11 → 10:13.

  The last two ran slower than the 5.1 studies/s measured at the start (HR epochs about 16 → 24 min) because the laptop GPU throttled thermally.
  - All four were packaged with F1–F3 and E9 as release `rsna-knee-v3-all` (8 checkpoints, SHA256SUMS verified). Its standalone script passed the 3-study local test.
  - Not uploaded and never LB-tested, so their effect is unknown.
- S1 scored 0.928 (§1). The single-model A/Bs S2a–S2d (plan §9) were not run: plan §10c removed their role in the final picks.
- The two A variants were submitted at 11:19 IST and both scored 0.954 (§1).

**Clean runtimes (measured 21:00, idle GPU, 50 studies; details in the efficiency report).**
- 0.927 release: 41.0 min per 1,300 studies as submitted, **30.6 min with the fast index**.
- S1: 39.1 min.
- F1 alone, one bag: 23.3 min.

**Pareto plot:** `reports/eda/emergency/pareto_accuracy_runtime.png`, from measured per-study costs.

## 3. Engineering verified this session

| Change | Verification |
|---|---|
| `data.py` crop / all-window / window-mask support | Old configuration reproduced exactly: identical eval and train tensors (same RNG draws), logits diff 0.0 vs the frozen 0.927 code; padding invariance 4.8e-7 |
| `model.py` per-window mask | Same as above |
| `predict.py` hr decode, crop, `infer_K`, ensemble weights, `fast_index` | Emergency notebook run end-to-end on a mixed v2 + hr release (3 local test studies): ALL OK, valid submission |
| Standalone inference script (`notebooks/rsna_knee_infer_standalone.py`) | Same predictions as the notebook (max diff 0.0); private package directory tested with the pure-Python wheels |
| Fast index (`fast_index`, one header per series) | Identical slots and steps to the training index on 400 studies and to the full inference pass on 50; side identical to the full inference pass (2 ambiguous studies differ from the *training* index, as before); identical submission on the local test studies |
| One-decode preprocessing (`preprocess_series_multi`, used automatically when v2 and hr models share slots) | v2, v6 and hr arrays bit-identical to the three training caches on 12 series (single and multi paths); mixed-release submission identical to the two-decode run (max diff 0.0) |
| Model selection in the notebook (`SELECT_MODELS`) | Selecting F1 from the S1 folder = an F1-only folder (max diff 0.0) |
| Emergency notebook vs the 0.927 notebook (F1–F3 only) | Same predictions (max diff 6e-8, float rounding of the weighted mean) |
| 0.927 notebooks and release folder | Unchanged (SHA-256 identical); packaging guard added |

## 4. Incidents

- **2026-10-18 19:30–20:10: fold-2 teacher (O2) stalled for about 20 min.**
  - Cause: local notebook tests ran on the 6 GB GPU at the same time as training (5.46 GB in use).
  - Windows (WDDM) then spills GPU memory to system RAM and both jobs crawl.
  - Fixed by stopping the test process; O2 resumed. No result was affected; O2 finished later than planned.
  - Rule since then: GPU tests run only in gated queue gaps.
