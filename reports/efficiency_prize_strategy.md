# Efficiency Prize strategy

Status: **draft, 2026-10-18 15:20 IST**. Measurements are added as they are made, each labelled **measured (local)**, **estimate** or **official**.

## 1. Official rules (read verbatim from the Overview page, 2026-10-18 09:10 UTC)

> We compute a submission's efficiency score by
> **Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400**
> where AUC is the submission's score on the main competition metric, Benchmark is the score of the benchmark `sample_submission.csv`, max AUC is the maximum AUC of all submissions on the Private Leaderboard, and RuntimeSeconds is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.

| Item | Official answer |
|---|---|
| Main metric | Macro ROC-AUC over 12 findings |
| Benchmark | Score of `sample_submission.csv`. Not printed on the pages read. A constant submission has AUC 0.5 for every finding, so Benchmark ≈ 0.5 (**inference**, not verified). |
| Max AUC | Best **private** LB score: unknown until the end (public #1 is 0.969 on 2026-10-18). |
| Runtime | "the number of seconds it takes for the submission to be evaluated" on Kaggle hardware for the hidden test set. CPU and GPU notebooks are both allowed (≤ 9 h each). |
| Eligibility | (1) the submission must be among the team's selected final submissions, or auto-selected; (2) it must rank above the `sample_submission.csv` benchmark on the private LB |
| Both prizes? | Yes: "A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize." |
| Selection restriction | Only 2 final selections, so an efficiency candidate costs one of the two picks unless it is also the main pick |
| Efficiency LB | Public-data **ranks only**, updated daily (`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`) |

## 2. What the formula rewards (analysis)

With Benchmark ≈ 0.5 and max AUC ≈ 0.97, the first term is −AUC/0.47:

- **+0.001 AUC ≡ −69 s of runtime** (68–70 s for a max AUC of 0.965–0.975).
- A 9 h run adds +1.0, so the formula strongly punishes long runs.

| Candidate (AUC, runtime) | Efficiency (max AUC 0.97) |
|---|---|
| Owner's current submission (0.927, about 45 min, estimate) | −1.889 |
| Single model 0.920 at 15 min | −1.930 |
| 0.935 at 10 min | −1.971 |
| 0.947 at 20 min | −1.978 |
| nartaa-like 0.945 at 7 min (their efficiency notebook reports a 7.2 min submit-to-score turnaround, queue included) | −1.998 |
| Public 0.958 stack at 3 h | −1.705 |

Third-party context (self-reported, discussion 735304): the efficiency-LB leader runs a small ResNet and reports 0.947 public LB with 5 folds.
- Winning a top-3 efficiency prize realistically needs **AUC ≥ 0.945 at ≤ 10 min**.
- At our current accuracy (about 0.93), no runtime reduction is enough. Accuracy improvements (HR / PL) are a prerequisite, not an alternative.

## 3. Where our runtime goes (measured, local RTX 4050 + i5-13420H, 4 workers)

The 50-study end-to-end benchmark (2026-10-18, before this mission) for the 3-model, 2-bag release:

| Stage | Seconds (50 studies) | Per study |
|---|---|---|
| Preprocessing: header pass over all series (about 184 files per study) + decode of 3 selected series (about 106 files) | 76.3 | 1.53 s |
| Prediction: 3 models × 2 bags × 24 windows | 19.4 | 0.39 s (0.065 s per model-bag) |
| **Total** | 95.7 | **1.91 s → about 41 min for 1,300 studies** |

**About 80% of the runtime is DICOM I/O and decoding**, not the network. The levers, in order:
1. **Index pass.** Read one header per series instead of all of them. Plane and fat-suppression come from `test_series.csv`, and slice counts from the directory listing. Implemented as `fast_index`; measurement and index-agreement check pending.
2. **One model, one bag.** Prediction 0.39 → 0.065 s/study.
3. **Workers.** Use every CPU core of the Kaggle machine for decoding.
4. **Not pursued: decoding only the slices the windows need.** It saves about 20% with K = 8 but changes the per-series intensity percentiles the model was trained with, and needs a fresh validation.

**Measured (local, 2026-10-18 15:50, 50 fold-0 studies, 4 workers; training ran on the GPU at the same time, so these are upper bounds):**

| Stage | s (50 studies) | per study |
|---|---|---|
| Full header pass (current: every header of every series, 9,188 files) | 48.4 | 0.97 |
| **Fast index** (one header per series, file counts from the directory listing) | 14.1 | **0.28** |
| Decode + preprocess of the selected series, v2 (224 px) | 33.9 | 0.68 |
| Decode + preprocess of the selected series, hr (0.365 mm/px, 352 px) | 32.9 | 0.66 |

- **The fast index gives the identical study index**: 0 mismatching cells in selected series UIDs, 2.5D steps and knee side on all 50 studies.
- It cuts preprocessing from 1.65 to 0.96 s/study (−42%).
- hr decoding costs the same as v2.
- Caveat: the fast pass ran second, so its file headers were partly in the OS cache. The 14.1 s may be optimistic, which a cold-cache repeat should check.

**Windows vs accuracy (measured, E1p on pseudo-val fold 0, paired vs K = 8):**

| Windows per slot (one bag) | Windows per study | Pseudo-val macro AUC | Δ vs K = 8 [95% CI] |
|---|---|---|---|
| 4 | 12 | 0.8585 | −0.0206 [−0.0252, −0.0161] |
| 6 | 18 | 0.8743 | −0.0048 [−0.0083, −0.0015] |
| **8 (training setting)** | 24 | **0.8791** | — |
| 8 + shifted bag (deployed TTA) | 48 | 0.8808 | +0.0017 [+0.0001, +0.0033] |
| every slice (about 84) | 84 | 0.8797 | +0.0006 [−0.0016, +0.0027] |

The network costs about 0.03 s/study per bag, so cutting windows saves seconds per 1,300 studies but costs 0.005–0.021 AUC, which the formula values at 6–24 minutes. **Keep K = 8. The efficiency lever is preprocessing (the fast index), not the network.**

**Clean end-to-end benchmark (measured 2026-10-18 21:00, idle GPU, 50 fold-0 studies, 4 workers; the files were partly in the OS cache from earlier runs, so cold-disk Kaggle runs will be slower):**

| Release | Preprocess (s) | Network (s) | s/study | min / 1,300 studies |
|---|---|---|---|---|
| 0.927 release as submitted (full index, 3 models × 2 bags) | 73.8 | 20.7 | 1.89 | 41.0 |
| 0.927 release + **fast index** | 50.3 | 20.4 | 1.41 | **30.6** |
| S1 (+ E9 hr, fast index, one decode for v2 + hr) | 50.7 | 39.4 | 1.80 | 39.1 |
| F1 alone, one bag, fast index | 47.2 | 6.6 | 1.08 | 23.3 |

Reading the table:
- The fast index cuts the release's preprocessing by 32% on this cleaner run, after −42% on the contended run.
- One-decode makes the HR member's decode nearly free (+0.4 s total).
- S1 costs no more than the release that scored 0.927. It scored 0.928 on the public LB (Δ +0.001, inside the ±0.002 noise band).
- Under the efficiency formula +0.001 AUC is worth about 69 s, while S1 costs about 8.5 min more than the 0.927 release with the fast index (local measurement, not Kaggle timing). S1 is therefore not an efficiency candidate.

**GPU forward cost (measured, idle RTX 4050, autocast bf16, random inputs, ms per study, 3 slots):**

| Encoder | px | Windows per slot | ms per study | Peak GB |
|---|---|---|---|---|
| ConvNeXt-T | 224 | 8 | 38.3 | 1.17 |
| ConvNeXt-T | 224 | 4 | 19.6 | 0.65 |
| ConvNeXt-T | 320 | 12 | 124.4 | 0.92 |
| ConvNeXt-T | 320 | 6 | 65.8 | 0.92 |
| ConvNeXt-T | 320 | 34 (every slice) | 381.7 | 2.38 |
| ConvNeXt-nano | 224 | 8 | 32.9 | 0.98 |
| EfficientNet-B0 | 224 | 8 | 14.6 | 0.84 |

Even the heaviest member (HR every slice, 0.38 s/study) costs less than the DICOM preprocessing (about 1.0 s/study). A smaller encoder saves at most about 0.02–0.03 s/study, about 1 min per 1,300 studies: worth less than 0.001 AUC under the formula.

## 4. Candidates (filled 2026-10-19 04:32 IST)

Efficiency is computed with Benchmark ≈ 0.5 and max AUC ≈ 0.97 (estimates); lower is better.

| Candidate | Accuracy evidence | Runtime | Efficiency estimate |
|---|---|---|---|
| A: main leaderboard model (A-strict / A-strict+, public OAI-trained pipeline; plan §10c) | 0.954 displayed on heliosli's notebook (third-party run); owner's run pending | 6–8 h (third-party, v27-type stack); not measured | −1.36 to −1.14 |
| B: efficiency model (F1 alone, one bag, fast index) | not scored; expected ≤ 0.927 | 1.08 s/study, 23 min per 1,300 studies (local) | about −1.93 at 0.927 |
| C: fallback (0.927 release) | 0.927 public LB (measured) | 1.89 s/study, 41 min (local) | about −1.90 |

None of these can win the efficiency prize (see the official leaderboard section below). Both final picks therefore go to candidate A variants.

## 5. Caveats

- Local runtime is not Kaggle runtime: different CPU core count, T4 vs RTX 4050, cold disk, and possibly compressed DICOMs (JPEG-Lossless / JPEG 2000) in the hidden test. Compressed DICOMs would make decoding slower than measured here.
- The only official runtime is the scored run. The efficiency LB shows ranks, not seconds.

## Official efficiency leaderboard (read 2026-10-19 02:40 IST; Kaggle's pinned notebook output `full_leaderboard.csv`, computed 2026-10-17 21:15 UTC)

- 6,046 teams. The file gives the efficiency rank and public AUC, but **no runtimes**.
- **Top 10 by efficiency** have public AUC 0.955–0.965. The overall #1 team (0.965) is 4th; #1 is 0.964.
- **Top 50** have public AUC 0.943–0.965 (median 0.9515).
- **Our best is 0.928** (S1).
  - Under the formula, 0.02 AUC is worth about 23 min of runtime, more than our whole notebook (about 25–40 min locally).
  - The best-placed entries at our AUC level (0.927–0.929) sit at efficiency ranks 152–170 (measured, same file). The efficiency rank does not let a short runtime close a 0.03 AUC gap in practice.
- **Conclusion (measured, not estimated): the efficiency prize is out of reach for our own models.**
  - A fast public OAI-trained model (e.g. nartaa's 224 model, 0.945 at a self-reported 7.2 min) might place higher, but where cannot be computed: the file has no runtimes. It would not be our work, and an efficiency pick costs one of the two final picks (estimate: low odds of a prize).
  - No further effort goes to the efficiency track. Candidate B stays documented (the fast-index single model) but is not one of the two final picks.
