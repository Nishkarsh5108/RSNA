# Emergency initial assessment (2026-10-18, 15:00 IST)

Purpose: where the 0.927 submission stands, what the official rules say, what the leaderboard and public solutions show, and the plan for the remaining 4.5 days. Every number is labelled **measured** (on this machine), **official** (competition pages), or **third-party** (self-reported by other competitors, not verified).

## 1. Deadline and rules (official, read 2026-10-18 09:10 UTC from the Overview page via a reader proxy)

| Item | Finding |
|---|---|
| Final submission deadline | **2026-10-22 23:59 UTC** = 2026-10-23 05:29 IST. Entry and team-merger deadlines (Oct 15) have passed. |
| Main metric | Macro ROC-AUC over 12 findings |
| Leaderboard | Public LB = about 30% of the test data; final = the other 70% |
| Code requirements | CPU or GPU notebook ≤ 9 h; internet off; "freely & publicly available external data is allowed, including pre-trained models" |
| Submissions | 5 per day; 2 final selections |
| Efficiency score (verbatim) | **Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400**, to be minimised. Benchmark = score of `sample_submission.csv`; max AUC = best private-LB AUC; RuntimeSeconds = time to evaluate the submission. |
| Efficiency eligibility | The submission must be one of the team's selected (or auto-selected) final submissions and rank above the sample-submission benchmark on the private LB. One submission may win both tracks. |
| Efficiency LB | `ryanholbrook/rsna-knee-abnormalities-efficiency-lb`, public-data ranks only, updated daily |

**What the formula implies.** The benchmark (constant predictions) scores about 0.5 and max AUC is about 0.97, so the denominator is about −0.47.
- One AUC point (0.001) is worth −0.0021 of efficiency score, the same as **about 69 s of runtime**.
- A 9 h run costs +1.0, which wipes out a 0.47 AUC advantage, so heavy ensembles are heavily penalised.
- These are estimates: the private max AUC is not known yet.

## 2. Baseline (fallback C)

- **Score:** 0.927 public LB, as reported by the owner. The scored notebook version cannot be read from here (no Kaggle credentials), so the configuration below is presumed from the packaged release.
- **Presumed configuration:** F1 + F2 + F3, each a ConvNeXt-T 2.5D MIL (3 fat-suppressed-first slots, K = 8 three-slice windows per slot, 224 px squash ≈ 0.71 mm/px, per-finding attention). Trained on all 4,349 non-gold studies with labels_v3, last-epoch EMA weights. Equal-weight mean with half-step shifted-window TTA.
- **Frozen copy:** `D:/rsna_work/fallback_0927/`, read-only, holding:
  - the Kaggle dataset folder: 3 checkpoints and 13 wheels;
  - both notebooks and the notebook builder;
  - all `src/imaging` sources;
  - `SHA256SUMS.txt` with 42 files.
- **Checkpoint SHA-256:**
  - F1 `c227fc1e…7ceaf`
  - F2 `ba3f3a64…607e`
  - F3 `70c5c5c2…24d8`
- **Measured validation, not comparable with each other:**
  - pseudo-val single models 0.879–0.885, 2 seeds 0.888;
  - gold-58 0.9065 [0.878, 0.931].
- **Measured per finding (gold, ensemble):** synovitis 0.680, lateral meniscus 0.850, PF OA 0.855; every other finding 0.896–0.977.

## 3. Leaderboard and public solutions (read 2026-10-18 09:12 UTC)

| Position | Public LB |
|---|---|
| #1 (RB & Matheus & Tucker & Tony) | 0.969 |
| Gold zone (about top 22) | ≥ 0.964 |
| Silver, from about #23 | 0.962–0.963 |
| **Best public notebooks** (goodpjw2008 "CoAtNet + R6 + 2.5D MIL", heliosli blends) | **0.958** |
| Owner (rank ≈ 3,747) | 0.927 |

These leaderboard figures are official and come from the leaderboard and code pages.

**Third-party findings that change the plan:**

1. **Public weights.** Since about 2026-10-07, every public notebook at 0.950 or above uses `nartaa/rsna-knee-publication-swa-weights-20261007`. This is one CoAtNet trained with **2,399 external OAI knee MRIs** (nartaa's post 746792; TranBaDat2607 repo).
   - OAI is controlled-access data: NIMH Data Archive, data use certification required.
   - The hosts had not ruled on OAI as of the latest reports (discussion 743416).
   - Without OAI, nartaa's own CoAtNet scored 0.939–0.942, and the best "clean" public blend reported is 0.945.
2. **Input geometry of the strong public models.** 140 mm field on a 384 px grid (**0.365 mm/px**), centre 320 px crop at inference, and **every slice used as a window** (94 windows per study).
   - nartaa measured 94 → 24 windows as **−0.006** on the LB.
   - Our model sees 0.71 mm/px and 24 windows per study.
3. **Labels are named as the main lever** by almost every strong single-model report (0.950–0.959). Pseudo-labels from out-of-fold image models are mixed (≥ 50%) into the report labels.
   - One careful counter-example (TranBaDat2607) saw ≤ +0.003 from this with weaker models.
   - About 34% of studies whose report is silent on synovitis are image-positive (discussion 733932, third-party).

## 4. The three most consequential bottlenecks (preliminary; deep dive in `current_model_mathematical_deep_dive.md`)

| Rank | Bottleneck | Evidence |
|---|---|---|
| 1 | **Labels: report-derived targets vs image-read test labels** | Measured: teacher gold macro 0.890; the image model already beats its own labels on gold (0.9065). Synovitis: every report-derived label 0.68–0.74 on gold. Third-party: OOF pseudo-labels named as the lever by most 0.95+ single models. |
| 2 | **Input resolution and slice coverage** | Measured: v2 is 0.62–0.86 mm/px (median 0.71). 8 windows per slot cover 24 of about 106 slices per study. Third-party: public 0.949 model uses 0.365 mm/px and all slices; 94 → 24 windows was −0.006 LB. Our E8 test (0.6 mm/px) was too small a step to decide this. |
| 3 | **Single recipe and limited diversity** | Measured: seed-pair correlation 0.95; every architectural variant within noise. Third-party: blends of different pipelines and labels are what lift public notebooks. |

## 5. Plan (in execution; rules pre-registered in `emergency_experiment_plan.md`)

| When (IST) | GPU (RTX 4050 6 GB) | CPU / other |
|---|---|---|
| Oct 18 15:00–21:30 | OOF teachers O1–O4 (E1p recipe, folds 1–4; fold 0 = E1p) | High-res cache (0.365 mm/px, centre 352 px) for 13,221 series, ETA about 65 min. Initial reports. |
| Oct 18 about 16:30 | **HR fold-0 experiment** (inserted after O1) | — |
| Oct 18–19 night | Pseudo-label students (224 and/or HR, per the HR result) | Inference support for hr / all-window. Efficiency benchmark. |
| Oct 19 (Mon) | Optional professor GPU (7 h): HR students / seeds (plan in `blackwell_gpu_plan.md`) | LB submissions: one question each |
| Oct 20–21 | Final seeds; blends; efficiency candidate | Package and test notebooks |
| Oct 22 | **Buffer only**: final submissions by about 18:00 UTC, picks selected | — |

## 6. Decisions the owner must make

1. **OAI-trained public weights.** Reaching about 0.958 quickly means building on the public notebooks that use nartaa's OAI-trained CoAtNet.
   - Rules allow freely and publicly available models. OAI itself is controlled-access, the hosts have not ruled, and an adverse ruling could void such a submission.
   - A clean public stack (competition-data-only checkpoints) reportedly reaches about 0.945.
   - Recommendation: if public weights are used at all, keep **one clean pick** (own models plus clean public models) as one of the two final selections.
2. **Repository exposure (verified again 2026-10-18 15:00 IST).**
   - `github.com/Nishkarsh5108/RSNA` is **public** (`"private": false`).
   - It serves `data/main_data/train (1).csv`, the full competition `train.csv` with all reports and gold labels, plus other competition files.
   - This very likely breaches the data-security rule.
   - Safe remediation: `reports/repo_remediation_plan.md`. Option A (make the repository private in GitHub settings) takes one minute and is reversible. Nothing has been changed on the remote.
3. **Optional enabler.**
   - Problem: without a Kaggle API token on this machine I cannot download public label tables or public checkpoints, so I cannot measure blend weights or correlations offline. Every public-model question has to be settled by LB submissions.
   - Fix: the owner places a token in `%USERPROFILE%\.kaggle\kaggle.json`.
   - Constraint: it would be used only to download public datasets.
