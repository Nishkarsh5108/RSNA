# Emergency experiment plan (2026-10-18 → 2026-10-22)

Status: **living document**; every rule below was written **before** the result it governs.

## 0. Evaluation channels and what each may decide

| Channel | Labels | n | Can judge | Cannot judge |
|---|---|---|---|---|
| Pseudo-val (fold 0) | report-derived teacher (labels_v3), binarised at 0.5 | 870 | architecture / input changes trained on the **same** labels (paired bootstrap) | label changes (biased toward the report teacher), synovitis (report blind spot) |
| Gold-58 | image read by radiologists | 58 (SE ≈ 0.02 macro) | large regressions; pre-registered hypotheses only | small differences (two seeds differ by 0.013) |
| Public LB | image read (hosts: 2 MSK radiologists + adjudicator, severity thresholds) | ≈ 30% of ≈ 1,300 | label / post-processing changes, final picks | anything at ±0.002 (rounded to 3 decimals; noise) |

Gold-58 history: scored once on 2026-10-18 04:15 (`GOLD_UNSEALED.json`). Any further gold look is listed in §3 *before* it is made.

## 1. Baseline (fallback C)

- Submission scored **0.927** public LB (reported by the owner). Presumed configuration: the packaged release
  `D:/rsna_work/kaggle_upload/rsna-knee-image-model` (F1 + F2 + F3, ConvNeXt-T 2.5D MIL, v2 224 px, K = 8, attn12,
  shift TTA). Frozen copy with SHA-256 list: `D:/rsna_work/fallback_0927/` (read-only attribute set).
- The exact Kaggle notebook version that produced 0.927 cannot be read from this machine (no Kaggle credentials);
  the owner should keep that notebook version untouched as fallback **C**.

## 2. Experiments, in priority order

| ID | Hypothesis (evidence) | Decided by | Pre-registered rule |
|---|---|---|---|
| H-SYN1 | Image-read synovitis on non-contrast MRI is graded with effusion (MOAKS "effusion-synovitis"); gold labels: corr(Syn, Eff) = 0.40; our synovitis head is 0.680 on gold. A rank average of the synovitis and effusion outputs ranks image-read synovitis better. | one gold look (guard) + one LB submission | adopt if LB ≥ baseline + 0.003 (≈ +0.036 synovitis AUC) and gold synovitis AUC does not fall |
| HR | 0.365 mm/px, 140 mm field, 320 px crop (public 0.949 CoAtNet geometry), train K = 6/slot, evaluate every window, beats v2 224 px squash (0.71 mm/px) | pseudo-val fold 0, paired vs E1p | adopt if Δ > +0.004 and confident-label AUC not lower (rule 1 of the previous protocol) |
| PL | Pseudo-labels: students trained on a blend of report labels and out-of-fold image-model predictions beat report-only students (community reports; our image model already beats its own labels on gold: 0.9065 vs 0.890) | LB (+ gold guard) | adopt if LB ≥ comparator + 0.003 and gold macro not lower by > 0.01 |
| ALLW | Evaluating every window (instead of K = 8) helps a model trained with few random windows (public: 94 → 24 windows cost −0.006) | pseudo-val | adopt if Δ > +0.002 (deterministic inference change: validation noise only) |
| EFF | Efficiency candidate: single model, fewer windows, decode only needed slices | measured runtime + pseudo-val | choose the minimum of AUC/(0.5 − 0.97) + T/32400 using measured local T × Kaggle factor |

## 3. Gold-58 looks (pre-registered)

1. **H-SYN1** (registered 2026-10-18 15:00 IST, before computing): on the sealed F1+F2+F3 gold predictions (no TTA),
   compare synovitis AUC of (a) the synovitis output and (b) mean of the within-set ranks of synovitis and effusion.
   No weights are fitted. Reported regardless of outcome.
   **Result (15:02):** (a) 0.680, (b) 0.702, Δ +0.022 [−0.043, +0.093]; effusion output alone 0.682. **Not adopted**
   (expected macro gain ≤ +0.002, below LB resolution; not worth a submission).
2. **H-SYN2** (registered 2026-10-18 15:10 IST, before computing; motivated by stevenleehans' pre-registered result,
   discussion 733932: filling report-silent synovitis cells from the effusion field raised synovitis label AUC
   0.678 → 0.790): *label-level* check. Synovitis label AUC on gold of (a) the final teacher and (b) the same teacher with
   synovitis := effusion label where our labeller's order-averaged assertion for synovitis is "not mentioned".
   Rule: use (b) in the training labels if its gold synovitis AUC exceeds (a) by ≥ 0.02 (a training-target decision;
   any model trained on it is still judged on the LB).
   **Result (15:14):** genuine teacher gold predictions rebuilt from its sources (macro 0.889, matching the label report).
   Synovitis (a) 0.756, (b) 0.738, Δ −0.018. Our labeller marks synovitis "not mentioned" in only 25% of reports
   (stevenleehans' labeller: 84%), and 7 of the 14 silent gold cells are positive. **Not adopted.**

3. **E9 regression guard** (registered 2026-10-18 18:05 IST, before E9 finished): E9's sealed gold predictions
   (fold-0 model, its validation K = 12, no TTA) vs E1p's (0.891, scored 2026-10-18 04:15). Used only as a guard:
   if E9's gold macro AUC < 0.871 (E1p − 0.02), HR is not used regardless of pseudo-val. No selection among
   HR variants is made on gold.
   **Result (18:40):** E9 gold 0.8962 vs E1p 0.8908 → guard passed.

## 4. Pseudo-label target definition (registered 2026-10-18 15:16 IST, before any OOF result)

- OOF source: fold 0 = E1p (holdout fold 0), folds 1–4 = O1–O4 (same recipe, index v2). Each study's OOF prediction
  comes from the model that did not train on it.
- **PL-g (primary):** y = 0.5 · teacher + 0.5 · OOF for the findings where the image ensemble already beats its report
  labels on gold (medial meniscus, medial OA, lateral OA, effusion, Baker's, contusion, fracture: model − labels
  +0.009 to +0.109); y = teacher for ACL, MCL, lateral meniscus, PF OA and synovitis (labels better on gold by
  0.015–0.076). The split uses the per-class gold numbers already published on 2026-10-18 04:15 (no new gold look);
  it is flagged as gold-informed.
- **PL-u (control, only if a submission slot is free):** y = 0.5 · teacher + 0.5 · OOF for all 12 findings.
- Students are trained on all 4,349 non-gold studies (no holdout), so they are judged on the LB against the matching
  report-label model (same recipe, same seed); the pseudo-val number of a fold-0 student is reported but is biased
  toward the report teacher.

## 5. Results log (appended as they arrive)

- **ALLW on E1p (15:30):** pseudo-val fold 0, paired vs the training window setting (K = 8): all windows (83.9 per study)
  0.8797, Δ +0.0006 [−0.0016, +0.0027]; deployed shift bag 0.8808, Δ +0.0017 [+0.0001, +0.0033] (reproduces the earlier
  TTA measurement exactly). **ALLW not adopted for K = 8-trained v2 models**; it is re-tested on the HR model, which is
  trained on fewer windows (K = 6) like the public model it copies.
- **S1 on the public LB (2026-10-19, submitted by the owner):** 0.928 vs 0.927 for the reference release (Δ +0.001,
  inside the ±0.002 noise band). Rule §6: middle band, so HR stays because its pseudo-val rule passed (§8). The LB
  neither confirms nor refutes the pseudo-val ensemble gain (+0.008 for E1p + E9 over E1p). E9 is a fold-0 model
  (80% of the training studies); the next HR check uses all-fold models: S2c vs S2a (§9).

## 6. Leaderboard submissions and their rules (registered 2026-10-18 17:45 IST, before any of these results)

Public LB scores are rounded to 3 decimals and cover about 390 studies. Differences of ±0.002 are treated as noise.
The reference is the 0.927 release (F1 + F2 + F3).

| # | Release | Question | Rule |
|---|---|---|---|
| S1 | F1 + F2 + F3 (v2, shift TTA) + E9 (hr, fold-0 model, best pseudo-val inference setting), equal weights | Does a high-resolution member add on image-read labels? | HR members stay in the final ensemble if S1 ≥ 0.929; if S1 ≤ 0.925, HR is dropped; in between, HR is kept only if its pseudo-val rule (HR) or diversity rule passed |
| S2 | PL-g students (same recipe as the final members they would join), all folds | Do pseudo-labels help on image-read labels? | Adopt PL-g targets for the final members if S2 ≥ the matching non-PL comparator + 0.003; otherwise report labels stay |
| S3+ | Final candidates A (main) and B (efficiency) | — | A = best LB among releases built only from rules above; B chosen by the efficiency estimate (§ efficiency report) |

No per-finding weights and no ensemble weights are fitted on the LB. At most two weight settings are ever compared per question.

## 7. Inference setting for HR checkpoints (registered 2026-10-18 18:00 IST, before E9's evaluation)

HR checkpoints are trained with K = 6 random windows per slot, so their inference setting must be set explicitly
(`infer_K` in the packaged args; the default would fall back to the training K = 6). Candidates, scored on pseudo-val:
(a) K = 12 (one bag), (b) K = 12 + half-step shifted bag, (c) every slice (K = 0). Choose (c) if it beats (a) by > +0.002
(the ALLW rule); otherwise choose (b) if it beats (a) by > 0; otherwise (a).

## 8. E9 (HR) decision (2026-10-18 18:50 IST)

Paired vs E1p in its deployed setting (K = 8 + shift, 0.8808), pseudo-val fold 0:

| Setting | Macro AUC | Δ [95% CI] | Confident-label AUC |
|---|---|---|---|
| E9, K = 12 | 0.8848 | +0.0040 [−0.0006, +0.0084] | 0.9286 |
| **E9, K = 12 + shift** | **0.8856** | **+0.0048 [+0.0005, +0.0090]** | 0.9290 |
| E9, every slice | 0.8846 | +0.0038 [−0.0005, +0.0082] | 0.9283 |
| E1s2 (v2 seed 2), shift | 0.8870 | +0.0062 [+0.0026, +0.0098] | 0.9300 |
| E1p + E1s2 (two v2 seeds) | 0.8880 | +0.0072 | 0.9306 |
| **E1p + E9 (v2 + HR)** | **0.8889** | **+0.0081 [+0.0057, +0.0104]** | 0.9318 |

- **HR rule** (Δ > +0.004 and confident-label AUC not lower): **passed narrowly** (+0.0048; 0.9290 ≥ 0.9231).
  - Caveat: a second v2 seed scores +0.0062, so HR as a *single* model is within seed noise of v2.
- **Gold guard** (§3.3): passed (0.8962 vs 0.8908).
- **Diversity:** E9 correlates less with E1p (r 0.934) than a second v2 seed (0.958). E1p + E9 beats E1p + E1s2 (0.8889 vs 0.8880).
- **Per finding:**
  - E9 gains on lateral meniscus +0.033, synovitis +0.020, medial meniscus +0.017 and MCL +0.014.
  - It loses on ACL −0.011, fracture −0.009, Baker's −0.005 and PF OA −0.004. The 117 mm field clips the patella and posterior soft tissue in some series.
- **HR inference setting** (§7): K = 12 + shift (all-slice −0.0002 vs K = 12; shift +0.0008).
- **Decision:** HR is adopted as an **ensemble family next to v2**, not as a replacement. HR students are kept in the overnight queue.
- **S1 weights (changed before any LB result):** the two families get equal weight, F1–F3 1/6 each and E9 1/2. Equal per-model weights would give HR only 25%, too little to detect on the LB, while the pseudo-val evidence is for a 50/50 family mix.
- **Bug found and fixed:** the shifted inference bag used the training K (6) for HR checkpoints. Fixed in `predict.window_bags` before any HR release; v2 bags unchanged (verified).

## 9. Monday LB design (registered 2026-10-18 20:45 IST, before any student exists)

**Setup.** One upload, `rsna-knee-v3-all`: F1–F3, E9, the students, and optionally the v2 fold models. Each submission chooses its model through `SELECT_MODELS`.

**Single-model A/Bs.** All four models are seed 42, all folds, and identical except for the factor named:

| # | SELECT_MODELS | Answers |
|---|---|---|
| S2a | F1 (v2, report labels) | reference |
| S2b | S224_PLg_all_s42 (v2, PL-g) | PL effect at v2 = S2b − S2a |
| S2c | SHR_rep_all_s42 (hr, report) | HR effect, single all-fold model = S2c − S2a |
| S2d | SHR_PLg_all_s42 (hr, PL-g) | PL effect at hr = S2d − S2c |

**Rules (same thresholds as §6).**
- **PL adoption** requires S2b ≥ S2a + 0.003 **and** S2d ≥ S2c + 0.003.
  - Each separate gain is reported.
  - If only one holds, PL is used only in that family.
- **HR** keeps the ensemble role decided in §8 unless S2c ≤ S2a − 0.005.
- These answers fix the members of the final ensembles. Their weights are equal per family, so no weights are fitted on the LB.

## 10. Public OAI-trained pipeline + our models (registered 2026-10-19 01:15 IST, before any submission of it)

**Decision (owner, 2026-10-19 01:10 IST).** Use public weights trained with OAI data, disclose them, and keep one final pick without them. The host's test (discussion 743416) allows datasets that need only registration and a licence acknowledgement; OAI appears to pass. Residual risk: the open geo-restriction question (research report, compliance note).

**Base pipeline.** The best-scoring public notebook (0.958 public LB) in which every weight file has a verified origin: competition data, public label tables, or OAI. Any arm with an unknown or prohibited origin (e.g. KneeCoT, banned by the host) is removed; if that is not possible, the base falls back to nartaa's documented single-model notebook (0.949, OAI only).

**Two settings only** (§6: at most two weight settings per question):

| # | Submission | Built as |
|---|---|---|
| P0 | Base notebook as published | Reference; re-scored in the owner's account |
| P1 | Base + our ensemble | Per finding: 0.8 × rank(P0 output) + 0.2 × rank(ours), ranks scaled to [0, 1] over the test studies |

- "Ours" = the best own release available when P1 is built (S1 today; the Monday members once packaged). It runs as a separate process with its own packages (`rsna_knee_infer_standalone.py`), so the base notebook's environment is unchanged.
- **Rule:** P1 is the main pick if P1 ≥ P0 on the public LB; ties go to P1 (a different label source and preprocessing are expected to generalise at least as well). Otherwise P0.
- No other weight is tried. The 0.2 share is fixed before any result: our ensemble is about 0.03 AUC weaker than the base, so it gets a minority share (third-party: a 0.929 2.5D reader helped a CoAtNet at a 30% rank share).
- **Final picks:** (1) the winner of P0/P1, OAI-derived, disclosed; (2) our best clean release, without OAI-derived weights.

### 10a. Amendment (2026-10-19 01:30 IST, before any P submission; reason: new third-party evidence, no result of ours)

**Provenance of the 0.956–0.958 public legs** ([V] = read on the dataset page; [S] = TranBaDat's phase-3 notes, 2026-10-14):

| Leg | External data | Grade |
|---|---|---|
| nartaa CoAtNet 384 + 224 (`nartaa/rsna-knee-publication-swa-weights-20261007`, 586 MB) | OAI, 2,399 knees. Terms: research and educational use including this competition; non-commercial adaptation; a winning submission may be licensed CC BY-NC 4.0 | [V] |
| heliosli R7 (`heliosli/rsna-knee-compact-r7cnxfsss`; also inside the "R6 fe055406" bundle with nartaa's weights) | fastMRI auxiliary training; knee-joint anchors | [S] |
| heliosli S6 / nartaa s6fall | knee-centred views from heliosli's SKM-TEA ("KneeXNet") anatomy model | [S] |
| goodpjw2008 2.5D ConvNeXt reader | none (competition data, report labels) | [S] |

- OAI-only route: nartaa + reader, about 0.950 public (TranBaDat A2 [S]).
- **Owner decision (2026-10-19 02:20 IST): option A.** OAI + SKM-TEA, no fastMRI: the base is a public notebook built on nartaa's CoAtNets and heliosli's **R6 proper** (the older `fe055406` bundle without `fm_aux`, about 0.957 public). R7 and any fastMRI or fastMRI+ leg are excluded. The exact notebook, dataset versions and bundle hash are verified from the code before use.
- **Rulings (checked 2026-10-19 01:25 IST):** none since the host's general test. Open participant threads ask about each leg (nartaa OAI, R7 fastMRI, the SKM-TEA anatomy model) and about what will be checked at final selection. A host post of 2026-09-21 (quoted in thread 749145) says OAI "does not violate the … rules if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge". Open: NIH blocks NDA access from China, Russia and Venezuela, whose residents are eligible.

**P1 rule changed.** Third-party evidence puts the expected effect of a ~0.93 leg on these blends at ≤ 0:
- TranBaDat: "every one [of our legs] so far cost ~0.001 on OAI blends";
- goodpjw2008: their 0.929 reader at full weight cost 0.002 on the 0.956 base and tied at 1/10.

P1 keeps the 0.2 share but becomes the main pick only if it beats P0 by **≥ 0.002** (the §6 noise band); otherwise P0.

**Runtime guard.** These pipelines take hours on the hidden test (TranBaDat: R7 ≈ 13 s/study on 2×T4; a 4-leg notebook ≈ 7.5 of 9 h). Our leg (≈ 40–60 min) is added only after reading the base's own time guards. It runs under its own time guard and must never cause the base to drop a leg. If the measured margin is under 1 h, P1 is not submitted.

### 10b. Option A checked against the notebook code (2026-10-19 02:40 IST, read-only Kaggle API)

- **goodpjw2008 v27 (0.958)** loads:
  - R6: bundle `fe055406fad6a12f` from rostigor's versioned copy, with every file hash-checked against manifest `6508ce69…`;
  - nartaa 384 + 224 and goodpjw2008's reader at 1/10.
  - The R7 dataset is attached but no code cell loads it.
  - The orthofoundation dataset supplies code and `anatomy.pt` (SKM-TEA). Its OFL-L weights load only for an `ofl_l` backbone, and R6 is a ConvNeXt-small.
- **R6's fastMRI status cannot be verified.**
  - Its manifest records an abbreviated training config without `fm_aux`.
  - But the training run is named `cnx_r6fm`, and it shares its trainer (md5 `8622009e…`) with R7. R7's full config sets `fm_aux` (a fastMRI 384-px cache, weight 0.3).
  - heliosli's R6 notebook now returns 404.
- **S6 shows no fastMRI marker.**
  - Run `cnx_s6`, built 2026-10-09, before R6 and R7.
  - ImageNet-22k ConvNeXt-small trained on 4,344 competition studies, gold excluded; knee-centred views from `anatomy.pt` (SKM-TEA).
  - heliosli's own `rsna-knee-blend-gold-dataset-restored` is nartaa 384 + S6 at 70/30 with an 8.3 h deadline, the recipe goodpjw2008 measured at 0.954 five times.
- **Owner decision needed:**
  - **A-strict:** nartaa + S6, 0.954, no fastMRI marker.
  - **A-R6:** goodpjw2008 v27, 0.958, R6 possibly fastMRI-trained.

### 10c. New evidence and revised Monday plan (2026-10-19 04:20 IST; read-only checks, no result of ours)

[V] = read by me on the primary page or file; [S] = a participant's statement; "hypothesis" = not measured.

- **fastMRI access [V, fastmri.med.nyu.edu]:** "The application must include the investigator's institutional affiliation and the proposed uses of the data." The host's test excludes datasets that "limit access to users with institutional affiliations". So R7 (fastMRI auxiliary training) very likely falls outside the rules, and so does R6 if it was trained the same way.
- **R6 vs R7 weights [V, fold 0 of each, loaded with the restricted loader, not run]:**
  - Layouts are identical: 376 tensors, 50,065,188 parameters.
  - Neither file holds a fastMRI head, although R7 is documented as trained with one. The head is dropped at export, so the weights cannot show whether R6 used fastMRI.
  - S6 differs from both only by one scalar (`syn_beta`).
- **The trainers settle it as far as the files can [V, the embedded trainer sources whose SHA-256 match each bundle's manifest]:**
  - R6's trainer (`compact_train.py` 04df7e2e…, 230 KB) is version 9.8. Its header says it was written on 10-11 01:00 to add `--fm-aux DIR = the fastMRI+ lesion-box auxiliary task on the shared encoder`, at weight 0.3. The aux head "is stripped from deploy_state_dict (fold models keep v9.7's key set)", which is why the weights show nothing.
  - R6 was built 14 hours later (2026-10-11 15:48) with this trainer, from a run named `cnx_r6fm`. Without the flag, version 9.8 is "v9.7 bit for bit", so the version and the name have no other purpose. **R6 was very likely trained with fastMRI+ (inferred, not stated by its author).**
  - S6's trainer (`compact_train.py` 14767c7a…, 80 KB, an earlier version) contains no fastMRI option at all (zero `fm_aux` references) and no external-data hook beyond the SKM-TEA anchors (`anatomy.pt`) for its knee-centred views. **S6 cannot have used fastMRI.**
- **Questions still open [V, 04:05 IST]:**
  - A participant asked heliosli in thread 743416 (about 18 h earlier) whether R6 used fastMRI: no reply yet.
  - The host has not answered the fastMRI question in thread 750259.
- **A-strict confirmed [V]:** `heliosli/rsna-knee-blend-gold-dataset-restored` shows public score 0.954 (version 1). Its inputs are S6 (`heliosli/rsna-knee-compact-v1`), nartaa's SWA weights and the orthofoundation dataset (`anatomy.pt`); no R6 and no R7. It sets `NB_T0`.
- **No fastMRI-free notebook above 0.954 [V, metadata of the top 40 by score]:**
  - Every notebook at 0.955 or above attaches R6 or R7.
  - kaiwalyaatulraut (×2) and kiyoshiohno at 0.958 are v27-type forks; prvsiyan's 0.957 uses R6; lavinwins's different stack scores 0.943.
  - Despite its name, `heliosli/rsna-knee-compact-r6cnxfsss` holds the S6 bundle (manifest byte-identical to S6's). heliosli's most-voted "old dataset" notebook is therefore the S6 recipe too; no fastMRI-free R6 exists.
- **goodpjw2008's measured ladder [S, their notebook "Does S6 help next to R6?"]:**

  | Recipe | Public LB |
  |---|---|
  | nartaa CoAtNet alone | 0.949 |
  | their reader alone (3 folds) | 0.929 |
  | nartaa + S6 (+ reader), versions 1–3 | 0.954 |
  | nartaa + R7 + 224 checkpoint at 35 % | 0.956 |
  | nartaa + R6 flip TTA, 50/50 | 0.957 |
  | + heliosli's per-finding weights + 224 mirror + reader at 1/10 (v27) | 0.958 |
  | v27 + S6 as a second compact model at 0.3 | 0.957 |
  | v27 with per-finding weights tuned on gold-58 | 0.957 / 0.955 |

- **Runtime [S, v27's author]:**
  - 6.5–8 h with the stack.
  - Run-to-run variation is ±0.001 (0.944/0.944/0.943).
  - Their reader is skipped after 8.2 h, and the 224 stage when under 70 min remain.
- **P1 is dropped, for two reasons:**
  1. With the base at 6.5–8 h, our cell (it needs 90 min free) would skip on most runs. The LB cannot tell a skip from "no effect".
  2. goodpjw2008 blends their 0.929 reader at 0.3–3.5 % per finding. At 45 % it lost to the stack alone. Our models score 0.928 at a 20 % share: expected effect ≤ 0, and unmeasurable at ±0.001.

  Our models stay in the report and in fallback C. `notebooks/part_c_ours_blend_cell.py` is kept but not used.
- **A-strict+ (hypothesis, not measured):** v27 with its R6 stage replaced by heliosli's S6 stage. Both write `/kaggle/working/ours_sub.csv`.
  - Kept: nartaa 384 + 224 (35 % in the Raptor slot) and the reader at 1/10.
  - S6 at heliosli's own share (Raptor 0.7 / S6 0.3, the 0.954 recipe).
  - Prior [S, varshithakolupuri]: the 224 checkpoint at 35 % of the Raptor slot was measured on this S6 recipe itself. The displayed score stayed 0.954 (Kaggle and the API show three decimals) while their rank moved from 437 to 320: a gain below 0.001. On the R7 recipe it moved rank 233 → 156 at a displayed 0.956. Expected A-strict+: 0.954–0.955.
  - Prior [S, goodpjw2008, versions 1–3 of the notebook above; added 2026-10-19 07:15 IST]: nartaa + S6 with their reader scored 0.954, the same as heliosli's nartaa + S6 without it. Neither added part has shown at three decimals on this base, so the most likely display for A-strict+ is 0.954.
  - Fit in 9 h [inferred from S; 07:15 IST]: v27 with S6 added as a second compact model (their versions 4–5) completed and scored 0.957. A-strict+ is that notebook without the R6 stage, so it should finish sooner. That notebook's current version is pinned to heliosli's image (`2757e0c7…`), the one A-strict+ pins, and contains the same 224 and reader cells. Whether those cells ran or fell back there, and which image versions 4–5 used, is not shown. Inside the notebook, the S6 deadline (8.3 h) counts from the notebook start (`T0 = NB_T0`), the 224 stage keeps 40 min in reserve, and Part B skips after 8.2 h.
  - **Rule:** A-strict+ becomes the main pick if it completes and its displayed public score is ≥ A-strict's. The plain A-strict fork is then the second pick; if A-strict+ scores lower, the order is reversed. A sub-0.001 difference cannot be resolved at three decimals, so the choice rests on the prior (both added parts measured ≥ 0 on other bases), not on LB probing. Both are fixed public recipes; nothing is tuned on gold-58 or the LB.
- **Recommendation to the owner:**
  - Both final picks: the two A-strict variants, ordered by the rule above. They match the owner's option-A policy (OAI + SKM-TEA, no fastMRI).
  - A-R6 (0.958) is not recommended: R6 was very likely trained with fastMRI+, and fastMRI's application requires an institutional affiliation. Reconsider only if heliosli states that R6 used no fastMRI data.
  - Not our own 0.928 ensemble: only the better pick counts, and ours cannot beat either variant.

### 10d. Result (2026-10-19 17:34 IST; measured on the public LB)

| Variant | Submission | Public LB |
|---|---|---|
| A-strict | 57384567 | **0.954** |
| A-strict+ | 57384570 | **0.954** |

- Both completed on the hidden test within the 9 h limit.
- **Rule applied:** 0.954 ≥ 0.954, so A-strict+ is the main pick and A-strict the second.
- **Prediction check:** the registered range was 0.954–0.955, most likely 0.954. The result is at the lower end. A-strict+'s extra parts (224 checkpoint, reader) are worth less than 0.001 public AUC on this base. That is consistent with the varshithakolupuri and goodpjw2008 priors and is not a measured gain.
- **Leaderboard (public, 6,211 teams):**
  - the owner's team ranks 1,979 (was 3,761 at 0.928);
  - #10 scores 0.966 and gold (#22) 0.964;
  - silver (#310) and bronze (#621) both lie inside the 1,231-team block at 0.958.
- No further submission is planned under this plan. Another push or submit needs the owner's new approval.

## 11. Sprint (2026-10-20): final-strict, registered 00:28 IST (push at 00:27, before any result)

Inputs: `reports/gap_discovery_audit.md`.

**Owner decisions (2026-10-20, between 00:18 and 00:27 IST):**
- The AWS G/VT quota request (4 vCPU) was declined, so there is no cloud GPU. No training track.
- Both final picks stay fastMRI-free. The R6/R7 "max" pick was offered and declined.
- final-strict: push and one submission approved.

**final-strict** (`notebooks/candidate_A/final_strict/`, built by `src/emergency/build_final_strict.py`):
- A-strict's stages 1–2 unchanged.
- New stage 2b: nartaa's all-data S6 ConvNeXt (s6fall, 0.950 solo [S]).
  - It runs through nartaa's script with the four-model notebook's v2 patch.
  - The patched script is byte-identical to that notebook's copy (sha256 2e8c4dca…).
- **Blend, fixed now:** per-finding percentile ranks, Raptor 0.5 / S6 0.25 / s6fall 0.25. If s6fall is skipped, fails, or misses more than 3 % of studies, the blend is A-strict's 0.7 / 0.3.
- Local blend tests pass (`src/emergency/test_final_strict_blend.py`, 4 cases). Without s6fall the output is identical to the unmodified A-strict cell.
- **Expected:** 0.955–0.956 (hypothesis). Basis:
  - s6fall is a different label set on the S6 architecture;
  - adding s6fall to the R7 stack moved 0.956 → 0.957 [S].

**Final-pick rule (fixed now):**

| final-strict displays | Final picks |
|---|---|
| ≥ 0.955 | final-strict + A-strict+ |
| 0.954 | final-strict + A-strict+ (equal display; final-strict has the extra leg and passed its commit checks) |
| < 0.954, or the hidden run fails | A-strict+ + A-strict, unchanged |

- No weight is tuned on the LB result.
- A second final-strict variant needs the owner's OK and a new registration.

### 11a. Result (2026-10-20, seen 05:37 IST; measured on the public LB)

- final-strict (submission 57406523) completed the hidden-test run and displays **0.956**, inside the expected 0.955–0.956.
- **Rule applied (row "≥ 0.955"): final picks = final-strict + A-strict+.** A-strict stays the fallback.
- The +0.002 over A-strict / A-strict+ (both 0.954) is one public-split comparison at three decimals. The public gap is roughly 0.001–0.003. The private-split effect is unknown.
- Public LB at 05:44 IST (file stamped 00:14:45 UTC, 6,239 teams):
  - the owner's team ranks 1,983 (0.956 block: ranks 1,684–1,984);
  - #10 0.966; gold (#22) 0.964;
  - silver (#311) and bronze (#623) both inside the 0.958 block (ranks 192–1,502).

## 12. Gold-58 study of the public legs (registered 2026-10-20 00:46 IST, before any prediction is computed)

Owner's brief: Tasks 1–4 (final-strict v2). Every rule below is fixed now. No rule changes after any number is seen.

**Sets.**
- **gold_58:** image-read labels (`data/labels_v3/gold_labels_original.csv`). Positives per finding range from 9 (MCL) to 35 (Effusion).
  - Out-of-sample for S6 (mean of 4 folds; gold never trained, manifest `gold_in_training: 0`), s6fall (gold excluded, per its own `gold58.json`), and our E9/E1p.
  - Raptor 384/224: the training corpus excludes gold (4,349 studies, per its docstring), but its recipe was selected on gold, so its gold AUC is optimistic.
  - Reader: to be checked in its code before use.
- **oof_300:** 300 non-gold studies, 75 per S6 hash fold (`sha1(uid)[:8] % 4`), seed 20261020.
  - Out-of-fold **only** for S6 (its held-out fold model) and our OOF teachers.
  - In-sample for Raptor and s6fall (both all-data). So oof_300 is used only for S6-only questions, against labels_v3 confident cells (≤ 0.2 or ≥ 0.8). Never for blend weights.

**Metric.**
- Macro AUC over the 12 findings.
- Paired bootstrap over studies: 10,000 resamples, seed 0. Report Δ and the 95 % CI.
- **Baseline:** final-strict v1 = rank blend 0.5 Raptor384 + 0.25 S6 + 0.25 s6fall.

**Upgrades.** Each is tested once. Adopt only if, on gold_58, Δ ≥ +0.0015, the CI lower bound ≥ 0, **and** Δ > 0 on at least 7 of 12 findings (added to limit chance adoptions across these 6 tests).

| # | Upgrade | Extra condition |
|---|---|---|
| A | 5 legs: + Raptor224 at 0.35 of the Raptor slot, + reader at A-strict+'s per-target weights | Runtime fits (from commit logs) |
| B | Mirror TTA on S6 and s6fall with R6's own mirror code (medial↔lateral swap; MCL from the plain pass) | S6-only Δ on oof_300 ≥ 0. Prior: S6's trainer has no mirror augmentation, so expected to fail |
| C | Probit-rank, then standardised-logit, instead of uniform rank | — |
| D1 | Compartment schedule, fixed as briefed: ACL/MCL/MedMen Raptor 0.55 / S6 0.225 / s6fall 0.225; PF OA, Med OA, Lat OA, Baker's, Synovitis, LatMen 0.40 / 0.30 / 0.30; the rest unchanged | — |
| D2 | Cross-finding log-odds terms (Effusion/OA/ACL → Synovitis; ACL/Fracture → Contusion), L2-penalised, fitted leave-one-study-out on gold_58 | Rule applied to the LOSO predictions |
| E (Task 3) | Residual adapter on the S6/s6fall class tokens, 5-fold CV on gold_58 + 300 anchors | **The brief's rule:** out-of-fold gold macro Δ ≥ +0.004 with lower bound > 0, **and** rank correlation ≥ 0.97 on the anchors |

**Raptor memorisation check (descriptive only):** Raptor's gold AUC and calibration next to S6's and s6fall's.

**Statistical power.** One macro-AUC difference on 58 studies has a bootstrap SD of roughly 0.005–0.01 [E]. Only large effects can pass. A null result is the expected outcome and is not evidence of no effect.

### 12a. Notes added 2026-10-20 05:51 IST, before any gold AUC was computed (no rule changed)

- **Local faults found and fixed before any gold number:**
  - Raptor: nartaa's reader calls `os.pread`, which Windows lacks. The error was swallowed, so the first local Raptor run scored empty volumes. Those outputs were discarded.
    - The patched reader (`patch_fastread_for_windows`) is byte-identical to the module's reference reader on 3 gold studies.
    - It now refuses empty volumes.
  - S6: on Windows, an empty `CUDA_VISIBLE_DEVICES` does not hide the GPU, so 3 loader workers held CUDA contexts and the card spilled into system RAM (one study took 7,719 s).
    - Now: 1 GPU loader worker (the configuration that matched Kaggle within 0.0003), a capped allocator, and the cache freed after every study.
    - Outputs are unchanged by this; only speed was affected.
- **B:** MCL keeps the plain pass (R6's rule), so MCL's Δ is 0 by construction. 11 findings can move.
- **D2:** it alters only Synovitis and Contusion, so it cannot meet "Δ > 0 on ≥ 7 of 12 findings". The registered rule stays as written; D2's numbers are descriptive.
  - Its leave-one-study-out refit also moves the intercept with each held-out label. That biases the pooled LOSO AUC downward. Noted, not corrected.
- **A2 (reader):** not run. The brief made it optional, and its per-finding weights (0.003–0.035) bound its rank effect. A-strict+, which has both of A's parts, scored the same 0.954 as A-strict on the public LB. A is tested as A1 (+ Raptor224) only.

### 12b. Task 2 result (2026-10-20 06:08 IST; gold_58, all 58 studies with every leg; `output/emergency_research/task1/task2_gold_eval.json`)

**Single legs (macro AUC on gold):**
- Raptor384 (pair) 0.9229, Raptor224 0.9160, Raptor384 plain 0.9219.
- S6 0.9267 (mirror TTA 0.9291).
- s6fall 0.9314 (mirror TTA 0.9203).
- **Baseline: final-strict v1 0.9361.** The blend beats every leg.
- **Raptor memorisation check (descriptive):** Raptor is the *weakest* leg on gold, so there is no sign that its gold-selected recipe inflates it.
  - It over-predicts heavily: mean prediction 0.23 above prevalence, against about 0.07 for S6 and s6fall.
  - That doesn't matter for ranks.
  - Its gold strengths are complementary: Fracture 0.958 vs S6 0.882 and s6fall 0.901; Effusion; Medial Meniscus. S6 and s6fall lead on Lateral OA, PF OA, Lateral Meniscus and Contusion.

| Upgrade | Gold macro | Δ vs v1 | 95 % CI | Findings up | Adopt? |
|---|---|---|---|---|---|
| A1 (+ Raptor224 in the Raptor slot) | 0.9360 | −0.0001 | [−0.0035, +0.0033] | 6/12 | no |
| B (mirror TTA on S6 and s6fall) | 0.9362 | +0.0002 | [−0.0050, +0.0057] | 7/12 | no |
| C1 (probit ranks) | 0.9362 | +0.0002 | [−0.0014, +0.0018] | 4/12 | no |
| C2 (standardised logits) | 0.9389 | +0.0028 | [−0.0011, +0.0071] | 8/12 | no: CI lower bound < 0 |
| D1 (compartment weights) | 0.9382 | +0.0021 | [+0.0000, +0.0047] | 5/12 | no: 5 < 7 findings up |
| D2 (cross-finding LOSO) | 0.9284 | −0.0077 | [−0.0148, −0.0016] | 0/12 | no (worse) |

- **No Task 2 upgrade is adopted.** Per §12, this null is the expected outcome at n = 58 and is not evidence of no effect.
- C2 and D1 have the largest point gains (+0.0028 and +0.0021). They stay untested beyond this one look, and no rule is relaxed after seeing them.
- B's mirror pass helps S6 on gold but hurts s6fall, and the blend is unchanged. B's oof_300 condition (S6 only) is still computed as registered, though B has already failed on gold.
- Task 3 (E, the adapter) runs after the oof_300 legs finish.

### 12c. B's oof_300 condition and Task 3 (2026-10-20, 08:20 and 08:21 IST; `task2b_oof_mirror_check.json`, `task3_adapter.json`)

- **B, oof_300 condition (S6 only): not met.**
  - Setup: each of the 300 studies is scored by the one S6 fold model that never trained on it, against labels_v3 confident cells.
  - Plain 0.9180, mirror 0.9113: Δ **−0.0066**, 95 % CI [−0.0107, −0.0026].
  - Only Effusion goes up, 1 of the 11 findings that can move.
  - B stays rejected; it had already failed on gold (§12b). This matches the registered prior (S6's trainer has no mirror augmentation).
- **E, the residual adapter: not adopted.**
  - Out-of-fold gold macro 0.9364 against 0.9364 frozen: Δ **+0.0000**, CI [−0.0003, +0.0004]. Anchor Spearman is 1.0000 for both legs.
  - The training settings fixed in `task3_gold_adapter.py` (L2 weight 10, Adam lr 1e-3, 400 steps) let the penalty dominate. The largest adapter weight is 0.0005, so the adapter stayed at the identity.
  - This test therefore says nothing about a less-regularised adapter.
  - It is not re-run with other settings. That would be a second look at the same 58 gold studies after seeing this result (§12; the brief's "do not repeatedly tune against the gold set").
- **Raptor memorisation on oof_300 (descriptive; `task1b_oof_memorisation.json`).** Added at about 08:30 IST, after the registered analyses. It decides nothing.
  - Against labels_v3 confident cells, the S6 fold model that never saw a study scores 0.9180.
  - Legs that trained on these studies score far higher. Lead over that reference, with 95 % CI:
    - S6 4-fold mean (3 of 4 folds trained on each study): 0.9638, +0.046 [+0.035, +0.057];
    - s6fall: 0.9734, +0.055 [+0.044, +0.068];
    - Raptor384: 0.9567, +0.039 [+0.029, +0.050];
    - Raptor224: 0.9602, +0.042 [+0.031, +0.054].
  - Every all-data leg fits its training studies. Raptor's in-sample lead is the smallest, and on gold it is the weakest leg (§12b). So there is no sign that it memorises more than the S6 family.
  - Caveats: the label sources differ between the two sets, so these numbers only rank legs within a set. A lower in-sample score can also mean a leg's own training labels agree less with labels_v3.
  - This is also why oof_300 is never used for blend weights.
- **Task 4: every upgrade failed its rule (A1, B, C1, C2, D1, D2, E), so there is no final-strict-plus to build.**
  - final-strict v1 stays as submitted, and the §11 picks stand: final-strict + A-strict+.
  - No push, no submission, no Kaggle GPU time used.

## 13. Owner's brief 2026-10-20 (received ~18:05 IST): final-max with R6, and a fine-tuned model. Registered 18:21 IST, before any R6 gold number or training result

**Owner decisions (from the brief):**
- **Pick 1 = `rsna-knee-final-max`**: v27 (R6) + nartaa's s6fall + (later) our fine-tuned model. This reverses, for pick 1 only, the 10-19 / 10-20 decision to keep both picks fastMRI-free.
- **Pick 2 = `rsna-knee-final-strict-v2`**, fastMRI-free.
- Authorised:
  - kg.py `datasets create` / `datasets version` (private) for our cache and weights;
  - `kernels push` and `competitions submit` for final-max, final-strict-v2 and the trainer notebook.
  - Titles, comments and descriptions in plain lowercase, with no tool names.
- The owner states that hidden-test scoring costs no GPU quota and that about 28 h of T4 x2 quota remain.
- **Risk, stated once:** R6 is fastMRI-assisted (§10c). If the host rules fastMRI-trained weights out, the violation is the team's, not only pick 1's.

**Track A: final-max v1.**
- Stages:
  - v27's A1 (Raptor 384 + mirror), A1b (Raptor 224, 35 % of the Raptor slot) and A2 (R6 + flip TTA), unchanged;
  - new A2b = final-strict's s6fall stage. It starts only if at least 45 min remain before NB_T0 + 8.35 h.
  - v27's A3 blend (edited as below) and Part B reader (1/10), otherwise unchanged.
- **Blend weights (the brief's, fixed now):**
  - Raptor-slot weight per finding: v27's `W_RAPTOR_C` (Medial OA 0.5, Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35), plus Fracture 0.65; else 0.5.
  - Compact slot: 0.60 R6 + 0.40 s6fall, or R6 alone if s6fall is missing.
- **Transform, chosen on gold_58 (owner's rule: whichever scores higher):**
  - T1 = v27's per-finding percentile ranks.
  - T2 = 0.5 x T1 + 0.5 x pct(the same weights applied to per-leg standardised logits, i.e. Task 2's C2 construction).
  - Correction to the brief: pct(zstd(logit(p))) equals pct(p), so T2 is the meaningful version of its 50/50 mix.
  - Because this is a two-way choice made on gold, the chosen variant's gold score is slightly optimistic.
- Gold legs come from local runs with each leg's own code: R384/R224 (`task1/raptor`), s6fall (`task1/s6fall`), R6 (`r6_gold_preds.py`, R6's v4c worker with v27's arguments).
- Test: `test_final_max_blend.py`. Then push, check the 3-study commit run, and submit.

**Tracks B–C: the fine-tuned model ("s6ft").**
- **Warm start: s6fall.**
  - Its licence (CC BY-NC 4.0) permits fine-tuning. S6's dataset licence is "other", so S6 is not fine-tuned.
  - Recipe and architecture: nartaa's `train_s6f0_all.py` (S6 Net, convnext_small, spatial attention, 320 px rows, 288 crop, 64 windows per study).
- **Training set:** the non-gold training studies with DICOMs. gold_58 is never read by the trainer (no images, no labels). No test data.
- **Targets (the brief's rules; nothing fitted on gold):**
  - P_report = 0.5 labels_v3 + 0.5 nartaa's JEV blend (both report-derived).
  - P_vis = our 5-fold out-of-fold image models (`pl_g_r1_oof_only.csv`), Platt-calibrated to P_report on the pool.
  - Deviation from the brief: no gold prevalence is used, so gold stays sealed.
  - Over-called findings (Medial OA, Lateral OA, PF OA, Baker's): where P_report > 0.6, target = 0.45 P_report + 0.55 P_vis.
  - Under-called findings (Synovitis, Fracture, Lateral Meniscus, ACL, Contusion): where 0.08 <= P_report <= 0.45, target = 0.70 P_vis + 0.30 P_report.
  - Otherwise target = P_report.
- **Recipe (fixed before launch; no checkpoint chosen on gold):**
  - s6fall weights; stem and stages 0–1 frozen.
  - AdamW: encoder lr 3e-5, head 1e-4, wd 0.05; 5 % warm-up then cosine; clip 5; EMA 0.995. The EMA weights after the last epoch are the model.
  - Loss = mean over findings of alpha_c (0.5 soft BCE + 0.5 soft pairwise AUC), with alpha 1.6 on Lateral OA, PF OA, Synovitis, Lateral Meniscus and Fracture, else 1.0. The pairwise term compares each study with a queue of recent detached logits.
  - fp16 + GradScaler on the T4 (no bf16 on Turing).
  - The epoch count is fixed from measured throughput before launch.
- **Data:** local precompute of the SKM-TEA anchors and 320-px rows (s6fall's own `precompute_anchors.py` / `px_cache` path). Per study, a fixed random subset of training windows is stored compressed, uploaded as a private Kaggle dataset, and trained on T4 x2.

**Track D: adoption (owner's rule).**
- s6ft is scored locally on gold_58 with s6fall's inference path (all windows).
- Two variants, fixed now:
  - V1: s6ft replaces s6fall in its slot (no runtime change);
  - V2: the s6fall slot is split equally between s6fall and s6ft.
- A variant goes into final-max v2 / final-strict-v2 only if its gold_58 blend beats the current blend with paired-bootstrap 95 % CI lower bound >= 0 (10,000 resamples, seed 0). If both pass, the one with the larger Δ is used.

### 13a. Track A result (2026-10-20 18:55 IST; gold_58, 58/58 studies with every leg; `task1/final_max_gold_eval.json`)
- **R6 on gold:** run locally with R6's own v4c worker and v27's arguments; 58/58 scored. It scores **0.9367**, the best single leg (s6fall 0.9314, S6 0.9267, Raptor384 0.9229, Raptor224 0.9160).
- **final-max blends:**
  - T1 0.9386, T2 **0.9391**;
  - v27's own blend 0.9384; final-strict v1 0.9361.
- **Owner's rule applied: T2.** Its edge over T1 is +0.0005 [−0.0021, +0.0033], which is noise; the rule picks it anyway.
- Descriptive comparisons:
  - final-max T2 vs final-strict v1: +0.0030 [−0.0023, +0.0090];
  - s6fall added to v27 (T1 vs v27's blend): +0.0002, so s6fall adds little next to R6 on gold.
- **Built:** `notebooks/candidate_A/final_max` (notebook sha256 a060ce02…), byte-identical to the T2 build that passed `test_final_max_blend.py` (12/12 cases).
- The push waits for Kaggle write permission (kg.py edit blocked by the auto-mode classifier; the owner was asked at 18:45 IST).

### 13b. Notes before any s6ft result (2026-10-20 20:22 IST; no rule changed)
- **Cache.**
  - The first full build (7 GPU-anchor workers) stalled at 1,315 / 4,349 studies at 19:44 IST; the loader timed out at 20:14.
  - The study it stalled on rendered normally on the restart, so the stall was transient (likely VRAM pressure from 7 workers).
  - Restarted 20:15 with 5 workers. A study the loader waits on for more than 600 s is now skipped and logged; a rerun retries it.
- **Deviation from nartaa's sampling.**
  - The cache keeps a fixed random 64 of each study's windows (JPEG q90, about one grey level of error), so every epoch sees the same 64 windows per study, with a fresh 288 crop.
  - nartaa's trainer draws 64 from all windows each epoch.
  - Reason: upload size (64 at q90 ≈ 16 GB; 96 at q95 ≈ 39 GB).
- **Where it trains.**
  - Kaggle T4 x2 needs the Kaggle write permission.
  - Until then the fallback is the local RTX 4050: bf16, batch 1 with accumulation 2 (an effective batch of 2), about 2.5 s per step, measured.
  - The same recipe either way. **3 epochs, fixed now**; the EMA after the last step is the model.
- **Track D script:** `s6ft_gold_eval.py` implements the rule registered in 13 (CI lower bound >= 0; if both pass, the larger Δ).
  - Dry run with s6fall standing in for s6ft: both bases reproduce 13a (final-max T2 0.9391, final-strict v1 0.9361).
  - Identical predictions give a CI of [0, 0], which "passes". This cannot happen with distinct weights.
- **21:00 IST, the owner's decisions (before any s6ft result):**
  - **Full-window cache.** Training is local, so the upload-size reason for 64 windows is gone, and the owner freed disk (361 GB).
    - The 64-window build was stopped at about 2,570 studies; it is kept as a fallback in `D:/rsna_work/s6ft_cache`.
    - The new build keeps every window (`--k 0`, q90, about 149 GB) in `D:/rsna_work/s6ft_cache_all`. The trainer then draws a fresh 64 per study each epoch, exactly nartaa's sampling.
    - Everything else is unchanged (3 epochs, the same recipe and rule).
  - **final-max v1 push approved;** pushed 21:02 IST (`nishkarshsinghal/rsna-knee-final-max` version 1, notebook sha a060ce02…). `kg.py` now allows this one folder, guarded by a private and no-tool-name scan, with at most 2 submissions (v1, plus v2 only if s6ft passes).

### 13c. final-max v1 submitted (2026-10-20 21:18 IST, submission 57444580)
- The commit run (3 public test studies, 21:02–21:17 IST) passed every gate:
  - `S6F_STAGE_DONE True` (112.6 s), `S6F_IN_BLEND used transform t2`, `NOTEBOOK_COMPLETE`;
  - R6 3/3 scored with no fallbacks; Raptor 3/3; s6fall failed on 0.
- Output kept in `D:/rsna_work/tmp/final_max_v1_out`.
- **Scored 0.958 on the public LB** (seen 2026-10-21 04:21 IST): +0.002 over final-strict's 0.956, equal to v27's displayed 0.958.
  - Snapshot at 22:51 UTC: 6,294 teams. Hent AI is listed at rank 422; the 0.958 block spans ranks 219–1,646.
  - Medal lines: silver cut (rank 314) and bronze cut (rank 629) are both 0.958; gold (#22) is 0.965.
  - The private LB decides.

### 13d. Track D result: s6ft NOT adopted (2026-10-21 04:13 IST; `task1/s6ft_gold_eval.json`)
- **Training:** local RTX 4050 on the all-window cache (4,349 non-gold studies), 3 epochs = 6,522 steps, bf16, 232 min; not stopped early.
  - The loss fell from 0.33 to 0.279. EMA weights are in `D:/rsna_work/s6ft_local/s6ft_ema.pt`, not uploaded.
- **gold_58, single legs:** s6ft 0.9279 vs s6fall 0.9314 (Δ −0.0035).
  - Better on MCL +0.020, Contusion +0.015, Lateral Meniscus +0.007, Medial OA +0.005, Effusion +0.003.
  - Worse on PF OA −0.020, Fracture −0.020, Lateral OA −0.019, Synovitis −0.013, ACL −0.010, Medial Meniscus −0.006.
  - Four of the five up-weighted findings got worse (all but Lateral Meniscus). These are the findings whose targets the vision-model rules changed.
- **Registered rule (13), applied:**
  - final-max: V1 +0.0001 [−0.0021, +0.0025]; V2 −0.0001 [−0.0011, +0.0008]. Neither passes.
  - final-strict-v2: V1 −0.0008 [−0.0033, +0.0017]; V2 +0.0002 [−0.0015, +0.0020]. Neither passes.
- **Decision:** no final-max v2 and no final-strict-v2. The picks stay final-max v1 (57444580, pending) and final-strict (57406523, 0.956).
- No further s6ft variant is tried on gold_58, under the standing rule against repeated tuning on the same 58 studies.

## 14. Mission 2026-10-21 (received ~05:50 IST): above the 0.958 block. Registered 06:10 IST, before any R7 gold number, any in-pass s6fall number or any new training result

**Claims checked against the code and logs (05:50-06:10 IST):**
- s6ft's vision teacher was `pl_g_r1_oof_only.csv` (our old E1p + O1–O4 out-of-fold models): **confirmed** (`s6ft_targets.py`).
- S6 (`rsna-knee-compact-v1`) is **384 px rows / 352 crop**, like R6 and R7, not 288. The all-window cache holds s6fall's **320 px** rows. S6 or R6 can only score it after resampling 320 → 384.
- R6 and R7 were trained with fastMRI (R7's manifest: `fm_aux`). A student distilled from R6 or R7 is no longer fastMRI-free, so a teacher for a pick-2 model may only use S6.
- `--models2` exists and shares decode, render and anchors. But on Kaggle, R6's stage is decode- and GPU-bound together: about 48 s CPU decode + 20 s GPU (4 folds x flip) per study (commit-run log). It is about 13 s per study on 2 x T4 (TranBaDat), so about 4.5 h for about 1,300 studies.
  - R7 as a second group doubles the GPU work (about +3.6 h). That cannot fit in 9 h next to v27's other stages (6.5–8 h in total).
  - R7 can only **replace** R6 at equal cost. Adding S6 the same way costs about +1.8 h. goodpjw2008's own v27 + S6 (0.3) scored 0.957 < 0.958.
- **final-max v1's s6fall very likely never ran on the hidden test.** Its stage decodes every DICOM again (Kaggle commit run: 76 s for 2 studies in one process). That is about 4 h for 1,300 studies.
  - It may only start with 45 min left before 8.35 h, and then only gets that time. A deadline is a hard error in `s6f_solo_v2.py`, so the leg is dropped and R6 is used alone.
  - That would make v1 = v27 on the hidden test (both 0.958), possibly also without the reader (skipped after 8.2 h).
  - final-strict's s6fall did run: A-strict's stages are about 3 h shorter. Its score moved 0.954 → 0.956.
- Per-finding selection on gold_58: goodpjw2008 tried it; v27 with gold-tuned per-finding weights scored 0.957 / 0.955 on the LB (vs 0.958). So **no per-finding selective blend** is registered.

**Registered actions and rules:**
- **A. final-max v2 = v1's blend unchanged** (compact 0.60 R6 + 0.40 s6fall, T2, Fracture 0.65), with s6fall scored **inside R6's worker** on a second render of the same decoded study.
  - Code: `build_compact_infer_x.py` patches the v4c worker → `compact_infer_x.py`. It uses s6fall's own trainer (14767c7a) for windows and Net, 320/288 px, every window, the plain fp16 pass. The separate a2b stage is removed.
  - Gates (fidelity, not selection):
    - (i) R6 through the patched worker matches the plain worker on gold_58: max |dp| ≤ 2e-3.
    - (ii) In-pass s6fall vs s6fall's own pipeline on gold_58: every per-finding Spearman ≥ 0.99, and macro AUC within ±0.002.
    - (iii) The Kaggle commit run scores 3/3 with R6 and s6fall, shows `S6F_IN_BLEND used`, and completes.
  - Expected runtime: v27 + about 5–8 % of R6's stage (one 320-px render, about 2.6 s CPU, plus one 288-px forward per window).
- **B. R7.** Scored on gold_58 with R6's worker and v27's arguments, including flip TTA and R7's own stackers (`r7_gold_preds.py`).
  - Only C1 (R7 replaces R6 in final-max's compact slot, same cost) can be adopted: paired bootstrap Δ(C1 − base) on gold_58, 10,000 resamples, seed 0, 95 % lower bound ≥ 0. Base = the final-max design with native legs (13a: 0.9391).
  - M1 (0.35 R6 + 0.35 R7 + 0.15 S6 + 0.15 s6fall) and an R6 + R7 compact slot are reported for information only; they do not fit 9 h.
- **C. s6ft_v2** (the brief's Track 2):
  - Teacher: S6's held-out fold (sha1 fold) on the cache. Windows are resampled 320 → 384 (bilinear), centre-cropped to 352, plain pass, up to 160 evenly spaced windows. It is first checked on oof_300 against S6's native held-out predictions. The full run goes ahead only if every per-finding Spearman there is ≥ 0.85 (added 06:20 IST, before the check ran).
  - Targets: 0.70 P_report + 0.30 S6-OOF quantile-mapped per finding to P_report on the pool. Gold stays sealed.
  - Training: 2 epochs from s6fall, alpha 1.0 for all findings, encoder lr 2e-5, head lr 5e-5.
  - Scored once on gold_58. Adoption is the rule of section 13 (V1 replace / V2 split; lower bound ≥ 0; the larger Δ if both pass).
- **D. final-strict-v2** (pick 2, fastMRI-free) is built only if C passes for the strict base. Otherwise it would repeat final-strict's predictions, and section 12 already found nothing to add.

### 14a. B result: R7 NOT adopted (2026-10-21 06:42 IST; `task1/r7_gold_eval.json`)
- R7 on gold_58 (58/58, R6's worker, v27's arguments, flip TTA, R7's own stackers): **0.9317**, vs R6 0.9367 (Δ −0.0050 [−0.0129, +0.0022]).
  - Per finding R7 − R6: Lateral Meniscus +0.014, Fracture +0.016; Synovitis −0.048, Baker's −0.025, PF OA −0.012.
  - R6 and R7 are almost the same ranking: per-finding Spearman 0.967–0.991.
- **C1** (R7 replaces R6 in final-max's compact slot): 0.9386 vs base 0.9391, Δ −0.0005 [−0.0045, +0.0036]. The lower bound is < 0, so **not adopted**.
- Information only (they do not fit 9 h): M1 (the brief's 0.35 R6 + 0.35 R7 + 0.15 S6 + 0.15 s6fall) 0.9384, Δ −0.0007 [−0.0039, +0.0024]; R6R7 0.9392, Δ +0.0001 [−0.0020, +0.0025].
- final-max v2 keeps R6 (action A unchanged).

### 14b. A result: final-max v2 submitted (2026-10-21 07:24 IST, submission 57460890)
- **Gold_58 fidelity** (`task1/r6x/xpass_check.json`; `xpass_gold_check.py`, all 58 studies):
  - (i) R6 through `compact_infer_x.py` vs the plain worker: max |dp| **0.0**.
  - (ii) In-pass s6fall vs s6fall's own pipeline: identical on 57/58 studies (max |dp| < 1e-9).
    - The other study (…910529174803) differs by up to 0.128. It is the study R6's side classifier re-renders; s6fall's own stage has no side check.
    - Minimum per-finding Spearman 0.9944; macro AUC 0.9316 in-pass vs 0.9314 native (Fracture 0.904 vs 0.901).
    - **Pass.**
  - Locally the worker exits 1 after writing every output: its receipt hashes `px_cache.py` etc. next to itself, and locally the patched file lives in `src/emergency`. On Kaggle they share SRC and the receipt is written; this does not affect the gate.
- **(iii) Kaggle commit run** (kernel version 2, pushed 06:43 IST; 3 public test studies):
  - `S6F_INPASS` 3/3 (s6fall GPU 27.7 s, render 4.4 s); R6 3/3 with no fallback; `S6F_IN_BLEND used transform t2`; `NOTEBOOK_COMPLETE`.
  - v2's `submission.csv` is **identical** to v1's commit-run file (max |d| 0.0). In-pass s6fall vs v1's separate stage: 4.7e-10.
  - The notebook took 539 s, vs about 665 s for v1.
  - **Pass.**
- **Submitted 07:24 IST (submission 57460890)**: "final-max v2: v27 (nartaa coatnet + heliosli r6) + nartaa s6fall scored inside the r6 pass".
  - Expected display 0.958–0.959. Gold says T2 vs v27's own blend is +0.0007, and the LB shows three decimals.
  - If v2 shows ≥ v1's 0.958, it becomes final pick 1. Otherwise v1 stays. Whether s6fall ran cannot be seen from outside, so v2 is the version where it can.

### 14c. C, teacher check (2026-10-21 07:32 IST; `track2/s6_oof_cache_check.json`)
- S6's held-out fold on cached 320-px windows resampled to 384 (up to 160 per study), vs S6's native held-out logits (full 384 render, every window), on oof_300.
- Per-finding Spearman **0.950–0.983**, minimum Fracture 0.9503. That is ≥ 0.85, so the registered bar passes and the full run goes ahead.
- Logits shift on the resampled input (mean |Δz| 0.8–1.3). The v5 targets use only the teacher's ranking (quantile map onto P_report), so the shift does not enter them.

### 14d. C result: s6ft_v2 NOT adopted (2026-10-21 11:55 IST; `task1/s6ft_v2_gold_eval.json`)
- **Teacher, full run:** 4,349 cached studies in 100 min (`track2/s6_oof_cache.csv`).
- **Targets v5:** 4,349 studies (`data/labels_v4/training_labels_v5_clean.csv`).
  - Teacher vs report Spearman per finding is 0.43–0.75; the mean |target change| is 0.025–0.070.
  - Checked 11:45 IST: none of the 58 gold studies is in the targets, the cache index or the cache files.
- **Training:** 2 epochs from s6fall, 4,348 steps, 151 min, loss 0.298 → 0.285. EMA weights `D:/rsna_work/s6ft_v2_local/s6ft_ema.pt`; no checkpoint choice.
- **Gold_58 leg:** S6FT_v2 **0.9366** vs s6fall 0.9314 (+0.0052). The first s6ft (13d) scored 0.928.
  - 9 of 12 findings are better.
  - Effusion +0.021, Synovitis +0.027, Contusion +0.017.
  - ACL −0.012, Fracture −0.015, Lateral Meniscus −0.003.
- **Blends (rule of section 13, 10,000 paired resamples):**

  | Pick | Variant | Gold | Δ vs base [95 % CI] | Pass |
  |---|---|---|---|---|
  | final_max (base 0.9391) | V1, S6FT replaces s6fall | 0.9398 | +0.0007 [−0.0009, +0.0025] | no |
  | final_max | V2, s6fall/S6FT split | 0.9392 | +0.0001 [−0.0009, +0.0011] | no |
  | final_strict_v2 (base 0.9361) | V1 | 0.9370 | +0.0010 [−0.0016, +0.0043] | no |
  | final_strict_v2 | V2 | 0.9369 | +0.0008 [−0.0008, +0.0027] | no |

- **Outcome:** no variant has a lower bound ≥ 0, so **s6ft_v2 is not adopted**.
  - There is no final-max v3.
  - Action D does not apply: final-strict-v2 is not built.
  - The picks follow 14b: pick 1 = v2 if it shows ≥ 0.958, else v1 (57444580). Pick 2 = final-strict (57406523).
  - A v3 would also add a second 288-px forward per window to R6's stage on the hidden test, for an expected gain below the LB's display step.

### 14e. final-max v2 scored 0.958 (seen 2026-10-21 13:19 IST)
- Submission 57460890 is COMPLETE, public **0.958**, the same display as v1 (57444580).
- It was submitted 07:24 IST and was complete by the 13:19 check (pending at 13:11), so it took ≤ 5.9 h, queue included. It fits the 9 h limit with room.
- **By the 14b rule (v2 ≥ 0.958), v2 is final pick 1.** It is the version whose s6fall leg can actually run on the hidden test.
- **Final picks:** final-max v2 (57460890, 0.958) + final-strict (57406523, 0.956). The owner ticks both before Fri 23 Oct 05:29 IST.
- Nothing else is registered. R7 (14a) and s6ft_v2 (14d) were not adopted.
