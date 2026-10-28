# Final emergency competition report: RSNA Knee Abnormality Detection

Status: **updated 2026-10-20 (score seen 05:37 IST)**. Both A variants were submitted on 2026-10-19 at 11:19 IST with the owner's approval and both scored **0.954** on the public LB (measured). final-strict (A-strict plus nartaa's s6fall leg) was submitted at 00:35 IST on 2026-10-20 with the owner's approval and scored **0.956** (measured). By the rule fixed before it ran (plan §11), **the final picks are final-strict + A-strict+**. The owner's team ranks 1,983 of 6,239 (§9). A local follow-up study on gold-58 and 300 held-out studies (seen by 08:21 IST) adopted no upgrade, so there is no final-strict-plus (§11). The owner still has to tick the two final picks by Thu 22 Oct 23:59 UTC.

Evidence labels:
- **measured**: on this machine, or on the official leaderboard (LB);
- **estimate**;
- **third-party**: stated by other competitors or read in their public code;
- **hypothesis**: not measured.

## 1. What the current model computes

Full derivation: `current_model_mathematical_deep_dive.md`.

**Input.**
- Per study, 3 series: the fat-suppressed sagittal, coronal and axial series (non-FS fallback).
- Each series: 1st/99th-percentile intensity window → 224 × 224 squash (about 0.71 mm/px).
- 8 windows of 3 slices about 3.5 mm apart, laterality-normalised (right knees mirrored).

**Model.**
- ConvNeXt-T encodes each window (768 → 384) and adds slot and slice-position embeddings.
- 12 finding-specific softmax attentions pool the 24 window tokens.
- One linear head per finding: z_c = Σ_j α_{c,j} ⟨t_j, w_c⟩ + b_c, an attention-weighted average of per-window scores.

**Training.**
- Soft BCE against report-derived labels (labels_v3, 0.890 gold agreement).
- AdamW, cosine schedule, 8 epochs, EMA 0.999.

**Submission.** The mean of 3 seeds × 2 shifted window bags.

## 2. The three most consequential limitations

1. **Labels.** Report-derived targets disagree with image reads (0.890 gold agreement; synovitis 0.68–0.76).
2. **No external data.** Every public notebook at ≥ 0.950 uses nartaa's CoAtNets trained with 2,399 extra OAI knees (third-party). Our models use competition data only.
3. **Evaluation.** Pseudo-val measures agreement with report labels. Gold-58 is too small to separate close candidates. Two runs of nearly the same recipe differ by 0.006 on pseudo-val (§4), so most single-change effects cannot be resolved.

Resolution (0.71 vs 0.365 mm/px) was tested directly and is not a measurable limitation on its own (§4, E9).

Evidence: deep-dive §8; plan §8.

## 3. Research approaches investigated

See `competition_and_medical_imaging_research.md`:
- public solutions (CoAtNet 2.5D MIL; OAI external data; pseudo-labels);
- MRNet and attention-MIL;
- 2.5D vs 3D;
- backbones;
- medical and self-supervised encoders;
- noisy labels, pseudo-labels and PU learning;
- efficient inference;
- provenance and rule compliance of the public pipelines (§1 of that report, updated 2026-10-19 04:20 IST).

## 4. Experiments

All values are **measured**. Full table: `output/emergency_research/experiments.csv`.

Pseudo-val is fold 0 (870 studies, report labels). Δ is a paired bootstrap against E1p, with its 95% CI.

| Experiment | Change | Result | Decision |
|---|---|---|---|
| E1p (reference) | v2 224 px, K = 8, position fix, 2 bags | 0.8808 (confident-label 0.9231) | reference |
| E1s2 (noise scale) | another seed and the older index | Δ +0.0062 [+0.0026, +0.0098] | **Single-run differences under about 0.006 are not meaningful** |
| ALLW | every slice at inference | Δ +0.0006 [−0.0016, +0.0027] | not adopted |
| K = 4 / K = 6 | fewer windows | Δ −0.0206 / −0.0048 | rejected |
| E9 (HR) | 0.365 mm/px input, crop 320, K = 6/12 | Δ +0.0048 [+0.0005, +0.0090] | ensemble family. The gain is inside the E1s2 seed spread, so not a proven single-model gain |
| E1p + E9 | two families | Δ +0.0081 [+0.0057, +0.0104] | Same-size two-seed v2 ensemble (E1p + E1s2): Δ +0.0072 [+0.0051, +0.0092]. HR adds about what another seed adds |
| E9 at 352 field | larger inference field | Δ −0.0007 [−0.0016, +0.0001] vs E9 | not adopted |
| H-SYN1 / H-SYN2 | synovitis label fixes | gold looks negative (0.756 → 0.738 for H-SYN2) | not adopted |
| Fast index | one header per series | identical study index on 50 studies; preprocessing −32 to −42 % | adopted |
| One decode | v2 and hr from one decode | HR member's decode +0.4 s per 50 studies | adopted |
| PL-g r1 | pseudo-labels from OOF teachers | four all-fold students trained overnight (S224_PLg, SHR_PLg ×2 seeds, SHR_rep control); all four completed. Their val numbers are training-fold scores (not a holdout). Packaged with the earlier models as `rsna-knee-v3-all` (8 checkpoints), which passed the 3-study local test | **no LB test**: once the main pick became a public pipeline, these tests stopped feeding any final pick (plan §10c) |
| S1 (LB) | 0.927 release + E9 at 50 % | **public LB 0.928** vs 0.927 | inside the pre-registered ±0.002 noise band |
| ConvNeXt-S | larger encoder | smoke test passed; 4.94 GB at batch 1 | needs the optional professor GPU |

## 5. Best validated model and macro AUC

**Our own models (measured):**

| Model | Pseudo-val macro (fold 0) | Gold-58 macro | Public LB |
|---|---|---|---|
| 0.927 release (F1–F3, all folds) | not measurable: its models trained on fold 0 | 0.9065 | **0.927** |
| E1p (fold-0 single, v2) | 0.8808 | 0.8908 | — |
| E9 (fold-0 single, HR) | 0.8856 | 0.8962 | — |
| E1p + E9 (fold-0 ensemble) | **0.8889** | — | — |
| S1 release (F1–F3 + E9) | — | — | **0.928** |

- Gold-58 was looked at only under pre-registered rules, never to select.
- With 58 studies, its differences of about 0.005 are noise.
- **Our best validated deployable model is the S1 release at 0.928 public**, which is not measurably better than the 0.927 baseline.

**Main-track candidate (not our model):**
- A-strict, heliosli's public pipeline: nartaa CoAtNet 384 (OAI) + heliosli S6.
- Kaggle displays **0.954** on its page (measured by Kaggle; a third-party notebook).
- The owner's own run (submission 57384567, submitted 2026-10-19 11:19 IST) scored **0.954** public (measured). It reproduces the page's score.
- final-strict adds nartaa's s6fall as a third leg (plan §11). The owner's run (submission 57406523, submitted 2026-10-20 00:35 IST) scored **0.956** public (measured). This is the best public score so far.

## 6. Per-finding AUCs

Measured, pseudo-val fold 0 (report labels):

| Finding | E1p | E1p + E9 | Δ | E9 gold-58 | E1p gold-58 |
|---|---|---|---|---|---|
| ACL | 0.8825 | 0.8816 | −0.0009 | 0.907 | 0.936 |
| MCL | 0.8363 | 0.8495 | +0.0132 | 0.950 | 0.909 |
| Medial Meniscus | 0.9184 | 0.9281 | +0.0097 | 0.974 | 0.933 |
| Lateral Meniscus | 0.8469 | 0.8771 | +0.0302 | 0.841 | 0.769 |
| Medial OA | 0.9031 | 0.9108 | +0.0077 | 0.989 | 0.983 |
| Lateral OA | 0.8632 | 0.8701 | +0.0069 | 0.781 | 0.847 |
| PF OA | 0.8432 | 0.8485 | +0.0053 | 0.842 | 0.815 |
| Effusion | 0.8749 | 0.8808 | +0.0059 | 0.918 | 0.937 |
| Synovitis | 0.8962 | 0.9120 | +0.0158 | 0.693 | 0.689 |
| Baker's | 0.9255 | 0.9274 | +0.0019 | 0.975 | 0.967 |
| Contusion | 0.8946 | 0.8979 | +0.0033 | 0.981 | 0.950 |
| Fracture | 0.8845 | 0.8826 | −0.0019 | 0.904 | 0.954 |
| **Macro** | **0.8808** | **0.8889** | **+0.0081** | **0.8962** | **0.8908** |

Synovitis is the weakest finding against image reads (gold about 0.69) despite a high pseudo-val. The report labels are the bottleneck there.

## 7. Fair comparison against the baseline

- **Paired, same holdout (measured):**
  - The best ensemble beats the baseline recipe by +0.0081 [+0.0057, +0.0104] on pseudo-val.
  - A two-seed ensemble of the baseline recipe alone gives +0.0072, so almost all of the gain is ensembling.
  - The HR-specific part, +0.0009, is noise.
- **LB (measured):** 0.927 → 0.928. That is inside the ±0.002 band fixed before the result.
- **Conclusion:** no change to our own model improved it measurably beyond the 0.927 baseline. The large available gain (+0.027) comes from public pipelines built on external OAI data, which the owner approved on 2026-10-19 (§9).

## 8. Measured inference runtime

All local measurements: RTX 4050 laptop, i5-13420H, 4 workers, 50 fold-0 studies, partly warm OS cache.

| Release | s/study | min per 1,300 studies |
|---|---|---|
| 0.927 release as submitted | 1.89 | 41.0 |
| 0.927 release + fast index | 1.41 | 30.6 |
| S1 (+ E9, fast index, one decode) | 1.80 | 39.1 |
| F1 alone, one bag, fast index (candidate B) | 1.08 | 23.3 |

- About 70–80 % of the time is DICOM reading and decoding; the network costs 0.02–0.38 s/study.
- **Kaggle timing is not observable.** The scored run's duration is not reported, and the efficiency LB shows ranks only. Kaggle's T4s, CPU count, cold disk and compressed DICOMs can all make the real run slower.
- **Public pipelines (third-party):** the 0.958 notebook's author expects 6.5–8 h on the hidden test. heliosli's notebooks bound their second stage with an 8.3 h deadline.

## 9. Recommended main-track submission

**Candidate A: fastMRI-free variants of the public OAI-trained pipeline** (plan §10c; final-strict added by plan §11):

| Variant | What | Score |
|---|---|---|
| **final-strict** | A-strict plus nartaa's all-data S6 ConvNeXt (s6fall, gold excluded from its training) as a third leg: Raptor 0.5 / S6 0.25 / s6fall 0.25 (`notebooks/candidate_A/final_strict/`, built by `src/emergency/build_final_strict.py`) | **0.956** (measured: owner's submission 57406523, seen 2026-10-20 05:37 IST). Predicted beforehand as 0.955–0.956 |
| **A-strict** | heliosli's `rsna-knee-blend-gold-dataset-restored`, unchanged: nartaa CoAtNet 384 + heliosli S6, 70/30 | **0.954** (measured: owner's submission 57384567; the page also displays 0.954) |
| **A-strict+** | goodpjw2008's 0.958 notebook with its R6 stage replaced by S6. Adds nartaa's 224 checkpoint at 35 % of the Raptor slot and goodpjw2008's reader at 1/10 (`notebooks/candidate_A/a_strict_plus/`, built by `src/emergency/build_a_strict_plus.py`) | **0.954** (measured: owner's submission 57384570). Predicted beforehand as 0.954–0.955, most likely 0.954 |

- **Rule (fixed before either run):** A-strict+ is the main pick if it completes and displays ≥ A-strict's score; A-strict is then the second pick.
- **Applied (2026-10-19 17:32 IST):** both completed and display 0.954, so **A-strict+ is the main pick and A-strict the second.**
  - At three decimals the two cannot be told apart: both true scores lie in [0.9535, 0.9545), so A-strict+'s extra parts are worth less than 0.001 public AUC here.
  - Under Kaggle's standard rule, the better private score of the two selected submissions counts, so the order is bookkeeping only.
- **Superseded by the plan §11 rule (fixed before final-strict ran; applied 2026-10-20 05:37 IST):** final-strict displays 0.956 ≥ 0.955, so **the final picks are final-strict + A-strict+**. A-strict stays the fallback.
  - The +0.002 is one public-split comparison at three decimals, so the true public gap is roughly 0.001–0.003. Whether it holds on the private split is unknown.
- **Leaderboard position now (measured, public LB at 05:44 IST on 2026-10-20, 6,239 teams):**
  - The owner's team ranks 1,983 (0.956 block: ranks 1,684–1,984). The field moved up overnight, so +0.002 roughly kept the rank.
  - #10 0.966; gold (#22) 0.964. Silver (#311) and bronze (#623) both fall inside the 0.958 block (ranks 192–1,502). A medal still needs about 0.959.
- **Leaderboard position (measured, public LB at 17:32 IST, 6,211 teams):**
  - The owner's team ranks 1,979, up from 3,761 with 0.928. The 0.954 block covers ranks 1,951–2,092 (142 teams).
  - #10 scores 0.966 and the gold cut-off (#22) 0.964. The silver (#310) and bronze (#621) cut-offs both fall inside a 1,231-team block at 0.958 (ranks 184–1,414).
  - A medal therefore needs about 0.959 or more, +0.005 over the clean recipe. No fastMRI-free public pipeline found scores above 0.954.
- **Why not the 0.957–0.958 notebooks:**
  - They use heliosli's R6 or R7. R7's manifest sets fastMRI+ auxiliary training.
  - R6's trainer (version 9.8, hash-matched to its manifest) exists to add that task and strips its head from the exported weights. R6 was built with it the same day, from run `cnx_r6fm`.
  - fastMRI's application "must include the investigator's institutional affiliation". The host's test excludes affiliation-limited datasets.
  - Expected cost of the strict choice: about 0.004 public AUC.
- **External data to disclose with A:**
  - OAI, via nartaa's weights, with the NDA acknowledgement kept in the notebook;
  - SKM-TEA, via `anatomy.pt`;
  - public LLM report-label tables, via the models' training.
- **Open risk:** the host has not ruled on OAI's country restrictions (rule 2.6.b).

## 10. Recommended efficiency candidate

**Candidate B = F1 alone, one bag, fast index** (1.08 s/study locally). **It is not recommended as a final pick.**

| Candidate | Public AUC | Runtime (estimate) | Efficiency (estimate; minimise) |
|---|---|---|---|
| A-strict | 0.954 | 6–8 h (third-party, v27-type stack; A-strict has fewer stages) | about −1.36 to −1.14 |
| B | not scored (expected ≤ 0.927) | about 23 min locally | about −1.93 if it scored 0.927 |
| C (0.927) | 0.927 | about 41 min locally | about −1.90 |

- Efficiency = AUC/(Benchmark − max AUC) + RuntimeSeconds/32400, with Benchmark ≈ 0.5 and max ≈ 0.97 (estimates of the leaderboard values).
- The official efficiency LB (measured, 2026-10-17 21:15 UTC) lists ranks and AUCs but no runtimes.
  - The top 10 all have public AUC 0.955–0.965; the top 50 have 0.943–0.965.
  - The best-placed entries at our AUC level (0.927–0.929) sit at efficiency ranks 152–170.
- **The efficiency prize is out of reach for our models**, so the two final picks go to the main track.

## 11. What remains unverified

- **Now measured (2026-10-20): gold-58 study of final-strict's legs (plan §12, §12b).** Each public leg was run locally with its own code.
  - Single legs on gold: Raptor384 0.9229, S6 0.9267, s6fall 0.9314. final-strict's blend: 0.9361.
  - Raptor is the weakest leg on gold, so there is no sign that its gold-selected recipe inflates it.
  - **None of the 6 pre-registered blend upgrades passed** (Δ ≥ +0.0015, CI lower bound ≥ 0, ≥ 7/12 findings up). The largest point gains were:
    - standardised logits: +0.0028 [−0.0011, +0.0071];
    - compartment weights: +0.0021 [+0.0000, +0.0047], 5/12 findings up.
  - At n = 58 a null is the expected outcome, not evidence of no effect.
  - **Mirror TTA on S6, held-out folds (plan §12c):** on 300 non-gold studies, each scored by the S6 fold model that never trained on it, mirror TTA *lowers* macro AUC by 0.0066 [−0.0107, −0.0026]. B stays rejected.
  - **Residual adapter on the S6 and s6fall class tokens (Task 3, plan §12c):** Δ +0.0000 [−0.0003, +0.0004], not adopted.
    - Its fixed L2 weight kept it at the identity (largest weight 0.0005), so the test is uninformative about weaker settings.
    - It was not re-run, to avoid a second look at the same 58 gold studies.
  - **Raptor memorisation on the 300 non-gold studies (descriptive, plan §12c):** every all-data leg scores far higher on the studies it trained on (0.957–0.973) than S6's never-seen fold (0.918). Raptor's lead is the smallest (+0.039; s6fall +0.055). With Raptor also the weakest leg on gold, there is no sign that it memorises more than the S6 family.
  - **So there is no final-strict-plus** (Task 4): nothing passed. No push, no submission, no Kaggle GPU time used.
- **final-strict's s6fall leg on the hidden test: inferred to have run.** Its hidden log is not visible. If s6fall had been skipped or failed, the notebook would have used A-strict's exact blend (tested identical locally), which scored 0.954. final-strict scored 0.956.
- **Now measured (2026-10-19):** both A variants completed on the hidden test within the 9 h limit and scored 0.954 public.
  - Still not observable: whether A-strict+'s 224 stage and reader ran on the hidden test or fell back. Its commit run on the 3-study stub ran every stage (224 stage `rc=0 ok=True`, the reader blended 3 checkpoints, 0 failed studies). The hidden-test log is not available to submitters.
  - Also not observable: how long either hidden-test run took.
- R6's training data: inferred from its trainer and run name; heliosli has not answered the public question.
- Host rulings: none on fastMRI (thread 750259), on OAI's country restrictions, or on OAI-trained public weights.
- Kaggle runtime of every candidate (not reported); private-LB behaviour of everything.
- The four overnight students: no holdout metric and no LB score (only the local 3-study run test). Their contribution, if any, is unknown.
- **GitHub repository:** made private by the owner on 2026-10-19; the history still holds the competition data. Rule 4.b.1 and the MIRA license allow a private research copy, so the owner chose not to rewrite the history. Still open (owner's call): rule 4.b.1's clause on notifying Kaggle of possible unauthorized access during the public period (from 2026-08-30), and any forks made while it was public (`repo_remediation_plan.md`).

## 12. Which submission to try first, which fallback to retain

1. **Done:** A-strict and A-strict+ were pushed as private notebooks `nishkarshsinghal/rsna-knee-a-strict` and `nishkarshsinghal/rsna-knee-a-strict-plus` (version 1 each, heliosli's Docker image). Both commit runs passed every marker in `notebooks/EMERGENCY_SUBMISSION_GUIDE.md`. Both were submitted on 2026-10-19 at 11:19 IST (submissions 57384567 and 57384570). Both scored **0.954** public (measured).
2. **Done:** final-strict was pushed as private notebook `nishkarshsinghal/rsna-knee-final-strict` (version 1) and submitted on 2026-10-20 at 00:35 IST with the owner's approval (submission 57406523). It scored **0.956** public (measured).
3. **Final picks (by the plan §11 rule, applied 2026-10-20 05:37 IST): final-strict (57406523) + A-strict+ (57384570).** A-strict (57384567) is the fallback. **The owner ticks both picks on the Submissions tab by Thu 22 Oct 23:59 UTC (Fri 05:29 IST).**
   - If nothing is ticked, Kaggle normally auto-selects the best public scores. That would take final-strict (0.956) plus one of the two 0.954 entries, and the tie-break between those is not documented. Ticking A-strict+ explicitly keeps the planned pair.
   - Do not submit any R6 or R7 notebook (the 0.956–0.958 ones), not even as a test: it would equal or outscore final-strict and could be auto-selected.

**Fallback C** is retained unchanged and never overwritten:
- the 0.927 release: the owner's notebook `notebook45cfc61c19` and the local frozen copy `D:/rsna_work/fallback_0927/`;
- S1 (0.928, `notebook535502ec96`).

They are the clean, competition-data-only results if the owner decides against external data.
