# Combining Multiple Pseudo-Label Sources into a Trustworthy Consensus

## 1. Problem

By this point in the project, the 12-pathology knee MRI labels had been generated
multiple times, by different methods, landing in different files under
`data/main_data/to_use/`:

| Source | File | What it is |
|---|---|---|
| `gpt56sol` | `report_labels_gpt56sol.csv` | LLM-based verdict + confidence extractor |
| `report_v2` | `report_labels_v2.csv` | Rule-based verdict/confidence extractor, continuous scores |
| `qwen14b_awq` | `train_folds_qwen14b_awq_v2.csv` | Qwen2.5-14B-Instruct-AWQ via vLLM |
| `qwen7_5b` | `train_folds_sota_qwen25.csv` | Qwen2.5-7B-Instruct via vLLM |
| `public_pseudo` | `train_folds_with_pseudo.csv` | Community/public Kaggle pseudo-labels |
| (dropped) | `labels_llm_gpt56sol.csv` | Byte-identical to `report_labels_gpt56sol.csv` on every row checked — same source as `gpt56sol`, would silently double its vote if included |

All 5 usable sources cover the same ~4,407 studies, of which 58 have real human
ground truth (`available_labels.csv`). The question: how do you combine 5
independently-generated, imperfect label sets into one label set you can actually
trust, rather than just averaging them and hoping for the best?

## 2. First finding: three of the five sources are not what they look like

Before any aggregation could happen, the 58 GT rows in each source file were
checked against `available_labels.csv` directly. Three sources
(`qwen14b_awq`, `qwen7_5b`, `public_pseudo`) turned out to be **100% identical**
to ground truth on those 58 rows — not because the model got them right, but
because the scripts that generated those files use a `combine_first()`-style merge
that **preserves the original ground truth for known studies and only fills in
model predictions for the unlabeled ones**. On those 58 rows, the file literally
contains the answer key, not a prediction.

This matters a lot: any calibration or reliability estimate computed from those
58 rows for those 3 sources would be circular — the source would look "perfect"
simply because it's the ground truth. `gpt56sol` and `report_v2`, by contrast,
are genuine, independent predictions: `gpt56sol` matches ground truth on ~80% of
individual cells (and only 6/58 rows match on all 12 columns at once), and
`report_v2` matches on 0/57 rows exactly — expected for a continuous-valued
scorer being compared row-wise, and confirmation it isn't a copy of the answer key.

Every method described below had to be built leakage-aware: a source flagged as
**leaked** never gets its own reliability/weight estimated from the 58 GT rows,
only from the ~4,349 studies where its votes are real. A source flagged
**genuine** gets to use its real votes on all 4,407 rows, including the 58 GT ones
as extra supervision.

> **Update (Section 3 below):** `qwen14b_awq` and `qwen7_5b` were later upgraded
> from leaked to genuine by regenerating real predictions for just the 58 GT
> rows. Only `public_pseudo` remains leaked as of the current pipeline.

## 3. Closing the leakage gap for two of the three leaked sources

`qwen14b_awq` and `qwen7_5b` were generated locally with models the project
controls (Qwen2.5-14B-Instruct-AWQ and Qwen2.5-7B-Instruct via vLLM) — unlike
`public_pseudo`, which is a third-party file with no way to regenerate it. That
meant genuine predictions for the 58 GT rows could simply be regenerated and
patched in, using the exact same model configs, prompts, and parsing logic as
the scripts that produced the rest of each file (otherwise the reliability
estimate this unlocks wouldn't represent the same generation process used for
the other ~4,349 rows).

**Scripts** (`src/label_generation/`):
- `regenerate_gt58_qwen14b.py` / `regenerate_gt58_qwen7b.py` — two plain,
  self-contained Kaggle-cell scripts (Cell 1 / Cell 2), each loading one model,
  generating on just the 58 GT reports, and saving its own CSV with a download
  link. Cell 1 ends with a best-effort GPU cleanup (`del` + `gc.collect()` +
  `torch.cuda.empty_cache()` + vLLM's distributed-state teardown) so Cell 2 has
  a decent shot at loading in the same kernel session; if it still OOMs, a
  kernel restart before Cell 2 is the fully reliable fallback.
- `patch_gt58_genuine_predictions.py` — run locally. Backs up
  `train_folds_qwen14b_awq_v2.csv` / `train_folds_sota_qwen25.csv` to `.bak`
  copies, then overwrites *only* the 58 GT rows' 12 label columns with the
  regenerated predictions (`Report`, `fold`, and the other ~4,349 rows are left
  untouched). Prints a before/after leakage check so the patch's effect is
  visible immediately rather than assumed.

**Result:** on the next `consensus_label_aggregation.py` run, the leakage check
showed `qwen14b_awq` at 3.4% identical to ground truth and `qwen7_5b` at 5.2% —
both now genuine. `public_pseudo` remains the only leaked source (100%
identical, unpatched). This took the pipeline from **2 genuine sources to 4**,
which meaningfully changed which aggregation method looks best (Section 4.6).

## 4. Methods tried, in order

### 4.1 Naive ensembling (rejected)

The first script (`calibrate_and_ensemble.py`) attempted straightforward
calibration + weighted averaging. It hit the leakage problem head-on: with the
`train_folds_*` sources' 58-row values being ground truth copies, there was no
honest way to calibrate them from that script's inputs alone. This is what led to
building the leakage auto-detection that every subsequent method reuses.

### 4.2 Dawid-Skene EM (diagnostic; revisited in 5.6)

`consensus_label_aggregation.py` implements Dawid & Skene's 1979 EM algorithm:
each source gets an estimated sensitivity/specificity per pathology, learned
iteratively from how often it agrees with a re-estimated consensus across all
4,407 studies. This sidesteps the leakage problem structurally (a leaked source's
parameters are estimated only from its genuine votes, excluding any known-GT
rows it doesn't actually have real predictions for), and lets genuine sources use
their real 58-row votes as extra supervision.

With only 2 genuine sources (before Section 3's fix), **it underperformed a plain
average on every one of the 12 pathologies**. Two reasons, confirmed empirically:

1. **Correlated sources break Dawid-Skene's core assumption.** `qwen14b_awq` and
   `qwen7_5b` are the same model family; `public_pseudo` may itself be LLM-derived.
   EM assumes independent errors — when sources are correlated, EM rewards a
   source for agreeing with a consensus it helped build, inflating its estimated
   reliability regardless of true accuracy. The smoking gun: `public_pseudo`'s
   estimated sensitivity came out at a suspiciously **flat 0.98 across all 12
   pathologies** (real sources vary a lot by class — `gpt56sol` ranged 0.66–0.98),
   despite `docs/label_generation_walkthrough.md` documenting ~60% real-world
   accuracy for that exact source. This pattern persisted unchanged even after
   Section 3's fix (`public_pseudo` is still leaked, and still shows the same
   flat 0.98) — strong, repeated confirmation this is a real property of leaked/
   correlated sources, not a fluke of the 2-genuine-source run.
2. **Too few genuine raters.** With only `gpt56sol` and `report_v2` as validated
   sources, EM barely had enough independent signal to estimate per-source
   confusion matrices reliably — not enough to earn back the variance it adds
   over a plain average. Section 4.6 shows this diagnosis was directly
   confirmed: once genuine sources went from 2 to 4, EM started actually winning
   some pathologies.

### 4.3 Logistic-regression stacking (diagnostic)

A simple meta-learner: train `LogisticRegression` on the 58 GT rows using each
genuine source's raw score as a feature, cross-validated (`StratifiedKFold`) for
an honest out-of-fold AUC. This is the "train an SVM/KNN-like classifier on the
sources" idea explored when asked about clustering/KNN/SVM approaches — a linear
model was used instead of SVM/KNN specifically because with n=58, a 2-3 parameter
model overfits far less than something with more effective degrees of freedom.
It has never won a single pathology in either the 2-genuine-source or
4-genuine-source comparison.

### 4.4 Validated-AUC-weighted average (kept, now one of two blend inputs)

A straightforward weighted average where genuine sources are weighted by their
own just-measured AUC-above-chance on the 58 GT rows, and leaked sources get a
small, fixed, manually-configured trust weight (`LEAKED_SOURCE_TRUST` in the
script) since their reliability can't be measured honestly and EM's self-estimate
for them isn't trustworthy either. This beat both Dawid-Skene and the stacker in
the 2-genuine-source comparison, and was the primary consensus for a while.

### 4.5 CROWDLAB — cleanlab (kept, now one of two blend inputs)

`cleanlab.multiannotator` implements CROWDLAB, which combines the raw annotator
votes with an auxiliary classifier's out-of-sample predictions. The auxiliary
classifier (`LogisticRegression`) is trained via `cross_val_predict` (5-fold) on
each source's raw continuous score, using the **majority-vote label as its
training target** — this needs no ground truth to run, since the majority vote is
computed purely from the 5 sources' own votes across all 4,407 studies.

With 2 genuine sources, CROWDLAB won the honest comparison on **7 of 12
pathologies**, concentrated on the harder, lower-scoring classes (Synovitis,
Fracture, Contusion, PF OA). More convincingly than the AUC scoreboard alone: on
the **full 5-source run**, CROWDLAB's own `annotator_quality` score for
`public_pseudo` came out at **0.08–0.58**, varying sensibly by class — correctly
identifying it as the weakest source, consistent with its documented ~60%
real-world accuracy, where Dawid-Skene concluded the opposite (flat 0.98) on the
same source. This became the primary consensus after this step.

### 4.6 Revisiting the primary once 4 genuine sources were available

After Section 3's regeneration, the honest comparison was rerun with
`gpt56sol` + `report_v2` + `qwen14b_awq` + `qwen7_5b` (4 genuine sources instead
of 2). The race tightened substantially — see the table in Section 6. **No
method is clearly dominant anymore.** Given that, the primary consensus was
changed from "pure CROWDLAB" to a **50/50 blend of CROWDLAB and the
validated-AUC-weighted average**, both computed on the full 5-source annotator
matrix. Blending is the more statistically prudent default when two methods are
roughly tied on a 58-row benchmark, rather than fully committing to whichever one
happens to win the noisy race in either direction. Dawid-Skene and the stacker
remain diagnostic-only, kept for QA/comparison but not blended into the primary.

## 5. Honest per-pathology AUC comparison

### 5.1 With 2 genuine sources (`gpt56sol` + `report_v2`)

| Pathology | Simple mean | Dawid-Skene EM | CROWDLAB | Stacker |
|---|---|---|---|---|
| ACL | 0.9859 | 0.9026 | 0.9835 | 0.9749 |
| MCL | 0.9819 | 0.9286 | **0.9909** | 0.9501 |
| Medial Meniscus | 0.9369 | 0.9020 | 0.9309 | 0.9081 |
| Lateral Meniscus | 0.8795 | 0.8522 | 0.8689 | 0.8447 |
| Medial OA | 0.9163 | 0.8837 | **0.9240** | 0.8721 |
| Lateral OA | 0.7892 | 0.8308 | 0.7814 | 0.7544 |
| PF OA | 0.8893 | 0.7915 | **0.9093** | 0.8320 |
| Effusion | 0.8503 | 0.6422 | **0.8565** | 0.7553 |
| Synovitis | 0.6870 | 0.6840 | **0.7258** | 0.5854 |
| Baker's | 0.9312 | 0.9031 | 0.9221 | 0.8768 |
| Contusion | 0.8367 | 0.7605 | **0.8516** | 0.8097 |
| Fracture | 0.8590 | 0.8069 | **0.8861** | 0.7632 |

CROWDLAB won (bold) on 7/12; simple mean won the remaining 5, all by small
margins in the already-high-scoring classes (≥0.88).

### 5.2 With 4 genuine sources (`gpt56sol` + `report_v2` + `qwen14b_awq` + `qwen7_5b`)

| Pathology | Simple mean | Dawid-Skene EM | CROWDLAB | Stacker |
|---|---|---|---|---|
| ACL | 0.9828 | 0.9491 | **0.9865** | 0.9681 |
| MCL | 0.9354 | **0.9467** | 0.9422 | 0.9388 |
| Medial Meniscus | **0.9297** | 0.9093 | 0.9014 | 0.8978 |
| Lateral Meniscus | **0.8752** | 0.8621 | 0.8739 | 0.8410 |
| Medial OA | **0.9233** (tie) | 0.9140 | **0.9233** (tie) | 0.8938 |
| Lateral OA | 0.8433 | **0.8762** | 0.8462 | 0.8037 |
| PF OA | 0.8719 | 0.8378 | **0.8835** | 0.8359 |
| Effusion | 0.8006 | 0.7950 | **0.8019** | 0.7602 |
| Synovitis | **0.7730** | 0.7336 | 0.7467 | 0.6930 |
| Baker's | 0.9275 | 0.9194 | **0.9339** | 0.8904 |
| Contusion | **0.8536** | 0.8313 | 0.8509 | 0.8111 |
| Fracture | 0.8639 | 0.8444 | **0.8847** | 0.7847 |

Tally: **CROWDLAB wins 5/12** (ACL, PF OA, Effusion, Baker's, Fracture),
**simple mean wins 4/12** (Medial Meniscus, Lateral Meniscus, Synovitis,
Contusion), **Dawid-Skene EM wins 2/12** (MCL, Lateral OA) — up from 0/12 with
only 2 genuine sources, directly confirming the too-few-raters diagnosis from
Section 4.2 — and Medial OA ties exactly between mean and CROWDLAB. The stacker
never wins a single pathology in either comparison.

A 50/50 blend of mean + CROWDLAB is also computed and printed
(`consensus_label_aggregation.py`'s "50/50 blend (mean + CROWDLAB)" line) but its
exact honest-AUC numbers weren't captured into this table — check the console
output of your own run for current figures.

## 6. Caveats

- **n=58 is small.** Per-class AUC differences of a few points, especially in the
  high-scoring classes, may not be statistically meaningful — several pathologies
  in Section 5.2 differ from Section 5.1's method ranking by margins well inside
  plausible noise for this sample size.
- **`report_v2` is missing 1 of the 58 GT studies** (57/58 covered) and covers
  4,406/4,407 studies overall. Handled by masking it out of per-source AUC
  calculations for that row, and by neutral-0.5 imputation before anything that
  can't accept `NaN` (e.g. `LogisticRegression`).
- **`LEAKED_SOURCE_TRUST` weights (for `public_pseudo` only, now) are manually
  set**, not measured — it's the sole remaining source with no honest
  calibration signal available.
- Dawid-Skene and the stacker are still computed and saved (see below) — not
  because they're recommended, but so regressions or disagreements are visible if
  the underlying data changes.

## 7. Final script and output

**Script:** `src/label_generation/consensus_label_aggregation.py`
**Output:** `data/main_data/to_use/train_folds_consensus.csv`

Per pathology (e.g. `ACL`), the output contains:

| Column | Meaning |
|---|---|
| `ACL` | **Primary.** 50/50 blend of `ACL__weighted_mean` and `ACL__crowdlab`. |
| `ACL__weighted_mean` | Validated-AUC-weighted average, full 5-source run — one of the two blend inputs. |
| `ACL__crowdlab` | CROWDLAB consensus, full 5-source run — the other blend input. |
| `ACL__dawid_skene` | Dawid-Skene EM consensus, diagnostic only, not blended into the primary. |
| `ACL__stacker` | Logistic-regression stacker, diagnostic only, not blended into the primary. |
| `ACL__disagreement` | Std-dev of raw scores across all 5 sources — flags rows where sources disagree most, for potential down-weighting or manual review before trusting them as soft labels. |

The 58 known-GT studies always use the true label directly in every one of these
columns (never model-inferred), and a fresh `StratifiedGroupKFold(n_splits=5)` is
computed on the final blended targets rather than inherited from any single
source file.
