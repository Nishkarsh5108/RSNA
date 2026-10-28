# Gap discovery audit (2026-10-20 00:15 IST)

Scope: the 72-hour sprint brief (four diagnosed gaps, five audit questions) checked against our code (`src/imaging/`), heliosli's S6 sources (hash-matched to bundle `209a32c8b50a6b1c`: `compact_train.py` 14767c7a…, `px_cache.py` b13e38a7…, `knee_anchor_bank_v2.py` e564807a…), nartaa's Raptor inference source as embedded in A-strict, the two A notebooks and their Kaggle commit logs.

Labels:
- **[M]** measured here;
- **[V]** read in code or logs;
- **[S]** third-party statement, not checked;
- **[E]** estimate.

## 0. Bottom line

1. **Top 10 (0.966) remains out of reach.**
   - No public model, fastMRI or not, scores above 0.958.
   - Nothing below can close the gap of +0.008 to +0.012 in 65 hours.
   - Realistic targets: the strict pick from 0.954 to 0.955–0.956; the max pick at 0.958 or slightly above.
2. **Cheapest real lever (hidden gap H1):** nartaa's public **s6fall** model.
   - It is heliosli's S6 architecture retrained on nartaa's own labels: 0.950 solo [S, discussion 749359]. Weights are public under CC BY-NC 4.0.
   - It is fastMRI-free by construction (S6 pipeline, competition data, SKM-TEA anchors).
   - Public notebooks use it only next to R7 [V, metadata of 3 notebooks]. **Nobody has scored nartaa + S6 + s6fall.** That costs no training.
3. **Retraining S6 ourselves is not feasible in this window (H3).**
   - The training cache alone is about 270–400 GB [E]; 234 GB are free on D:.
   - Uploading it is impossible.
   - Compute is about 20 A10G-hours [E].
4. **What would make a hybrid-label teacher honest (H2):** S6's folds are reproducible, `fold = int(sha1(uid)[:8], 16) % 4` [V]. So S6 out-of-fold predictions on all 4,344 non-gold studies can be computed locally; nartaa's all-data models cannot provide that.
5. **AWS (H5):**
   - G/VT quota is 0 in all three regions [M]; requests are pending (us-east-1, ap-south-1).
   - The existing $80 budget action denies the agent user all IAM permissions but **does not stop a running instance** [M].

## 1. The four diagnosed gaps

### Gap 1: supervision. Partly confirmed; the over-reporting list is wrong.

Prevalence on the 58 gold studies vs report labels (labels_v3 ≥ 0.5 on the 4,349 non-gold studies) [M]:

| Finding | Gold | Reports | Direction |
|---|---|---|---|
| Synovitis | 0.466 | 0.123 | under-reported |
| Fracture | 0.310 | 0.065 | under-reported |
| Lateral Meniscus | 0.397 | 0.155 | under-reported |
| ACL | 0.414 | 0.200 | under-reported |
| Contusion | 0.328 | 0.180 | under-reported |
| Medial Meniscus / MCL / Effusion | 0.448 / 0.155 / 0.603 | 0.414 / 0.147 / 0.594 | about equal |
| Medial OA / Lateral OA / PF OA | 0.259 / 0.190 / 0.362 | 0.357 / 0.235 / 0.446 | **over-reported** |
| Baker's | 0.207 | 0.246 | over-reported |

- The gold set is enriched: every gold study has at least one positive. Gold prevalence should therefore run *above* report prevalence everywhere.
- That makes the OA and Baker's reversals strong evidence of report over-calling, consistent with the strict gold OA rule (≥ 1 cm, > 50 % thickness loss).
- The brief's "over-reported mild findings" list (Effusion, menisci, ACL, MCL, Contusion) is **not** supported. ACL, Lateral Meniscus and Contusion go the other way. **Any hybrid rule should pull down OA and Baker's, not Effusion or the menisci.**
- The per-group hidden-test readout attributed to Danial Zakaria (nartaa) is [S]. It was not checked.
- Our overnight students were distilled from our own 0.88 OOF teachers [V]. True; no OOF vision teacher existed then (H2 makes one possible).
- heliosli handles report-vs-image noise inside the loss rather than by relabelling [V, `compact_train.py`]:
  - the E_lrprior arm treats the report as likelihood-ratio evidence: `−log(σ(z)·LR + σ(−z))`;
  - optional report-state heads map predicted report states to official-label probabilities through frozen per-state priors.

### Gap 2: geometric cropping. Real, but not the main reason for the 0.928 vs 0.949 gap.

- **Ours [V]:** v2 squashes the full field to 224 px; HR crops 140 mm around the image centre.
- **nartaa's 0.949 CoAtNet [V]** also crops **140 mm around the image centre**, with no anatomy model, so anatomy-guided cropping cannot explain its lead. Its differences from us:
  - OAI pretraining [S];
  - native 384 px with a 320 crop;
  - 96 slices in five fixed slots, fat-suppressed and not;
  - slices spread over 6–94 % of each stack;
  - CoAtNet-2 capacity;
  - a blended label set.
- **S6 [V]:**
  - SKM-TEA joint and patella anchors;
  - plus the full field of view;
  - 100 mm locals: joint centre, patellofemoral and posterior (sagittal/axial); medial and lateral 30 mm either side (coronal).

### Gap 3: one series per plane. Confirmed.

- Our s3 keeps one fat-suppressed series per plane [V].
- nartaa uses five slots, including non-FS sagittal and coronal [V].
- S6 uses **every** non-duplicate series [V], with slot, view and geometry embeddings and per-class attention over all windows.
- Our E4 six-series test lost (0.8791 vs 0.8817) [M, earlier].

### Gap 4: both picks are S6-based. Confirmed.

- A-strict+ adds only small-weight legs (Q5), so the two picks are nearly redundant.

## 2. The five audit questions

**Q1. How S6 builds its input [V].**
- **Views (v4), per non-duplicate series:**
  - full field (id 0);
  - sagittal/axial: joint-centred 100 mm local (1), patellofemoral local (4, at the segmented patella, else 40 mm anterior), posterior local (5);
  - coronal: medial (2) and lateral (3) locals 30 mm either side of the joint.
- Anchors come from `anatomy.pt`, an SKM-TEA V-Net: the centroid of femoral/tibial cartilage and meniscus on 7 sagittal slices about 4 mm apart, gated by a slab prior from the header geometry.
- **Pixels:**
  - 384 px cache; `series_companded` intensity mapping; INTER_AREA resize;
  - stacks thinner than 1.75 mm are averaged into 3.5 mm slabs.
- **Windows:** 3 slices at a physical step of about 3.5 mm.
  - Training: 176 windows per study, cover-sampled with a random phase.
  - Inference: every slice as a centre (eval_stride 1).
- **Model:**
  - `convnext_small.fb_in22k_ft_in1k`, fully fine-tuned (the freeze option applies only to the ViT backbone), crop 352;
  - meta embeddings (slot, view, relative position, log spacing, extent);
  - per-class softmax attention over windows; spatial attention and OA-extent heads (the logs show `spatial_models 4, extent_models 4`).
- **Side:** DICOM tags, then sagittal IPP x (±20 mm), then a side classifier (`side_cls.pt`) for untagged studies.
- **Training:** 4 hash folds; gold never trained on; 16 epochs; batch 2 studies; lr 1e-4 (encoder) / 1e-3 (head); wd 0.05.
- **Missing from our `model.py`:** all-series input with view embeddings; anatomy-anchored locals; physical-step windows; 384 px; ConvNeXt-small in22k; LR/state losses; side classifier.

**Q2. nartaa's CoAtNet [V].**
- `coatnet_rmlp_2_rw_384`. Five slots: Sag FS 26, Sag non-FS 22, Cor FS 18, Cor non-FS 12, Ax 18 = 96 slices.
- Series are chosen by the series table's fluid-sensitive flag (no description parsing), with slices evenly spread over 6–94 % of the stack.
- 140 mm crop at the image centre, native 384, 320 input, ImageNet normalisation, 94 eval windows.
- **Anatomical L/R mirror TTA:** sagittal reverses the slice channels; coronal/axial flip the width.
  - It is already active in A-strict's Raptor stage (`gpu_windows(..., mirror=True)`), so no TTA is left unused there.

**Q3. Bilateral studies and orientation.**
- Discussion 733343 was not re-read here [S].
- No pipeline detects a two-knee field explicitly [V, inferred from code]:
  - S6's anchor picks one joint;
  - nartaa and we crop the centre.
- That is at most about 7 of 4,407 training studies (0.16 %), negligible for AUC.
- Laterality:
  - ours: majority tag, else geometry (97.8 % agreement with the tag [M, earlier]);
  - S6: tag + position + image classifier (18 training studies had a tag contradicted, per its comments) [V].

**Q4. Rank blending.**
- A-strict blends per-finding percentile ranks (`rank(pct=True)`, ties averaged): 0.7 Raptor / 0.3 S6 [V].
- Probit-rank or log-odds blending changes the relative pull at the tails. Checking that needs held-out predictions of every leg, and nartaa's models are all-data, so none exist for them.
- goodpjw2008 tuned per-finding weights on gold-58 and *lost* LB (0.958 → 0.957/0.955) [S].
- **Recommendation:** keep fixed public weights and fit nothing on gold-58.

**Q5. Why A-strict+ did not beat A-strict [V + M].**
- The reader's share is `0.15 × OWN_W_PER_TARGET`, at most 3.5 % of any finding: Baker's 0.035; others 0.008–0.03.
- The 224 checkpoint gets 0.35 of the Raptor slot (24.5 % overall). It is the same model family on the same labels as the 384 model, so highly correlated.
- Neither dilutes the blend. Neither can move a three-decimal score [M: both 0.954].

## 3. Hidden gaps found

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| H1 | nartaa's **s6fall** (S6 architecture on JEV-blend labels, 320 px cache cropped to 288, 64 windows/study, 16 epochs, all 4,263 report-labelled studies, gold excluded) scores 0.950 solo | [S] 749359; [V] notebook metadata | Zero-training third leg for the strict pick. Its labels differ from heliosli's targets (label diversity). Only used publicly with R7 (Aleksei/Laimer four-model, 0.957 [S]) |
| H2 | S6 fold = `sha1(uid)[:8] % 4` over the non-gold pool; gold = mean of 4 folds, never trained | [V] `compact_train.py` l.654 | Real OOF vision teacher for Track 1; fastMRI-free |
| H3 | An exact S6 retrain needs a cache of about 24,371 series × 3.7 views × 30 slices × 147 KB ≈ 400 GB at 384 px (about 270 GB for nartaa's 320 px variant) | [M] series count; [E] slices/size | Not buildable on 234 GB free, not uploadable; on-the-fly rendering is CPU-bound |
| H4 | Over-reporting is OA/Baker's, not Effusion/menisci | [M] §1 | Re-target the hybrid rule |
| H5 | AWS: quota 0; the budget action is IAM-deny only (instances keep running); no S3 permission, so data goes by scp | [M] | Per-instance stop action + on-instance timers are mandatory. The idle watchdog and CPU alarm would kill the instance during a long upload unless the upload keeps it alive |
| H6 | Both A stages already shard across both T4s (`[gpu cuda:0]/[gpu cuda:1]`, worker0/worker1) | [V] commit logs | No multi-GPU work left to do |
| H7 | v27-type stacks run 6.5–8 h | [S] | Each added S6-type stage (about +15–40 min [E]) needs the time guards |

## 4. Revised priorities (expected gain per hour)

1. **Strict pick: A-strict + s6fall.** No training. Expected 0.955–0.956 [hypothesis]. Build and commit-test now; one LB test.
2. **Max pick: the owner's R6/R7 choice.** Start from v27 (0.958, public) unchanged; optionally + s6fall at a small share.
   - Risk: a fastMRI ruling against R6/R7 is a team-level matter. It may not leave the strict pick standing.
3. **Training track: only if a GPU arrives by Tuesday evening.** The local S6-OOF run (H2) comes first.
   - A new model must reach about 0.95 solo to add anything (third-party evidence: 0.93 legs cost about 0.001).
   - No path to 0.95 fits the remaining time without the 270–400 GB cache.
4. **No per-finding weight fitting on gold-58** (Q4).

**Outcome (2026-10-20, between 00:18 and 00:27 IST):**
- The AWS quota request was declined. No cloud GPU, so the training track is closed.
- The owner declined the R6/R7 max pick. Both final picks stay fastMRI-free.
- final-strict (priority 1) was built and passed 4 local blend tests. It was pushed as a private notebook at 00:27 IST with the owner's approval. Its commit run passed every marker, and it was submitted at 00:35 IST (submission 57406523). Pick rule: plan §11.
- **Scored 0.956** (seen 05:37 IST), at the top of the predicted 0.955–0.956. Final picks by plan §11: final-strict + A-strict+. The owner's team ranks 1,983 of 6,239 (public LB at 05:44 IST).
