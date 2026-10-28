# Competition solutions and medical-imaging research (2026-10-18)

Scope: only what changes an experiment decision in the last 4.5 days.

Evidence grades:
- **[V]** verified here, from official pages or code/data I ran;
- **[S]** self-reported by a competitor, not reproducible here;
- **[L]** peer-reviewed literature.

Sources were read on 2026-10-18 through a reader proxy (Kaggle pages) and the GitHub API. Kaggle datasets and notebooks could not be downloaded: there is no Kaggle token on this machine.

## 1. This competition: public solutions and reports

| Source | Architecture and input | Labels | Validation | Ensembling | Inference cost | Reported score | Grade |
|---|---|---|---|---|---|---|---|
| goodpjw2008 notebook "CoAtNet + R6 + 2.5D MIL" | nartaa CoAtNet (OAI-trained) + "R6" + 2.5D ConvNeXt reader + "Compact R7 CNX" | public weights | — | multi-arm blend | committed run 8 m 15 s on 2×T4 (3-study public test stub, **not** the scored runtime) | **0.958** public LB | [V] score shown on the code page; contents [S] |
| heliosli notebooks ("best public dataset 0.9586", "blend gold") | blends of public checkpoints (S6 weights, nartaa) | public | — | blend | — | 0.956–0.958 | [V] score; contents [S] |
| nartaa, discussion 746792 | **One CoAtNet, 2.5D attention MIL**: 5 slots (sag FS/other, cor FS/other, axial) resampled to 96 slices; **140 mm field on a 384 px grid; centre 320 px crop at inference**; 12 windows per study in training, **94 windows** (every slice) at inference; anatomical-mirror TTA; SWA of 3 epochs; 16 epochs; backbone LR 4e-5, head LR 2e-3 | own report soft labels + public tables (stevenleehans, pilkwang, JEV) | gold-58 (reused for selection) | single model | turnaround 7.2 min (efficiency model) / 26.1 min (accuracy model), queue included | competition data only: 0.939 → 0.940 → 0.942; with **2,399 OAI knees**: 0.945–0.949; efficiency model (307 px crop → 224) 0.945 | [S] |
| TranBaDat2607 repo (docs/report.md, phase-2/3 notes) | rebuilt the public "Raptor" CoAtNet-RMLP-2 @ 384 MIL (corpus recipe: span 15–85%, per-series 2–98 pct, 140 mm, area resize); ConvNeXt-nano 336 own models | public label tables (teach4 0.897 gold agreement); OOF mixes R1–R4 | gold-58 as regression guard; pre-registered rules | clean pipeline (4 arms) 0.939; clean + reader + own fine-tune 0.945; OAI public notebook 0.954 | clean pipeline ≈ 2.5–3 h on 2×T4 | own ConvNeXt-nano solo 0.917–0.924; **OOF pseudo-label mixing did not move their LB** (0.920 vs 0.922 report-only) | [S], unusually well documented |
| zoliseress repo | 2.5D MIL, **EfficientNet-B0 at 320 px**, fixed 150 mm physical crop, 24 windows per slot, masked mean + max pooling | Qwen (vLLM) report labels; not-mentioned → 0.07 | own radiologist-read reference set of 450 training studies (hand labels; not available to others) | single model | — | **0.942** public LB, single model | [S] |
| nathirad repo | ConvNeXt-T 224 (own); public 2.5D reader retrained | report labels; "calibrated" P(expert positive \| report state) fitted on gold-58 | gold-58 + OOF, pre-registered | public CoAtNet + reader 0.30 | — | own 0.882 gold; blends 0.944–0.949 | [S] |
| Discussion 735304 "Best single-model score" (69 comments) | many: ConvNeXt-T 5-fold 0.959; ResNet-50 224 5-fold 0.954; CoAtNet 224 single fold 0.950; DINOv2 ViT-S 0.951–0.952; Qwen3.5-2B VLM fine-tune 0.950 | consistent theme: **labels and pseudo-labels** ("model is not the key, it is your labels") | — | — | the efficiency-LB leader reports a small ResNet, 0.947 at 5 folds | 0.93–0.959 | [S] |
| Discussion 733932 (stevenleehans) | — | LLM labels with an explicit "not addressed" option: 25% of cells undecided, 84% for synovitis. Filling undecided synovitis cells from effusion: synovitis label gold AUC 0.678 → 0.790 | pre-registered on gold | — | — | label macro 0.878 → 0.887 | [S]. **Did not transfer to our labels** (H-SYN2: 0.756 → 0.738) [V] |
| Leaderboard (2026-10-18 09:12 UTC) | — | — | — | — | — | #1 0.969; gold ≥ 0.964; silver 0.962–0.963; owner 0.927 (rank ≈ 3,747) | [V] |

**Compliance note (corrected 2026-10-18 23:45 IST after reading the full thread 743416).**
- The morning read of discussion 743416 was incomplete. The host (Chris Carr, RSNA, about 2026-10-14) declined to rule dataset by dataset, restated rule 2.6, and gave a test:
  - "Datasets that require only registration and acknowledgement of licensing terms would clearly meet these criteria."
  - "Datasets that require an institutional research agreement or limit access to users with institutional affiliations clearly do not."
  - Teams are asked to disclose the datasets they used.
- KneeCoT is explicitly prohibited (institutional agreement). Non-commercial terms alone do not exclude a dataset.
- OAI is distributed by the NIMH Data Archive. Participants report individual approval at once via Login.gov plus click-through terms, with "Institutional Approval Required: No" (third-party). It therefore appears to meet the test.
- Still open (no host answer): access is blocked from some countries (e.g. China), and rule 2.6.b names geo restrictions. Whether that also applies to public weights trained on OAI is likewise unanswered.
- Residual risk for OAI-derived weights: low to moderate, not zero.

**Update 2026-10-19 01:30 IST (public-leg provenance; the owner approved OAI-trained public weights).**
- The 0.956–0.958 public notebooks use three external datasets, not one. Sources: a participant's audit of the datasets' manifests (thread 749858) [S], TranBaDat's notes [S], and nartaa's dataset page [V].
  - nartaa CoAtNets: OAI. The dataset terms allow use in this competition and CC BY-NC licensing of a winning submission [V].
  - heliosli R7: fastMRI images with fastMRI+ boxes as an auxiliary task.
  - `anatomy.pt`, which places the crops for heliosli's R6, R7 and S6: SKM-TEA.
- R6 proper (an older version of the `fe055406` bundle) has no fastMRI. goodpjw2008's 2.5D reader uses no external data.
- fastMRI access is a request stating affiliation and intended use (participant, thread 750259). Of the three, it is closest to the host's exclusion of affiliation-limited datasets.
- Host, 2026-09-21 (thread 741819, quoted in 749145): OAI does not violate the rules "if it is generally accessible to all participants without an escalated (i.e. institutional) review and sign-off or access to a legal team, and is free of charge".
- No newer ruling (checked 01:25 IST). Open threads ask about each leg and about what will be checked at final selection.
- **Checked in the code (02:40 IST, Kaggle API, read-only).** The 0.958 notebook (goodpjw2008 v27) loads R6, not R7.
  - R6's fastMRI status cannot be verified: its training run is named `cnx_r6fm` and shares R7's trainer, but its manifest omits the config.
  - S6 (run `cnx_s6`, built before R6 and R7) shows no fastMRI marker. nartaa + S6 scores 0.954.

**Update 2026-10-19 04:20 IST (primary sources; details in plan §10c).**
- fastMRI's own access page [V]: "The application must include the investigator's institutional affiliation and the proposed uses of the data." That matches the host's exclusion of datasets that "limit access to users with institutional affiliations".
- R6's trainer [V, source embedded in the 0.958 notebook, hash-matched to R6's manifest] is version 9.8, written on 2026-10-11 to add a fastMRI+ lesion-box auxiliary task whose head is stripped from the exported weights. R6 was built with it 14 h later from run `cnx_r6fm`. R6 was therefore very likely trained with fastMRI+ (inferred; its author has not answered a public question about it).
- S6's trainer [V, hash-matched] predates that option and has no external-data hook besides the SKM-TEA anchor model. S6 is fastMRI-free.
- heliosli's `rsna-knee-blend-gold-dataset-restored` (nartaa + S6, no R6/R7) displays 0.954 [V]. No fastMRI-free public notebook above 0.954 was found among the 40 highest-scoring ones [V, metadata].

## 2. What the evidence says, by question

| Question | Evidence | Weight |
|---|---|---|
| Does in-plane resolution matter? | Public strong models use 0.365–0.47 mm/px. nartaa: 336 → 384 native cache + more windows +0.001; 320 → 288 crop −0.002. zoliseress: EffNet-B0 at 320 px reached 0.942 single. Counterpoint: several 0.95 single models run at 224 px. Ours: E8 (0.6 mm/px) −0.002 on pseudo-val. | Medium: worth one clean test (**HR**) |
| More windows at inference? | nartaa: 94 → 24 windows −0.006 LB. Ours: K = 8 → 16 at inference +0.000 for a K = 8-trained model. | Medium: test with a few-window-trained HR model (**ALLW**) |
| Labels / pseudo-labels? | Most 0.95+ single models credit labels. TranBaDat measured no LB gain from OOF mixing with weaker models. Ours: the image model beats its labels on gold for 7/12 findings. | High expected value, high variance: **PL-g** judged on the LB |
| Backbone? | TranBaDat: swapping backbones on the same labels and views gave +0.001. Ours: EfficientNetV2-S −0.015, transformer aggregation +0.002. | Low |
| External data (OAI)? | +0.005–0.009 for nartaa | Not used: rules risk, no access |
| Ensembling? | Pipeline or label diversity helps (TranBaDat A05: different-teacher head +0.007 on gold); seeds of one pipeline correlate at about 0.95. Ours: seeds +0.003. | Medium: an HR or PL model is the diverse member |
| Synovitis? | Report blind spot (all report labels 0.68–0.79 on gold). Effusion proxy fails for our labeller [V]. | No cheap fix found |

## 3. Literature (only what bears on a decision)

| Topic | Key work | Relevance here |
|---|---|---|
| Knee MRI classification | Bien et al., *PLoS Med* 2018 (MRNet): per-plane 2D CNN over slices with max pooling; ACL/meniscus/abnormal | Established the 2.5D slice-feature + pooling design used by nearly every solution here [L] |
| Attention-based multiple-instance learning | Ilse, Tomczak & Welling, *ICML* 2018 | Our attn12 head. Measured +0.008 over mean pooling (E2a) [L][V] |
| 2.5D vs 3D | 2.5D with ImageNet-pretrained 2D encoders dominates when slice spacing (3–4 mm) is much coarser than in-plane resolution (0.3–0.6 mm). 3D needs isotropic data and pretraining that is not available offline at this scale. | 3D not pursued: 6 GB GPU, anisotropic 3.5 mm slices, no compatible pretrained 3D weights in 4 days |
| Backbones | ConvNeXt (Liu et al., *CVPR* 2022); CoAtNet (Dai et al., *NeurIPS* 2021); EfficientNet (Tan & Le, *ICML* 2019) | ConvNeXt-T is our strongest tested; CoAtNet-RMLP-2 is the public workhorse but too heavy to train on 6 GB [L] |
| Medical / self-supervised encoders | RadImageNet (Mei et al., *Radiol AI* 2022); DINOv2 (Oquab et al., 2023) | Public arms show them weaker than CoAtNet here (gold 0.84–0.85 [S]). Not pursued. |
| Noisy labels, self-training | Noisy Student (Xie et al., *CVPR* 2020); distillation (Hinton et al., 2015) | Basis of **PL**: out-of-fold teachers, soft targets, student retrained on blended targets [L] |
| Positive–unlabeled learning | Elkan & Noto, *KDD* 2008 | Relevant to report silence (unreported positives). Our soft teacher already gives silent cells non-zero mass. A full PU objective is not justified within the deadline. |
| Semi-quantitative knee MRI grading | MOAKS (Hunter et al., *Osteoarthritis Cartilage* 2011): effusion-synovitis and Hoffa-synovitis on non-contrast MRI | Motivated H-SYN1/2; both negative here [L][V] |
| Weight averaging | SWA (Izmailov et al., *UAI* 2018); our EMA | Already use EMA 0.999; nartaa's 3-epoch SWA is equivalent in spirit |

## 4. Shortlist (feasible before 2026-10-22, ranked by expected gain per GPU-hour)

| # | Approach | Expected effect | Cost | Limitation | Status |
|---|---|---|---|---|---|
| 1 | **HR input** (0.365 mm/px, 320 crop) + all-window inference | +0.000–0.010 | about 90 min per fold-0 run; about 2.5 h per all-fold model | 6 GB limits batch and backbone | E9 queued (starts after O1) |
| 2 | **PL-g pseudo-labels** from 5-fold OOF | +0.000–0.010 LB | 4 OOF runs (about 4.5 h) + students | Judged only on the LB; one careful team saw no gain | O1 running |
| 3 | Seeds or members of the best recipe | +0.003 | 85–150 min per model | Diminishing returns | after 1–2 |
| 4 | Efficiency variant: single model, few windows | about −40 to −70% runtime, small AUC loss | benchmark only | Kaggle runtime unknown until submitted | after E9 |
| 5 | Public-weight blend (owner's decision) | to about 0.945 (clean) / about 0.958 (OAI) | none locally; LB submissions | OAI rules risk; cannot be tested offline without a Kaggle token | prepared as an option |

### 3D model feasibility (estimate, not run)

**Input.** One 3D series per plane at 32 × 256 × 256 voxels is far from isotropic: about 3.5 mm slices vs 0.6 mm pixels.

**Memory.** A 3D ResNet-18 at batch 2, fp16, would need roughly 4–6 GB of activations for one plane. Three planes do not fit in 6 GB without gradient checkpointing and batch 1.

**Pretraining.** The only common 3D medical pretraining is MedicalNet (Chen et al., 2019), segmentation-pretrained on CT/MRI. Its weights would have to be bundled offline. There is no evidence here that it beats ImageNet-pretrained 2.5D encoders on 3.5 mm-spaced knee MRI.

**Time.** About 2–3 h per fold-0 pilot locally, plus a new cache.

**Verdict.** No pilot can produce decision-grade evidence and a submittable model before 2026-10-22. 2.5D with three-slice windows already covers the through-plane resolution available (3.5 mm).

Not pursued, with reasons:
- **3D CNNs:** see the estimate above.
- **DINOv2 / RadImageNet:** shown weaker here [S].
- **VLM fine-tuning:** needs hours of A100-class compute.
- **New LLM labels:** label agreement has plateaued at about 0.90 on gold across public tables; the image side is the bigger gap.
