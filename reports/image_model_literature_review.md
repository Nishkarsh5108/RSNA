# Image-model literature review: RSNA Knee Abnormality Detection

Date: 2026-10-17. Each statement is tagged as one of:
- **[V]** verified: I read the primary source or code, or measured it locally;
- **[A]** author or participant claim, not reproduced;
- **[H]** my hypothesis.

Leaderboard (LB) scores from different validation schemes are not comparable to each other or to gold-58 numbers.

## 1. Public solutions for this exact competition

| Source | Input / series | Slices and windows | Orientation / laterality | Aggregation | Backbone and weights | Labels | Validation | Reported score | What it teaches |
|---|---|---|---|---|---|---|---|---|---|
| goodpjw2008 "2.5D ConvNeXt reader" (Kaggle dataset; Apache-2.0); code reused in aditbytes `src/knee.py` [V, code] | 6 slots = 3 planes × {FS, non-FS}; FS first; largest series first; random candidate in training | 8 windows per series; each window = slices [c−1, c, c+1] as 3 channels; centres in [4%, 96%], random per bin (train) or linspace (eval) | Canonical reorientation in `preprocess.py` (not inspected). **Train-time left–right mirroring of coronal/axial at p = 0.5 without label swapping** | Slot + slice-position embeddings → 2-layer transformer (d = 384, 8 heads) → **per-finding attention pooling** (12 queries) → per-finding linear | `convnext_tiny` (timm, ImageNet), drop-path 0.1; 256 px; EMA weights | Report-derived soft labels; gold-58 excluded | 5-fold (3 folds released) | 0.929 public LB (3 folds); 0.944 in a 30% rank blend [A] | A strong, simple recipe: FS-first slots, 3-slice windows, per-finding attention |
| aditbytes reader (GitHub, 2026-10-06 snapshot) [A] | Up to 6 series | 8 three-slice windows per series; ≤ 32 slices cached | Canonical reorientation; **0.6 mm in-plane resampling, 153.6 mm crop** | Small transformer + per-finding attention | EfficientNetV2-S | LLM soft labels (≈0.89 vs gold) | 4 folds; gold-58 never trained on | v1 224 px, 8 epochs: gold 0.838 / LB 0.860. v3 256 px, 12 epochs, LR 1.5e-4 + warm-up: gold ≈0.90, LB 0.923. v2 diverged with a higher LR. | **Resolution and LR schedule matter**; gold tracks LB at this level |
| Y. Bishnoi, HF blog (2026-08-09) [A] | ≤ 3 series, 1 per plane, FS first | T = 12 slices, 288 px | No horizontal flips | MIL: per-series pooling, mean over series; **max pooling beat gated attention** | EfficientNet-B3 (timm), single channel | LLM labels + public set | GroupKFold-5; leak-corrected gold 0.857 | LB 0.903 (rank 18/792, early) | Simple pooling can win; leakage inflated gold by 0.08 |
| TranBaDat2607 (GitHub, 2026-10) [A] | — | 96-slice native-384 input (320 crop) for the best clean model | Anatomical-mirror augmentation (presumably with label swap) | Attention MIL | CoAtNet ("Raptor" v5/v8/v10, public checkpoints), ConvNeXt-nano 336 px, DINOv2 baseline | LLM soft labels, OOF pseudo-labels | Gold-58 as a regression guard (SE ≈ 0.02) | Clean single model 0.921 gold; clean ensemble 0.930 gold / 0.939 LB; **0.954 LB only with OAI-trained public weights** (rank about 1,387 / 5,980 on 10-14) | Top of LB > 0.954. Public CoAtNet-384 checkpoints underpin most top stacks. External-data (OAI) status unclear. |
| dreaddevelopment "Raptor" CoAtNet-384 (Kaggle) [A] | Pre-decoded slice stacks | — | — | 2.5D attention MIL | CoAtNet-384 | Report-distilled soft labels | — | 0.917 macro AUC on gold-58 | Public strong single model (384 px) |
| mmatmath two-stage (GitHub) [V, README] | 6 slots, 24 slices, 224 px, slot masks | 2.5D windows | **No laterality correction** (stated) | Target-specific attention MIL → small DeBERTa over frozen embeddings | ConvNeXt-base DINOv3 weights | Qwen3-4B 1/0/null labels | Round-robin folds by study | none | Same label model as ours; acknowledges missing laterality fix |

Patterns across public work [H, consistent with the above]:
1. Fine-tuned 2.5D slice encoders (ConvNeXt, EfficientNetV2, CoAtNet) with MIL pooling over slices and series dominate.
2. Higher resolution (256–384) beats 224.
3. FS-first series selection is standard.
4. Per-finding attention lets each target attend to different planes and slices.
5. Ensembles of 3–5 models add about 0.01–0.02 LB.
6. No public solution we inspected handles **sagittal slice-order laterality** (see `image_pipeline_audit.md`). Some apply random left–right mirroring, which *destroys* medial/lateral information unless labels are swapped (MCL has no lateral counterpart among the targets).

## 2. Knee-MRI literature

| Paper | Design | Data and labels | Reported results | Transfer to this task |
|---|---|---|---|---|
| Bien et al., *MRNet*, PLOS Med 2018 [V] | Per-series 2D CNN over all slices (AlexNet features, max-pooled over slices); logistic regression combines the 3 series | 1,370 exams (Stanford); report-derived labels: abnormal / ACL / meniscus | Internal AUC 0.937 / 0.965 / 0.847; external ACL 0.824 zero-shot, 0.911 after retraining | The slice-encoder + pooling MIL template works with study-level labels. Its cross-site drop (0.965 → 0.824) warns about site shift, which matters here (16–19 sites). |
| Tsai et al., *ELNet*, MIDL 2020 [V, abstract] | Lightweight CNN from scratch, single stack | MRNet data | "Compares favourably" with MRNet using one stack | Small data can support compact models; pretraining matters less there. Our data (4.3k studies, noisy labels) favours pretrained encoders [H]. |
| Ilse et al., *Attention-based Deep MIL*, ICML 2018 [V] | Permutation-invariant attention / gated-attention pooling over instances | MIL benchmarks, histopathology | Attention ≥ max/mean on several benchmarks | Fits the setting: study-level labels, findings visible in a few slices. Compared empirically below (mean / max / gated / per-finding). |
| Mei et al., *RadImageNet*, Radiol AI 2022 [V] | ImageNet-style pretraining on 1.35 M radiologic images (CT/MR/US) | — | Better than ImageNet on 8 small medical tasks | CC BY 4.0, but weights exist only for ResNet50 / DenseNet121 / Inception (no ConvNeXt). Gains are largest on *small* datasets. Not tested here (time). [H: small or no gain over IN-22k ConvNeXt at this data size] |
| Oquab et al., *DINOv2*, 2023 (Apache-2.0) [V] | Self-supervised ViT foundation features | LVD-142M natural images | Strong frozen features | The user's v4 frozen DINOv2-small reached 0.755 gold (selection-biased) vs about 0.73–0.76 for frozen EfficientNet-B0. **Frozen** foundation features did not beat a fine-tuned CNN in any public result found. ViT patch 14 at 224 is costly for many slices. |
| fastMRI / fastMRI+ (Zbontar 2018; Zhao 2022) [V] | Raw k-space + reconstructions; fastMRI+ adds bounding boxes and study labels | NYU knees; data-use agreement | — | External data requires an agreement. Eligibility under the "reasonably accessible" rule is unclear, and there's no time to integrate it. Not used. |
| ConvNeXt (Liu et al., CVPR 2022); EfficientNetV2 (Tan & Le, ICML 2021); CoAtNet (Dai et al., NeurIPS 2021) [V] | Modern CNN / hybrid backbones | ImageNet-1k/22k | — | All available in timm with permissive licences. Measured local throughput at 224 px: ConvNeXt-T 232 img/s, EffNetV2-S 218, CoAtNet-0 178. |

## 3. 2D vs 2.5D vs 3D for this dataset

| Approach | Spatial context | Variable slice counts | Data efficiency (4.3k studies) | Compute (6 GB GPU) | Verdict |
|---|---|---|---|---|---|
| 2D slices + MIL | In-plane only | Natural (bag of slices) | High (ImageNet pretraining) | Cheap | Baseline |
| **2.5D (3 adjacent slices as channels) + MIL** | Local through-plane (about ±3.5 mm here) | Natural | High: still uses 2D pretrained weights | Same as 2D | **Chosen**. Used by all strong public readers. |
| 2.5D + transformer over windows | Plus long-range across slices and planes | Needs position embeddings and masks | Medium (more parameters) | Small overhead | Tested as an ablation (E2c) |
| 3D CNN (e.g. MedicalNet / R3D) | Full volume | Needs resampling to a fixed grid | Low: few pretrained 3D weights; anisotropic 2D sequences (3–4 mm slices) | 3D activations do not fit 6 GB at useful resolution | Not pursued; cost was estimated, not measured [H] |
| Foundation encoders (frozen) | Per slice | Natural | High for frozen | Medium | Already shown inferior frozen (v4) |

## 4. Methods relevant to training

| Technique | Evidence | Decision here |
|---|---|---|
| Soft-label BCE on report-derived teachers | Label study: soft ≥ hard downstream (0.770 vs 0.766, frozen features) [V]; CheXpert U-policies [V] | Soft BCE default; hard labels as ablation (E5) |
| Noisy labels: EMA / mean-teacher weights | Public reader ships EMA weights [V] | EMA 0.999 used for selection and inference |
| Class imbalance: pos_weight / focal | AUC is rank-based. pos_weight changes gradient balance, not the metric. | Plain BCE default (`--pos-weight-clip` available, untested) |
| Gold-58 as validation | n = 58, SE ≈ 0.02–0.025 [V]; gold-based selection inflates results (previous v1–v4) [V] | **Gold sealed**. Selection on the pseudo-labelled val fold; gold scored once at the end. |
| TTA (flip / shift) | Mirroring is anatomically wrong without label swaps [H] | No flip TTA. Window-shift TTA (2 bags of 8 windows, offset by half a step) measured on E1: 0.8836 vs 0.8824 [V]. Re-tested on the final configuration before adoption (see experiments report). |
| Domain generalisation (16–19 sites, vendor × 25 intensity range) | MRNet external drop [V]; EDA [V] | Per-series percentile normalisation (cache); gamma/gain/affine augmentation per series |
| Ensembling | +0.01–0.02 LB in public stacks [A] | Prediction averaging of complementary runs (E7) |

## 5. Ranked approaches to test (decided before running)

1. **Fine-tuned 2.5D ConvNeXt-T, 3 FS-first planes, 8 windows, per-finding attention** (E1) vs frozen baseline (E0): the expected big step.
2. Laterality normalisation on vs off (E1 vs E1o): validates the pipeline fix on medial/lateral targets.
3. Aggregation: mean / max / per-finding attention / transformer (E2).
4. Planes: sagittal / coronal / axial alone vs all (E3).
5. Series: 3 slots vs 6 (FS + non-FS) (E4).
6. Labels: soft vs hard (E5).
7. Backbone: EfficientNetV2-S vs ConvNeXt-T (E6).
8. Ensemble of the best complementary runs (E7).
9. Not tested for lack of time: higher resolution (256–384), which needs a new cache; CoAtNet; RadImageNet; 3D CNNs.

**Added after E1 (2026-10-17, before their results were known).** Item 9's resolution test became feasible once the v6 cache was built in the background:
- **E8**: 0.6 mm/px aspect-preserving resampling with a 256 px crop (the aditbytes v3 recipe) vs the 224 px squash.
- **E1L**: a 12-epoch cosine schedule (E1 was still improving when the learning rate reached zero).
- **E5w**: confidence-weighted soft labels (|2y−1| weights), from the brief's training-target list.
- **E1s2**: a seed replicate, for seed-to-seed variability.
- **Window-shift TTA**: tested at inference only (`eval_tta.py`).

## Sources

- Bien N. et al. MRNet. PLOS Med 2018 — <https://pmc.ncbi.nlm.nih.gov/articles/PMC6258509/>
- Tsai C.-H. et al. ELNet. MIDL 2020 — <https://proceedings.mlr.press/v121/tsai20a.html>
- Ilse M. et al. Attention-based Deep MIL. ICML 2018 — <https://arxiv.org/abs/1802.04712>
- Mei X. et al. RadImageNet. Radiol AI 2022 — <https://pmc.ncbi.nlm.nih.gov/articles/PMC9530758/>
- Oquab M. et al. DINOv2. 2023 — <https://arxiv.org/abs/2304.07193>
- Liu Z. et al. ConvNeXt. CVPR 2022 — <https://arxiv.org/abs/2201.03545>; Tan M., Le Q. EfficientNetV2. ICML 2021 — <https://arxiv.org/abs/2104.00298>; Dai Z. et al. CoAtNet. NeurIPS 2021 — <https://arxiv.org/abs/2106.04803>
- Zbontar J. et al. fastMRI 2018 — <https://arxiv.org/abs/1811.08839>; Zhao R. et al. fastMRI+ 2022 — <https://doi.org/10.1038/s41597-022-01255-z>
- goodpjw2008 reader — <https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader>; aditbytes repo (code `src/knee.py`) — <https://github.com/aditbytes/kaggle-rsna-knee-abnormality>
- Bishnoi HF blog — <https://huggingface.co/blog/bishnoiyash/rsna-competetion>
- TranBaDat2607 repo — <https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection>
- mmatmath repo — <https://github.com/mmatmath/rsna-knee-two-stage>
- Raptor fine-spacing corpus (dreaddevelopment) — <https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-finespacing>
