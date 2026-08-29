# Model Architecture Research: Knee MRI Multi-Label Pathology Classification

This document surveys all relevant architectures for classifying 12 knee pathologies (ACL, MCL, Meniscus tears, OA, Effusion, etc.) from MRI scans, grounded in published literature and current RSNA 2026 Kaggle competition insights.

---

## Ranked Architecture Summary

| Rank | Architecture | Type | Reported AUC | Complexity | Key Advantage |
|:----:|:-------------|:-----|:-------------|:-----------|:--------------|
| 🥇 1 | **DINOv2 + MIL/Attention Head** | Foundation Model | **0.91–0.92** (Kaggle LB) | Medium | Best out-of-box features; current competition SOTA |
| 🥈 2 | **2.5D CNN + BiLSTM/GRU** | Hybrid CNN-RNN | **0.85–0.92** | Medium-High | Proven in RSNA competitions; captures slice context |
| 🥉 3 | **CoPAS (Co-Plane Attention)** | Attention-based | **0.81** (Nature Comms) | High | Multi-sequence/multi-plane fusion; published on exactly this task |
| 4 | **ConvNeXt + Attention** | Modern CNN | **0.85–0.90** | Medium | Best pure-CNN backbone; data-efficient |
| 5 | **EfficientNetV2 + MIL** | Efficient CNN | **0.83–0.88** | Low-Medium | Excellent efficiency-accuracy trade-off |
| 6 | **3D ResNet / ResNet3D** | 3D CNN | **0.80–0.87** | High | True volumetric context; no aggregation needed |
| 7 | **Vision Transformer (ViT)** | Transformer | **0.85–0.90** | High | Global attention; great with large data |
| 8 | **BiomedCLIP** | Multimodal FM | **0.80–0.85** | Medium | Leverages radiology text + image jointly |
| 9 | **ResNet50 + Max-Pooling (MRNet-style)** | Classic CNN | **0.84–0.87** | Low | Simple baseline; well-documented |
| 10 | **EfficientNet3D** | 3D CNN | **0.82–0.86** | High | Balanced 3D scaling |

---

## Detailed Architecture Breakdown

---

### 🥇 Rank 1: DINOv2 + MIL/Attention Head

> [!TIP]
> **This is the current top-performing approach on the RSNA 2026 Kaggle leaderboard (public AUC ~0.91–0.915).**

**What it is:** DINOv2 is Meta's self-supervised Vision Transformer, pre-trained on 142M images. It produces extremely rich visual features without any labeled data. For MRI, you freeze (or lightly fine-tune) the DINOv2 backbone and attach a lightweight classification head.

**How it works for MRI:**
```
MRI Volume (N slices) 
  → Each slice → DINOv2 encoder → [CLS] token embedding (768-dim)
  → Stack N embeddings → Attention/MIL pooling
  → Multi-label classifier (12 outputs)
```

**Key papers & resources:**
- GitHub: `AIM-Harvard/DINOv2-3D-Med` — 3D medical adaptation
- GitHub: `MohammedSB/DINOv2ForRadiology` — radiology benchmarks
- Kaggle RSNA 2026 top notebooks using `dinov2-small` as frozen encoder

**Pros:**
- Strongest transfer features available (142M pre-training images)
- Works well with frozen backbone → minimal GPU memory
- Achieves SOTA with simple MIL heads
- Efficient enough for the competition's Efficiency Track

**Cons:**
- Inherently 2D → requires slice aggregation strategy
- Large model variants (ViT-L, ViT-G) need significant VRAM
- Not specifically trained on medical data

---

### 🥈 Rank 2: 2.5D CNN + BiLSTM/GRU (Sequential Aggregation)

> [!IMPORTANT]
> **This is the proven "workhorse" architecture from RSNA 2022 (Cervical Spine) and RSNA 2024 (Lumbar Spine) winning solutions.**

**What it is:** Process each MRI slice (or triplet of adjacent slices) through a 2D CNN backbone, then feed the sequence of slice embeddings into a bidirectional LSTM or GRU to capture inter-slice dependencies.

**How it works:**
```
MRI Volume → Slice triplets (2.5D)
  → 2D CNN backbone (EfficientNet/ConvNeXt) per triplet → Feature vectors
  → BiLSTM/GRU sequence → Hidden states
  → Attention-weighted pooling → Multi-label head (12 outputs)
```

**Common backbone choices:**
- EfficientNetV2-S (best efficiency)
- ConvNeXt-Tiny/Small (best accuracy)
- CoaT Lite (competition-tested)

**Key papers:**
- RSNA 2024 Lumbar Spine winning solutions (YOLOv8 + 2.5D classifiers)
- RSNA 2022 Cervical Spine (3D segmentation + 2.5D LSTM classifiers)

**Pros:**
- Battle-tested in medical imaging competitions
- Leverages powerful ImageNet-pretrained 2D backbones
- BiLSTM captures long-range slice dependencies
- Moderate GPU memory requirements

**Cons:**
- Two-stage pipeline (backbone + RNN) adds complexity
- Sequential processing is slower than attention-based pooling
- Requires careful slice ordering and padding

---

### 🥉 Rank 3: CoPAS (Co-Plane Attention across MRI Sequences)

> [!NOTE]
> **Published in Nature Communications (2024) — specifically designed for classifying 12 knee abnormalities from multi-sequence MRI.**

**What it is:** A purpose-built architecture that learns to fuse information across different MRI planes (sagittal, coronal, axial) and sequences (T1W, T2W, PDW) using co-plane attention modules.

**Architecture:**
```
Multi-sequence MRI (T1, T2, PD × Sagittal, Coronal, Axial)
  → Per-sequence CNN encoder → Feature maps
  → Co-Plane Attention Module → Cross-sequence feature fusion
  → Global classifier → 12 knee pathology scores
```

**Key paper:**
- *"Learning Co-plane Attention across MRI Sequences for Diagnosing Twelve Types of Knee Abnormalities"* — Nature Communications, 2024
- Dataset: 1,748 patients from 5 clinical centers in China
- **AUROC: 0.812**, Accuracy: 0.78 (competitive with senior radiologists at 0.80)

**Pros:**
- Specifically designed for exactly this clinical task (12 knee pathologies)
- Multi-sequence fusion mimics radiologist workflow
- Clinically validated (improved junior radiologist accuracy from 0.73 → 0.79)

**Cons:**
- Requires multiple MRI sequences as input (your dataset may only have limited sequences)
- Lower raw AUC than DINOv2-based approaches on Kaggle
- Complex to implement from scratch

---

### Rank 4: ConvNeXt + Attention Pooling

**What it is:** ConvNeXt is a "modernized ResNet" that borrows design choices from Vision Transformers (large kernels, LayerNorm, GELU) while remaining a pure CNN.

**Why it's strong:**
- Outperforms EfficientNet and ResNet50 in many medical imaging benchmarks
- Data-efficient: works well even with limited labeled samples
- ConvNeXt-Tiny has only 28M params (vs. ViT-B's 86M)

**Recommended variant:** `ConvNeXt-Tiny` or `ConvNeXt-Small` (ImageNet-22k pretrained)

---

### Rank 5: EfficientNetV2 + MIL

**What it is:** Google's compound-scaled CNN optimized for training speed and parameter efficiency.

**Why it's strong:**
- EfficientNetV2-S is the most common backbone in Kaggle medical competitions
- Excellent latency/accuracy trade-off for the Efficiency Track
- Extensive community support and pre-trained weights

**Recommended variant:** `EfficientNetV2-S` (ImageNet-21k pretrained)

---

### Rank 6: 3D ResNet / ResNet3D

**What it is:** Standard ResNet adapted for 3D convolutions. Processes the entire MRI volume as a 3D tensor.

**Key papers:**
- Multiple studies on ACL tear and cartilage defect classification using ResNet3D-18/34

**Pros:**
- True volumetric understanding; no aggregation tricks needed
- Clean, single-stage pipeline

**Cons:**
- Much higher GPU memory (3D convolutions are expensive)
- Limited pre-trained weights compared to 2D models
- Harder to train from scratch on small datasets

---

### Rank 7: Vision Transformer (ViT)

**What it is:** Pure transformer architecture applied to image patches. Each MRI slice is split into 16×16 patches, linearly projected, and processed with self-attention.

**Key papers:**
- OA grading with ViT: ~88–90% accuracy (KL grading)
- MR-Transformer: SOTA for predicting total knee replacement

**Pros:**
- Global receptive field from the start (vs. CNNs' local receptive fields)
- Scales well with data and compute

**Cons:**
- Requires large datasets or strong pre-training
- ViT from scratch on small medical datasets underperforms CNNs
- Higher compute requirements than CNNs

---

### Rank 8: BiomedCLIP

**What it is:** Microsoft's biomedical CLIP model (ViT image encoder + PubMedBERT text encoder), pre-trained on 15M figure-caption pairs from biomedical literature.

**Why it's interesting for this competition:**
- Can leverage both MRI images AND radiology report text
- Zero-shot and few-shot capabilities

**Cons:**
- Not specialized for MRI volumetric data
- Lower raw performance than DINOv2 for pure image classification
- Research-only license

---

### Rank 9: MRNet (ResNet + Max-Pooling)

**What it is:** The original Stanford baseline for knee MRI classification. Uses AlexNet/ResNet as a slice-level feature extractor, max-pools across slices, and classifies via a fully connected layer.

**Key paper:**
- Bien et al. (2018), *"Deep-learning-assisted diagnosis for knee MRI"*, PLoS Medicine
- Dataset: MRNet (1,370 exams from Stanford)
- AUC: 0.937 (abnormality), 0.965 (ACL), 0.847 (meniscus)

**Pros:**
- Extremely simple to implement
- Well-documented baseline
- Good starting point for ablation studies

**Cons:**
- Max-pooling loses inter-slice ordering information
- AlexNet backbone is outdated
- Single-sequence only

---

### Rank 10: EfficientNet3D

**What it is:** 3D adaptation of EfficientNet's compound scaling methodology.

**Pros:** Balanced depth/width/resolution scaling in 3D
**Cons:** Limited pre-trained weights; high memory; less community adoption than 2D variants

---

## Recommended Strategy for RSNA 2026

Based on the research, here's what I recommend:

### Primary Model (Start Here)
```
DINOv2-Small (frozen) → Slice embeddings → Attention Pooling → 12-class sigmoid head
```
- Quickest path to a competitive score (~0.91 AUC)
- Minimal GPU requirements (frozen backbone)
- Well-supported by community notebooks

### Secondary Model (For Ensembling)
```
EfficientNetV2-S (fine-tuned) → 2.5D triplets → BiGRU → Attention → 12-class head
```
- Complementary features to DINOv2
- Proven in past RSNA competitions
- Different inductive biases → diverse ensemble

### Advanced (If Time Permits)
```
ConvNeXt-Tiny (fine-tuned) → Multi-view fusion → MIL Attention → 12-class head
```

### Ensemble
- Average or learned stacking of the above 2–3 models
- Past RSNA winners consistently use multi-backbone ensembles

---

## Key Insights from Kaggle Discussions

> [!WARNING]
> **Scanner-Grouped Cross-Validation:** Community discussions highlight that metadata-only classifiers show significant gaps depending on scanner manufacturer (Siemens/Philips/GE) and field strength (1.5T/3T). Consider scanner-grouped CV instead of random K-fold splits.

> [!IMPORTANT]
> **Efficiency Track:** There's a separate $18,000 prize pool for efficient models. If targeting this, DINOv2-Small + lightweight MIL head is ideal.

---

## References

| # | Paper/Source | Year | Focus |
|---|:------------|:----:|:------|
| 1 | Bien et al., *Deep-learning-assisted diagnosis for knee MRI* (PLoS Medicine) | 2018 | MRNet baseline |
| 2 | CoPAS, *Learning Co-plane Attention across MRI Sequences* (Nature Communications) | 2024 | 12 knee pathologies |
| 3 | Oquab et al., *DINOv2: Learning Robust Visual Features* (Meta AI) | 2023 | Foundation model |
| 4 | Liu et al., *A ConvNet for the 2020s* (CVPR) | 2022 | ConvNeXt |
| 5 | Tan & Le, *EfficientNetV2: Smaller Models and Faster Training* (ICML) | 2021 | EfficientNetV2 |
| 6 | Dosovitskiy et al., *An Image is Worth 16x16 Words* (ICLR) | 2021 | Vision Transformer |
| 7 | Zhang et al., *BiomedCLIP* (Microsoft Research) | 2023 | Multimodal FM |
| 8 | `AIM-Harvard/DINOv2-3D-Med` (GitHub) | 2024 | 3D medical DINOv2 |
| 9 | RSNA 2024 Lumbar Spine winning solutions (Kaggle) | 2024 | 2.5D + LSTM |
| 10 | RSNA 2022 Cervical Spine winning solutions (Kaggle) | 2022 | 3D seg + 2.5D classif |
