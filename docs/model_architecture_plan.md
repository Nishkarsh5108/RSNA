# Implementation Plan: DINOv2 + Region-Guided Attention (RGA)

## Goal
Build a **DINOv2-Small (frozen) + Region-Guided Attention** model that classifies 12 knee pathologies from multi-sequence MRI scans. The "region-specific" enhancement groups pathologies by anatomical compartment and uses dedicated attention heads per region, so the model learns to focus on the *right part* of the knee for each condition.

## Dataset Structure Understanding

```
D:\Coding\kaggle\RSNA\data\
├── main_data/
│   ├── train_folds_sota_qwen25.csv   # Labels: StudyInstanceUID → 12 targets + fold
│   ├── train_series.csv              # SeriesInstanceUID → StudyUID, Plane, Fluid_Sensitive
│   ├── test.csv                      # Test study IDs
│   └── sample_submission.csv         # Submission format
├── <SeriesInstanceUID>.npz           # ~24K preprocessed MRI volumes (flat directory)
└── medical_llm_pseudo_labels.csv
```

Each `.npz` contains a preprocessed 3D array for one MRI series. Each study has multiple series (Sagittal, Axial, Coronal; fluid-sensitive & non-fluid-sensitive).

---

## Architecture Design

### The Core Idea: Region-Guided Attention (RGA)

Instead of a single global attention head for all 12 pathologies, we **group targets by anatomical region** and assign dedicated attention mechanisms to each group. This forces the model to learn *where* to look for each condition.

```
┌─────────────────────────────────────────────────────────────────┐
│                    STUDY-LEVEL PIPELINE                         │
│                                                                 │
│  Series 1 (Sagittal)  ──┐                                      │
│  Series 2 (Axial)     ──┼─→  Per-slice DINOv2-S (frozen)       │
│  Series 3 (Coronal)   ──┘    → [CLS] embeddings (384-dim)      │
│                                                                 │
│         ↓ Stack all slice embeddings across series              │
│                                                                 │
│  ┌──────────────────────────────────────────────┐               │
│  │       SHARED TRANSFORMER ENCODER             │               │
│  │  (2-layer, learnable positional encoding)    │               │
│  │  + Plane embedding (Sag/Ax/Cor)              │               │
│  │  + Sequence embedding (Fluid/Non-Fluid)      │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓                                      │
│  ┌──────────────────────────────────────────────┐               │
│  │      REGION-GUIDED ATTENTION HEADS           │               │
│  │                                              │               │
│  │  🦴 Ligament Head → ACL, MCL                 │               │
│  │     (Learns to attend to central knee)       │               │
│  │                                              │               │
│  │  🔵 Meniscus Head → Med. Meniscus, Lat. M.   │               │
│  │     (Learns to attend to medial/lateral)     │               │
│  │                                              │               │
│  │  🟠 OA Head → Medial OA, Lateral OA, PF OA  │               │
│  │     (Learns to attend to joint spaces)       │               │
│  │                                              │               │
│  │  💧 Fluid Head → Effusion, Synovitis, Baker's│               │
│  │     (Learns to attend to fluid areas)        │               │
│  │                                              │               │
│  │  💥 Trauma Head → Contusion, Fracture        │               │
│  │     (Learns to attend to bone structures)    │               │
│  └──────────────────────────────────────────────┘               │
│                          ↓                                      │
│           12 sigmoid outputs (confidence scores)                │
└─────────────────────────────────────────────────────────────────┘
```

### Pathology → Region Grouping

| Region Group | Pathologies | Anatomical Focus |
|:-------------|:------------|:-----------------|
| **Ligaments** | ACL, MCL | Intercondylar notch, medial compartment |
| **Meniscus** | Medial Meniscus, Lateral Meniscus | Medial/lateral joint line |
| **Osteoarthritis** | Medial OA, Lateral OA, PF OA | Cartilage surfaces, joint spaces |
| **Fluid/Inflammation** | Effusion, Synovitis, Baker's | Suprapatellar recess, posterior |
| **Trauma** | Contusion, Fracture | Bone marrow signal |

---

## Proposed File Structure

```
D:\Coding\kaggle\RSNA\src\
├── model/
│   ├── __init__.py
│   ├── config.py              # All hyperparameters and paths
│   ├── dataset.py             # KneeMRIDataset (loads .npz, groups series per study)
│   ├── dino_backbone.py       # DINOv2-Small frozen feature extractor
│   ├── region_attention.py    # Region-Guided Attention heads
│   ├── knee_model.py          # Full model assembly
│   └── train.py               # Training loop with K-fold CV
├── EDA/
│   └── eda_qwen_labels.ipynb  # (Already created)
└── label_generation/
    └── label_generation_kaggle.py  # (Already created)
```

---

## Proposed Changes

### Model Components

---

#### [NEW] `config.py`
Central configuration file containing:
- Data paths (local + Kaggle)
- DINOv2 model ID (`facebook/dinov2-small`)
- Region-to-pathology mapping
- Training hyperparameters (lr, epochs, batch_size)
- GPU/memory settings for 2× T4

---

#### [NEW] `dataset.py`
`KneeMRIDataset` class that:
1. Groups `.npz` files by `StudyInstanceUID` using `train_series.csv`
2. Loads and stacks slices from all series for a study
3. Uniformly samples N slices per series (to control memory; e.g., 16 slices per series)
4. Returns: `(stacked_slices, plane_ids, sequence_ids, labels)`
5. Handles augmentation (random horizontal flip, intensity jitter)

---

#### [NEW] `dino_backbone.py`
Wraps `facebook/dinov2-small` (ViT-S/14):
- 384-dim output embeddings
- Completely frozen (no gradients)
- Processes slices in batches to avoid OOM
- Input: grayscale MRI slice → replicate to 3-channel → resize 224×224 → DINOv2 → `[CLS]` token

---

#### [NEW] `region_attention.py`
Implements the Region-Guided Attention mechanism:
- **Shared Transformer Encoder** (2 layers, 384-dim, 6 heads): Processes all slice embeddings with learned plane + sequence positional embeddings
- **Per-Region Attention Pooling**: Each region group has its own query vector(s) that attend over the encoded slice features via cross-attention
- **Per-Region Classification Head**: Small MLP (384 → 128 → N_targets) per region group

---

#### [NEW] `knee_model.py`
Full model assembly:
- DINOv2 backbone → Shared encoder → Region attention heads → 12 sigmoid outputs
- Forward pass orchestration
- Memory-efficient inference (process slices in chunks through DINOv2)

---

#### [NEW] `train.py`
Training script for Kaggle:
- 5-fold cross-validation using existing `fold` column
- BCE loss with soft labels (continuous confidence scores from Qwen)
- AdamW optimizer, cosine annealing LR scheduler
- Mixed precision (fp16) training
- Gradient accumulation (effective batch size = 16, actual = 2-4)
- Macro-AUC ROC as validation metric
- Model checkpointing (best macro-AUC)
- Designed for 2× T4 GPUs via DataParallel

---

## Memory Budget (2× T4 = 2 × 15.5 GB)

| Component | VRAM Estimate |
|:----------|:-------------|
| DINOv2-Small (frozen, fp16) | ~160 MB |
| Shared Transformer (2 layers) | ~15 MB |
| Region Attention Heads (5 groups) | ~5 MB |
| Input batch (4 studies × ~48 slices × 224² × 3) | ~800 MB |
| DINOv2 activations (inference mode) | ~2 GB |
| Training activations + gradients | ~3 GB |
| **Total per GPU** | **~6 GB** ✅ |

> [!TIP]
> With DINOv2 frozen, we only backprop through the lightweight attention heads and shared encoder (~20M trainable params). This fits very comfortably on T4s.

---

## Training Strategy

1. **Phase 1 — Frozen DINOv2 + Train heads** (15-20 epochs, ~2 hours on 2× T4)
   - Freeze DINOv2 entirely
   - Train shared encoder + region attention heads
   - LR: 1e-3 with cosine annealing

2. **Phase 2 — Light fine-tuning** (optional, 5-10 epochs, ~4 hours)
   - Unfreeze last 2 blocks of DINOv2
   - Use discriminative LR: backbone 1e-5, heads 1e-4
   - Monitor for overfitting

---

## Verification Plan

### Automated Tests
- Validate model forward pass with dummy input before training
- Assert output shape = `(batch_size, 12)` with values in `[0, 1]`
- Monitor macro-AUC on validation fold during training

### Manual Verification
- Compare fold-0 validation AUC against baseline (Qwen soft labels = ~0.85)
- Visualize attention heatmaps to confirm region heads focus on correct anatomy
- Submit predictions to Kaggle leaderboard after training completes

---

## Open Questions

> [!NOTE]
> **NPZ File Format (resolved 2026-08-30)**: confirmed by direct inspection. Each `.npz` holds one key, `data` — a `uint8` array of shape `(24, 224, 224)`: 24 slices, pre-resized to 224×224, already 8-bit windowed/normalized. Filenames are keyed by `SeriesInstanceUID` (one file per series, not per study) — 24,371 files for 24,371 series across 4,407 studies. Series-per-study ranges 3–14 (mode 5); series-per-(study, plane) ranges 1–7 (mode 2) — "one series per plane" is not a safe assumption, so the data loader needs to pool over a variable number of series within each plane.

> [!IMPORTANT]
> **Kaggle vs Local**: Since you'll be training on Kaggle, I'll write the code with configurable paths (local for debugging, Kaggle for training). The `.npz` data — is it also available on Kaggle as a dataset, or will you upload it?
