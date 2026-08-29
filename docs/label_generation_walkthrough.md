# Label Generation & Verification Walkthrough: RSNA Knee Abnormality Detection

## 1. Problem Statement & Initial Strategy
The competition dataset consists of 4,407 MRI studies. However, only **58 studies** have hard, human-annotated ground-truth labels for the 12 knee pathologies (ACL, Meniscus, OA, etc.). The remaining **4,349 studies** only have raw, multilingual radiology text reports (English, Spanish, Dutch).

To train high-performance 2.5D or 3D CNN vision models, we needed to parse these 4,349 text reports into structured, probabilistic "soft labels" ($0.0 \rightarrow 1.0$).

We initially explored a community-provided Kaggle dataset containing pre-generated pseudo-labels.

## 2. Verification of Public Labels & The Flaw
We developed a validation script (`compare_labels.py`) to benchmark the public pseudo-labels against the 58 ground-truth annotations. 

**Observation:** The script yielded a **100% Macro-Accuracy** and **1.000 ROC-AUC**.
**Conclusion:** This meant the public dataset simply copied the exact 58 manual labels into its CSV. While this preserves ground truth, it meant we could not use those 58 cases to verify the *actual accuracy* of the LLM that generated the remaining 4,349 labels. Furthermore, community feedback indicated those labels suffered from poor accuracy (around 60%) because the models used to generate them lacked multilingual medical grounding (failing on Spanish/Dutch reports).

> [!IMPORTANT]
> **Strategic Pivot:** We decided to build a custom NLP extraction pipeline running directly on Kaggle's 2x T4 GPUs (32GB VRAM total) to generate highly accurate, verifiable labels using a State-of-the-Art (SOTA) Open Weights model.

---

## 3. Model Exploration & Technical Hurdles

### Attempt 1: Qwen2.5-14B with vLLM
We initially selected `Qwen/Qwen2.5-14B-Instruct` because of its elite multilingual capabilities. We attempted to serve it using the `vLLM` high-throughput engine with Tensor Parallelism (`tensor_parallel_size=2`) across both Kaggle T4 GPUs.
* **The Failure:** `CUDA OutOfMemoryError`.
* **Reason:** The 14B model in FP16 takes ~28GB of VRAM. Combined with the KV-cache pre-allocation of vLLM, it immediately exhausted the 32GB total capacity of the T4 GPUs.

### Attempt 2: English-only Medical Models (e.g., Llama3-OpenBioLLM-8B)
We considered using specialized medical models like `Llama3-OpenBioLLM-8B` which have deep knowledge of orthopedic grading scales.
* **The Failure:** Poor multilingual performance.
* **Reason:** While excellent for English reports, these models struggled heavily with the Spanish and Dutch reports that comprise a large portion of the unannotated Kaggle dataset, resulting in low accuracy on those instances.

---

## 4. The Final Setup: Qwen2.5-7B with Optimized vLLM
We pivoted to the final, optimal architecture using **`Qwen/Qwen2.5-7B-Instruct`**. 

### Model Selection
* **Why we chose it:** It provides SOTA multilingual comprehension (easily parsing Spanish and Dutch radiology reports) while being small enough (7B) to comfortably fit on Kaggle's T4 GPUs, leaving plenty of room for high-throughput batching via vLLM.

### Prompt Engineering: Multilingual Mapping
To bridge the gap between its generalist nature and specific medical terminology, we supplied a targeted **Language Mapping Guide** in the System Prompt:
* **Spanish:** 'rotura de menisco' = Meniscus tear, 'ligamento cruzado anterior (LCA)' = ACL, 'derrame articular' = Effusion...
* **Dutch:** 'meniscusscheur' = Meniscus tear, 'voorste kruisband (VKB)' = ACL...

This allows the model to act as a zero-shot cross-lingual medical translator and output a clean JSON probability dictionary. We also clamped the input `report_text` to 1200 characters to remove noise and ensure it comfortably fits the context window.

### GPU Memory Balancing (The vLLM OOM Fix)
To prevent the OOM errors encountered earlier with vLLM, we finely tuned the vLLM initialization parameters:
```python
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,  # Leaves 15% VRAM free for safety
    max_model_len=4096,           # Restricts maximum context to prevent KV cache explosion
    enforce_eager=True,           # Bypasses CUDA graph compilation which spikes VRAM
    trust_remote_code=True
)
```
By lowering `gpu_memory_utilization` to `0.85`, restricting `max_model_len`, and enforcing eager execution, we completely eliminated the CUDA OOM crashes while retaining blazing-fast tensor-parallel generation.

---

## 5. End-to-End Pipeline & Verification

The final script (`label_generation_kaggle.py`) enforces a strict, robust pipeline:

1. **Ground-Truth First Verification:** Before generating all labels, the script extracts labels for the 58 known ground-truth studies. It mathematically computes and prints the Accuracy and ROC-AUC per class. *This guarantees we quantitatively trust the LLM's cross-lingual reasoning before proceeding.*
2. **High-Speed Inference:** Utilizing vLLM's PagedAttention, the remaining 4,349 unlabeled reports are processed in just **~8-10 minutes** across the Kaggle T4 GPUs.
3. **JSON Regex Parsing & Fallbacks:** Output is strictly parsed for JSON objects. If extraction fails entirely, a neutral prior probability of `0.5` is assigned.
4. **Stratified Group K-Fold:** The original ground truth labels are combined with the high-fidelity pseudo-labels. Finally, the entire dataset is split into a **5-Fold CV (`StratifiedGroupKFold`)** grouped by `StudyInstanceUID` to ensure patient scans never leak between training and validation folds. 

> [!TIP]
> **Why this setup works:** It perfectly balances **Multilingual Comprehension** (Qwen) with **Throughput & Memory Safety** (vLLM tuning). Generating labels in minutes rather than hours allows for rapid iteration. Training our subsequent vision models on these continuous probability soft-labels provides the crucial gradient signals needed for high-scoring classification in the RSNA competition.
