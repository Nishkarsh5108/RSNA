# 1. Install High-Speed vLLM Engine
#!pip install -U vllm -q

import os
import re
import json
import numpy as np
import pandas as pd
from IPython.display import FileLink, display
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from vllm import LLM, SamplingParams

# -------------------------------------------------------------
# 2. LOAD DATASET
# -------------------------------------------------------------
train_csv_path = "/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"
print(f"Loading dataset: {train_csv_path}")
train_df = pd.read_csv(train_csv_path)
print(f"Total records loaded: {len(train_df)}")

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# -------------------------------------------------------------
# 3. MULTILINGUAL RADIOLOGY PROMPT (Spanish / Dutch / Greek)
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist.
Analyze the knee MRI report (written in Spanish, Dutch, Greek, or English) and evaluate the presence of 12 knee pathologies.

Language mapping guide:
- Spanish: 'rotura de menisco' = Meniscus tear, 'ligamento cruzado anterior (LCA)' = ACL, 'ligamento colateral medial (LCM)' = MCL, 'derrame articular' = Effusion, 'artrosis/condropatia' = OA, 'quiste popliteo/Baker' = Baker's cyst, 'edema oseo/contusion' = Contusion, 'fractura' = Fracture.
- Dutch: 'meniscusscheur' = Meniscus tear, 'voorste kruisband (VKB)' = ACL, 'binnenband/MCL' = MCL, 'gewrichtsvocht' = Effusion, 'kraakbeenslijtage/artrose' = OA, 'botkneuzing' = Contusion.
- Greek: 'ρήξη μηνίσκου' = Meniscus tear, 'πρόσθιος χιαστός σύνδεσμος (ΠΧΣ)' = ACL, 'έσω πλάγιος σύνδεσμος (ΕΠΣ)' = MCL, 'άρθρωση/υγρό' or 'ενδαρθρική συλλογή' = Effusion, 'οστεοαρθρίτιδα/χόνδρινη βλάβη' = OA, 'κύστη Baker/ιγνυακή κύστη' = Baker's cyst, 'οστική θλάση/οίδημα μυελού' = Contusion, 'κάταγμα' = Fracture.

Output strictly a valid JSON object mapping each pathology to a float confidence score between 0.0 (definitely absent/normal/intact) and 1.0 (definitely present/torn/pathologic):
{
  "ACL": float,
  "MCL": float,
  "Medial Meniscus": float,
  "Lateral Meniscus": float,
  "Medial OA": float,
  "Lateral OA": float,
  "PF OA": float,
  "Effusion": float,
  "Synovitis": float,
  "Baker's": float,
  "Contusion": float,
  "Fracture": float
}
Output ONLY the JSON object. No explanations or extra text."""

def create_prompt(report_text):
    clean_text = str(report_text)[:1200]
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nKnee MRI Radiology Report:\n\"\"\"\n{clean_text}\n\"\"\"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

# -------------------------------------------------------------
# 4. INITIALIZE VLLM ENGINE (AWQ-quantized 14B, 4096 Context Length)
# -------------------------------------------------------------
# NOTE: plain FP16 Qwen2.5-14B-Instruct OOMs on 2x T4 (~28GB weights alone,
# no room left for KV cache). The official AWQ checkpoint below is ~9GB,
# which is what actually makes the 14B model feasible on this hardware.
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-AWQ"
print(f"Initializing vLLM on 2x T4 GPUs (AWQ 4-bit, 4096 context)...")

llm = LLM(
    model=MODEL_ID,
    quantization="awq",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    max_model_len=4096,       # 4096 token context
    enforce_eager=True,       # Bypasses compiler latency
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=220,
    stop=["<|im_end|>"]
)

def parse_predictions(outputs, label):
    """Parse vLLM outputs into a list of {col: float} dicts, tracking fallback hits."""
    predictions = []
    fallback_count = 0
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        if i < 3:
            print(f"[{label}] Sample raw output {i}: {text[:300]!r}")
        match = re.search(r"\{[\s\S]*\}", text)
        try:
            predictions.append(json.loads(match.group(0)))
        except Exception:
            fallback_count += 1
            predictions.append({col: 0.5 for col in TARGET_COLS})

    fallback_rate = fallback_count / len(outputs) if outputs else 0.0
    print(f"[{label}] JSON parse failures: {fallback_count}/{len(outputs)} ({fallback_rate*100:.1f}%)")
    if fallback_rate > 0.05:
        print(f"[{label}] WARNING: fallback rate exceeds 5% — check sample outputs above before trusting these metrics.")
    return predictions

# -------------------------------------------------------------
# 5. STEP 1: BENCHMARK ON 58 GROUND-TRUTH STUDIES
# -------------------------------------------------------------
gt_df = train_df[train_df["ACL"].notnull()].reset_index(drop=True)
print(f"\n[Step 1] Benchmarking {len(gt_df)} Ground-Truth studies...")

gt_prompts = [create_prompt(r) for r in gt_df["Report"]]
gt_outputs = llm.generate(gt_prompts, sampling_params)
gt_predictions = parse_predictions(gt_outputs, "GT")

gt_pred_df = pd.DataFrame(gt_predictions)

print("\n" + "=" * 100)
print(" GROUND-TRUTH VALIDATION BENCHMARK RESULTS")
print("=" * 100)
print(f"{'PATHOLOGY':<20} | {'ACC':<8} | {'AUC':<8} | {'PREC':<8} | {'REC':<8} | {'F1':<8}")
print("=" * 100)
accs, aucs, precs, recs, f1s = [], [], [], [], []

for col in TARGET_COLS:
    y_true = gt_df[col].astype(float).values
    y_prob = gt_pred_df[col].astype(float).values
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    accs.append(acc)
    precs.append(prec)
    recs.append(rec)
    f1s.append(f1)
    if not np.isnan(auc):
        aucs.append(auc)

    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    print(f"{col:<20} | {acc*100:6.2f}% | {auc_str:<8} | {prec:6.4f} | {rec:6.4f} | {f1:6.4f}")

print("=" * 100)
print(f" • Macro Accuracy  : {np.mean(accs)*100:.2f}%")
print(f" • Macro ROC-AUC   : {np.mean(aucs):.4f}")
print(f" • Macro F1-Score  : {np.mean(f1s):.4f}")
print("=" * 100)

# -------------------------------------------------------------
# 6. STEP 2: GENERATE ALL ~4,349 UNLABELED REPORTS
# -------------------------------------------------------------
unlabeled_df = train_df[train_df["ACL"].isnull()].reset_index(drop=True)
print(f"\n[Step 2] Processing {len(unlabeled_df)} studies via vLLM PagedAttention...")

unlabeled_prompts = [create_prompt(r) for r in unlabeled_df["Report"]]
unlabeled_outputs = llm.generate(unlabeled_prompts, sampling_params)
unlabeled_predictions = parse_predictions(unlabeled_outputs, "UNLABELED")

all_results = []
for idx, parsed in enumerate(unlabeled_predictions):
    uid = unlabeled_df.iloc[idx]["StudyInstanceUID"]
    entry = dict(parsed)
    entry["StudyInstanceUID"] = uid
    all_results.append(entry)

pseudo_df = pd.DataFrame(all_results)

# -------------------------------------------------------------
# 7. STEP 3: MERGE & SAVE MASTER CSV WITH 5-FOLD CV
# -------------------------------------------------------------
final_df = train_df.copy().set_index("StudyInstanceUID")
pseudo_df = pseudo_df.set_index("StudyInstanceUID")

for col in TARGET_COLS:
    final_df[col] = final_df[col].combine_first(pseudo_df[col])

final_df = final_df.reset_index()

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
strat_target = (final_df[TARGET_COLS] >= 0.5).sum(axis=1)

final_df["fold"] = -1
for fold, (t_idx, v_idx) in enumerate(sgkf.split(final_df, strat_target, groups=final_df["StudyInstanceUID"])):
    final_df.loc[v_idx, "fold"] = fold

out_path = "/kaggle/working/train_folds_qwen14b_awq_v2.csv"
final_df.to_csv(out_path, index=False)
print(f"\nMaster training dataset saved successfully to: {out_path}")

print("\nClick below to download your file immediately:")
display(FileLink("train_folds_qwen14b_awq_v2.csv"))
