# 1. Install High-Speed vLLM Engine
#!pip install vllm -q

import os
import re
import json
import numpy as np
import pandas as pd
from IPython.display import FileLink, display
from sklearn.metrics import roc_auc_score, accuracy_score
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
# 3. MULTILINGUAL RADIOLOGY PROMPT
# -------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert musculoskeletal radiologist.
Analyze the knee MRI report (written in Spanish, Dutch, or English) and evaluate the presence of 12 knee pathologies.

Language mapping guide:
- Spanish: 'rotura de menisco' = Meniscus tear, 'ligamento cruzado anterior (LCA)' = ACL, 'ligamento colateral medial (LCM)' = MCL, 'derrame articular' = Effusion, 'artrosis/condropatia' = OA, 'quiste popliteo/Baker' = Baker's cyst, 'edema oseo/contusion' = Contusion, 'fractura' = Fracture.
- Dutch: 'meniscusscheur' = Meniscus tear, 'voorste kruisband (VKB)' = ACL, 'binnenband/MCL' = MCL, 'gewrichtsvocht' = Effusion, 'kraakbeenslijtage/artrose' = OA, 'botkneuzing' = Contusion.

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
# 4. INITIALIZE VLLM ENGINE (4096 Context Length)
# -------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
print(f"Initializing vLLM on 2x T4 GPUs (4096 context)...")

llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.85,
    max_model_len=4096,       # 4096 token context
    enforce_eager=True,       # Bypasses compiler latency
    trust_remote_code=True
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=150,
    stop=["<|im_end|>"]
)

# -------------------------------------------------------------
# 5. STEP 1: BENCHMARK ON 58 GROUND-TRUTH STUDIES
# -------------------------------------------------------------
gt_df = train_df[train_df["ACL"].notnull()].reset_index(drop=True)
print(f"\n[Step 1] Benchmarking {len(gt_df)} Ground-Truth studies...")

gt_prompts = [create_prompt(r) for r in gt_df["Report"]]
gt_outputs = llm.generate(gt_prompts, sampling_params)

gt_predictions = []
for out in gt_outputs:
    text = out.outputs[0].text
    match = re.search(r"\{[\s\S]*\}", text)
    try:
        gt_predictions.append(json.loads(match.group(0)))
    except:
        gt_predictions.append({col: 0.5 for col in TARGET_COLS})

gt_pred_df = pd.DataFrame(gt_predictions)

print("\n" + "="*80)
print(" GROUND-TRUTH VALIDATION BENCHMARK RESULTS")
print("="*80)
aucs, accs = [], []
for col in TARGET_COLS:
    y_true = gt_df[col].astype(float).values
    y_prob = gt_pred_df[col].astype(float).values
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    accs.append(acc)
    if not np.isnan(auc):
        aucs.append(auc)
    auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
    print(f"{col:<20} | Accuracy: {acc*100:6.2f}% | ROC-AUC: {auc_str}")

print("="*80)
print(f" • Macro-Averaged Accuracy : {np.mean(accs)*100:.2f}%")
print(f" • Macro-Averaged ROC-AUC  : {np.mean(aucs):.4f}")
print("="*80)

# -------------------------------------------------------------
# 6. STEP 2: GENERATE ALL 4,349 UNLABELED REPORTS (~8-10 mins)
# -------------------------------------------------------------
unlabeled_df = train_df[train_df["ACL"].isnull()].reset_index(drop=True)
print(f"\n[Step 2] Processing {len(unlabeled_df)} studies via vLLM PagedAttention...")

unlabeled_prompts = [create_prompt(r) for r in unlabeled_df["Report"]]
unlabeled_outputs = llm.generate(unlabeled_prompts, sampling_params)

all_results = []
for idx, out in enumerate(unlabeled_outputs):
    uid = unlabeled_df.iloc[idx]["StudyInstanceUID"]
    text = out.outputs[0].text
    match = re.search(r"\{[\s\S]*\}", text)
    try:
        parsed = json.loads(match.group(0))
        parsed["StudyInstanceUID"] = uid
        all_results.append(parsed)
    except:
        entry = {col: 0.5 for col in TARGET_COLS}
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

out_path = "/kaggle/working/train_folds_sota_qwen25.csv"
final_df.to_csv(out_path, index=False)
print(f"\n Master training dataset saved successfully to: {out_path}")

print("\nClick below to download your file immediately:")
display(FileLink("train_folds_sota_qwen25.csv"))
