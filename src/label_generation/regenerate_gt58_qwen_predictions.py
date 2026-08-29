# 1. Install High-Speed vLLM Engine (skip if already installed in this session)
#!pip install -U vllm -q
#
# Regenerates genuine (non-leaked) Qwen predictions for the 58 ground-truth studies,
# using the exact same model configs / prompts / parsing logic as the two original
# generation scripts. Paste this as a single Kaggle notebook cell.
#
# Why subprocess isolation instead of loading both models in one process: vLLM (especially
# with tensor_parallel_size=2) does not reliably release GPU memory when you just `del` the
# LLM object -- this is a well-documented, still-unresolved pain point (see e.g.
# github.com/vllm-project/vllm/issues/5211 and #1908). Running each model in its own
# subprocess sidesteps this entirely: when a subprocess exits, its CUDA context is torn
# down by the OS, guaranteeing full GPU memory release before the next model loads --
# no manual cleanup needed, and no risk of the second model OOMing on leftover memory.
#
# Output: /kaggle/working/gt58_regenerated_qwen_predictions.csv
# Columns: StudyInstanceUID, {pathology}_true (the real ground truth, for a quick sanity
# check), {pathology}_14b (fresh Qwen2.5-14B-Instruct-AWQ predictions), {pathology}_7b
# (fresh Qwen2.5-7B-Instruct predictions).
#
# What to do with it afterwards: open train_folds_qwen14b_awq_v2.csv and
# train_folds_sota_qwen25.csv, and overwrite ONLY the 58 GT rows' 12 label columns with
# the matching _14b / _7b columns from this output (leave Report, fold, and the other
# 4,349 rows untouched). Once those 58 rows are genuine predictions instead of copied
# ground truth, consensus_label_aggregation.py's leakage auto-detection will pick this up
# automatically on the next run -- no script changes needed there.

import sys
import subprocess
import time

import pandas as pd
import numpy as np
from IPython.display import FileLink, display
from sklearn.metrics import roc_auc_score, accuracy_score

WORKER_PATH = "/kaggle/working/_gt58_infer_worker.py"
OUT_14B = "/kaggle/working/_gt58_preds_14b.csv"
OUT_7B = "/kaggle/working/_gt58_preds_7b.csv"
FINAL_OUT = "/kaggle/working/gt58_regenerated_qwen_predictions.csv"

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# -------------------------------------------------------------
# Worker script -- self-contained, takes --variant 14b|7b, generates on the 58 GT
# studies only, and writes a small CSV. Each variant's model/prompt/parsing logic is
# copied verbatim from the script that produced the corresponding full dataset, so the
# reliability estimate this unlocks is representative of that exact generation process.
# -------------------------------------------------------------
WORKER_SOURCE = r'''
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SYSTEM_PROMPT_14B = """You are an expert musculoskeletal radiologist.
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

SYSTEM_PROMPT_7B = """You are an expert musculoskeletal radiologist.
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

VARIANTS = {
    "14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "quantization": "awq",
        "max_tokens": 220,
        "system_prompt": SYSTEM_PROMPT_14B,
    },
    "7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "quantization": None,
        "max_tokens": 150,
        "system_prompt": SYSTEM_PROMPT_7B,
    },
}


def create_prompt(system_prompt, report_text):
    clean_text = str(report_text)[:1200]
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\nKnee MRI Radiology Report:\n\"\"\"\n{clean_text}\n\"\"\"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["14b", "7b"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = VARIANTS[args.variant]

    train_df = pd.read_csv("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    gt_df = train_df[train_df["ACL"].notnull()].reset_index(drop=True)
    print(f"[{args.variant}] Regenerating predictions for {len(gt_df)} ground-truth studies...")

    llm_kwargs = dict(
        model=cfg["model_id"],
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enforce_eager=True,
        trust_remote_code=True,
    )
    if cfg["quantization"]:
        llm_kwargs["quantization"] = cfg["quantization"]

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=cfg["max_tokens"], stop=["<|im_end|>"])

    prompts = [create_prompt(cfg["system_prompt"], r) for r in gt_df["Report"]]
    outputs = llm.generate(prompts, sampling_params)

    predictions = []
    fallback_count = 0
    for out in outputs:
        text = out.outputs[0].text
        match = re.search(r"\{[\s\S]*\}", text)
        try:
            predictions.append(json.loads(match.group(0)))
        except Exception:
            fallback_count += 1
            predictions.append({col: 0.5 for col in TARGET_COLS})

    print(f"[{args.variant}] JSON parse failures: {fallback_count}/{len(outputs)}")

    pred_df = pd.DataFrame(predictions)
    pred_df["StudyInstanceUID"] = gt_df["StudyInstanceUID"].values
    pred_df.to_csv(args.out, index=False)
    print(f"[{args.variant}] Saved: {args.out}")


if __name__ == "__main__":
    main()
'''

with open(WORKER_PATH, "w", encoding="utf-8") as f:
    f.write(WORKER_SOURCE)

# -------------------------------------------------------------
# Run each model in its own subprocess, one at a time (subprocess.run blocks until
# the process exits, so the first model's GPU memory is guaranteed fully released
# before the second one starts loading).
# -------------------------------------------------------------
for variant, out_path in [("14b", OUT_14B), ("7b", OUT_7B)]:
    print(f"\n{'='*100}\n Running {variant} in a subprocess\n{'='*100}")
    subprocess.run(
        [sys.executable, WORKER_PATH, "--variant", variant, "--out", out_path],
        check=True,
    )
    time.sleep(5)  # small safety margin for the driver to fully release GPU memory

# -------------------------------------------------------------
# Merge results with ground truth, print a quick sanity-check benchmark, save + download
# -------------------------------------------------------------
train_df = pd.read_csv("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
gt_df = train_df[train_df["ACL"].notnull()].reset_index(drop=True)

preds_14b = pd.read_csv(OUT_14B).set_index("StudyInstanceUID")
preds_7b = pd.read_csv(OUT_7B).set_index("StudyInstanceUID")

final_df = gt_df[["StudyInstanceUID"] + TARGET_COLS].set_index("StudyInstanceUID")
final_df.columns = [f"{c}_true" for c in TARGET_COLS]

for col in TARGET_COLS:
    final_df[f"{col}_14b"] = preds_14b.loc[final_df.index, col]
    final_df[f"{col}_7b"] = preds_7b.loc[final_df.index, col]

final_df = final_df.reset_index()
final_df.to_csv(FINAL_OUT, index=False)

print("\n" + "=" * 100)
print(" QUICK SANITY-CHECK BENCHMARK (regenerated 58-row predictions vs ground truth)")
print("=" * 100)
print(f"{'PATHOLOGY':<20} | {'14B ACC':<8} | {'14B AUC':<8} | {'7B ACC':<8} | {'7B AUC':<8}")
print("=" * 100)
for col in TARGET_COLS:
    y_true = final_df[f"{col}_true"].astype(float).values
    y_14b = final_df[f"{col}_14b"].astype(float).values
    y_7b = final_df[f"{col}_7b"].astype(float).values
    has_both_classes = len(np.unique(y_true)) > 1

    acc_14b = accuracy_score(y_true, (y_14b >= 0.5).astype(int))
    acc_7b = accuracy_score(y_true, (y_7b >= 0.5).astype(int))
    auc_14b = roc_auc_score(y_true, y_14b) if has_both_classes else np.nan
    auc_7b = roc_auc_score(y_true, y_7b) if has_both_classes else np.nan

    auc_14b_str = f"{auc_14b:.4f}" if not np.isnan(auc_14b) else "N/A"
    auc_7b_str = f"{auc_7b:.4f}" if not np.isnan(auc_7b) else "N/A"
    print(f"{col:<20} | {acc_14b*100:6.2f}% | {auc_14b_str:<8} | {acc_7b*100:6.2f}% | {auc_7b_str:<8}")

print(f"\nSaved: {FINAL_OUT}")
print("\nClick below to download:")
display(FileLink(FINAL_OUT.replace("/kaggle/working/", "")))
