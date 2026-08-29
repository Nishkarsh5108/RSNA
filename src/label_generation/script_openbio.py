import os
import re
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Paths
CSV_PATH = r"D:\Coding\kaggle\RSNA\data\main_data\train (1).csv"
OUTPUT_PATH = r"D:\Coding\kaggle\RSNA\data\medical_llm_pseudo_labels.csv"

# 2. 4-bit Quantization Configuration (Crucial for 6GB VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

MODEL_ID = "aaditya/Llama3-OpenBioLLM-8B"
print(f"Loading {MODEL_ID} in 4-bit on RTX 4050...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
    dtype=torch.float16
)

# 3. Medical System Prompt
SYSTEM_PROMPT = """You are a board-certified musculoskeletal radiologist.
Your task is to analyze the knee MRI report (which may be in English, Spanish, or Dutch) and identify the presence of 12 knee pathologies.

Output ONLY a valid JSON dictionary mapping each pathology to a probability score between 0.0 (definitely absent/normal) and 1.0 (definitely present/torn/pathologic):
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
Do not write explanations, markdown fences, or introduction. Return ONLY the JSON string."""

# 4. Extraction Helper with explicit Llama-3 prompt formatting
def parse_report(report_text):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Knee MRI Radiology Report:\n\"\"\"\n{report_text}\n\"\"\"<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.01,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Extract JSON with regex
    match = re.search(r"\{[\s\S]*\}", output_text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            return {k: float(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None

# 5. Process Unlabeled Studies
df = pd.read_csv(CSV_PATH)
unlabeled_df = df[df["ACL"].isnull()].reset_index(drop=True)
print(f"Total reports to extract: {len(unlabeled_df)}")

results = []
# Check if resume file exists
if os.path.exists(OUTPUT_PATH):
    try:
        saved_df = pd.read_csv(OUTPUT_PATH)
        results = saved_df.to_dict("records")
        processed_uids = set(saved_df["StudyInstanceUID"].dropna().unique())
        unlabeled_df = unlabeled_df[~unlabeled_df["StudyInstanceUID"].isin(processed_uids)].reset_index(drop=True)
        print(f"Resuming: {len(results)} already completed, {len(unlabeled_df)} reports remaining.")
    except Exception as e:
        print(f"Could not load previous results ({e}), starting fresh.")

# Batch loop
for idx, row in tqdm(unlabeled_df.iterrows(), total=len(unlabeled_df)):
    uid = row["StudyInstanceUID"]
    report = str(row["Report"])
    
    scores = parse_report(report)
    if scores:
        scores["StudyInstanceUID"] = uid
        results.append(scores)
    else:
        results.append({"StudyInstanceUID": uid})
        
    # Periodic auto-save every 25 records
    if (idx + 1) % 25 == 0:
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

# Final save
pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
print(f"Extraction complete! Saved {len(results)} records to: {OUTPUT_PATH}")
