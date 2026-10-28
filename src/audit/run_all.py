"""Re-run the audit / EDA / label pipeline in dependency order (local paths as in configs/label_pipeline.yaml).

    python src/audit/run_all.py            # everything except the GPU LLM passes
    python src/audit/run_all.py --with-llm # also the two LLM passes (~1.5-2.5 h on an RTX 4050)
"""
import argparse
import subprocess
import sys

PY = sys.executable
R = "D:/Coding/kaggle/RSNA"
STEPS = [
    ("DICOM header inventory", [f"{R}/src/audit/dicom_inventory.py", "--root", f"{R}/data/Raw_dicom/train_series", "--out", f"{R}/reports/eda/dicom", "--workers", "8", "--slice-sample", "2000"]),
    ("report profile (language/structure)", [f"{R}/src/audit/report_profile.py"]),
    ("metadata EDA + study inventory", [f"{R}/src/audit/dicom_eda.py"]),
    ("pixel EDA (sample) + orientation", [f"{R}/src/audit/image_eda.py"]),
    ("laterality table", [f"{R}/src/audit/laterality_table.py"]),
    ("existing-label audit", [f"{R}/src/audit/audit_existing_labels.py"]),
    ("rules extractor", [f"{R}/src/labeling/run_rules.py"]),
    ("report/label EDA", [f"{R}/src/audit/report_label_eda.py"]),
]
LLM = [
    ("LLM pass (canonical order)", [f"{R}/src/labeling/llm_assertion_labeler.py", "--out", f"{R}/data/labels_v3/llm_qwen3_4b_assert_v3.1_raw.csv"]),
    ("LLM pass (reversed order)", [f"{R}/src/labeling/llm_assertion_labeler.py", "--reverse-items", "--out", f"{R}/data/labels_v3/llm_qwen3_4b_assert_v3.1_reversed_raw.csv"]),
]
POST = [
    ("gold validation (locked variants)", [f"{R}/src/labeling/validate_labels.py"]),
    ("order-averaging validation", [f"{R}/src/labeling/validate_order_avg.py"]),
    ("final labels + provenance", [f"{R}/src/labeling/build_final_labels.py"]),
    ("downstream label sets", [f"{R}/src/labeling/make_label_sets_v3.py"]),
]
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true")
    a = ap.parse_args()
    for name, cmd in STEPS + (LLM if a.with_llm else []) + POST:
        print(f"==> {name}", flush=True)
        r = subprocess.run([PY] + cmd)
        if r.returncode != 0:
            sys.exit(f"step failed: {name}")
    print("done. Downstream: python src/labeling/downstream_eval.py --label-sets <json> --device cuda")
