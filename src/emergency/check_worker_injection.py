import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb").read_text(encoding="utf-8-sig"))

for idx, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "compact_infer_x" in src:
        print(f"Cell {idx} mentions compact_infer_x:")
        for line in src.splitlines():
            if "compact_infer_x" in line or "SOURCES" in line or "WORKER" in line:
                print("  ", line)
