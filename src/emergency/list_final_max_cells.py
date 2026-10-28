import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb").read_text(encoding="utf-8-sig"))

for idx in range(len(nb["cells"])):
    c = nb["cells"][idx]
    src = "".join(c["source"])
    first = src.splitlines()[0] if src.splitlines() else ""
    print(f"Cell {idx} ({c['cell_type']}): {first[:70]} | length: {len(src)}")
