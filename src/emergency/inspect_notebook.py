import json
from pathlib import Path

nb_path = Path("notebooks/candidate_A/final_strict/rsna-knee-final-strict.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8-sig"))

for idx in [4, 5, 7, 8]:
    cell = nb["cells"][idx]
    src = "".join(cell.get("source", []))
    print(f"=== CELL {idx} ===")
    print(src[:600])
    print(f"... ({len(src)} total chars)\n")
