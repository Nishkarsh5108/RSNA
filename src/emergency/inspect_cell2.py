import json
from pathlib import Path

nb_path = Path("notebooks/candidate_A/final_strict/rsna-knee-final-strict.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8-sig"))

src = "".join(nb["cells"][2]["source"])
lines = [l for l in src.splitlines() if "coatnet" in l.lower() or "checkpoint" in l.lower() or "models" in l.lower() or "weights" in l.lower()]
print("Matching lines in cell 2:")
for l in lines[:30]:
    print(l)
