import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_strict/rsna-knee-final-strict.ipynb").read_text(encoding="utf-8-sig"))
src = "".join(nb["cells"][2]["source"])
print("Checking for 224 in raptor script:")
for l in src.splitlines():
    if "224" in l or "ARMS" in l or "arms" in l:
        print(l)
