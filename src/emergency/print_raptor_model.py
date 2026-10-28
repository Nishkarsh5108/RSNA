import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_strict_v2/rsna-knee-final-strict-v2.ipynb").read_text(encoding="utf-8-sig"))
src = "".join(nb["cells"][2]["source"])
lines = src.splitlines()
for idx, l in enumerate(lines[180:260], 181):
    print(f"{idx}: {l}")
