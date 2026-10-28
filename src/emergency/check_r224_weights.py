import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb").read_text(encoding="utf-8-sig"))
src6 = "".join(nb["cells"][6]["source"])
print("Raptor 224 weights candidates in Cell 6:")
for line in src6.splitlines()[:50]:
    if "candidates" in line or "weights" in line or "weight_name" in line or "pt" in line:
        print("  ", line)
