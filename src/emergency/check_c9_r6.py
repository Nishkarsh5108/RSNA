import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb").read_text(encoding="utf-8-sig"))
src9 = "".join(nb["cells"][9]["source"])
for l in src9.splitlines():
    if "c_slug" in l.lower() or "bundle_id" in l.lower() or "manifest" in l.lower():
        print("  ", l)
