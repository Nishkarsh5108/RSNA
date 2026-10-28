import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb").read_text(encoding="utf-8-sig"))
src9 = "".join(nb["cells"][9]["source"])
print("Lines 60-90 of cell 9 in final_max:")
for idx, l in enumerate(src9.splitlines()[60:90], 61):
    print(f"{idx}: {l}")
