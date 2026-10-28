import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_strict/rsna-knee-final-strict.ipynb").read_text(encoding="utf-8-sig"))

out = []
for idx in range(len(nb["cells"])):
    c = nb["cells"][idx]
    out.append(f"######################################## CELL {idx} ({c['cell_type']}) ########################################\n")
    out.append("".join(c["source"]))
    out.append("\n\n")

Path("src/emergency/final_strict_dump.txt").write_text("".join(out), encoding="utf-8")
print("Wrote final_strict_dump.txt")
