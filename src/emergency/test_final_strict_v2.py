import json
import os
import sys

from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, ".")
import kg

p = os.path.abspath("notebooks/candidate_A/final_strict_v2")
err = kg.scan_folder(p)
print(f"kg.scan_folder result: {err}")
assert err is None, f"scan_folder failed: {err}"

# Check metadata
meta = json.loads(Path(p, "kernel-metadata.json").read_text(encoding="utf-8-sig"))
print(f"Kernel ID: {meta['id']}")
print(f"Datasets: {meta['dataset_sources']}")
assert "nishkarshsinghal/rsna-knee-s6ft-weights" in meta["dataset_sources"]
assert meta["is_private"] is True

# Check notebook compiles
nb = json.loads(Path(p, "rsna-knee-final-strict-v2.ipynb").read_text(encoding="utf-8-sig"))
print(f"Num cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if not src.lstrip().startswith("%%"):
            compile(src, f"cell_{idx}", "exec")

print("All final_strict_v2 tests passed 100%!")
