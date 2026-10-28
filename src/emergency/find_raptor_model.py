import json
from pathlib import Path

nb = json.loads(Path("notebooks/candidate_A/final_strict_v2/rsna-knee-final-strict-v2.ipynb").read_text(encoding="utf-8-sig"))
src = "".join(nb["cells"][2]["source"])

lines = src.splitlines()
for idx, l in enumerate(lines):
    if any(k in l for k in ["class ", "def load_model", "def build_model", "nn.Module", "forward"]):
        print(f"Line {idx}: {l[:80]}")
