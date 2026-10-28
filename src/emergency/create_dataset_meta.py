import json
import os

meta = {
    "title": "rsna-knee-s6ft-weights",
    "id": "nishkarshsinghal/rsna-knee-s6ft-weights",
    "licenses": [{"name": "CC0-1.0"}]
}
dest = r"D:\rsna_work\kaggle_upload\rsna-knee-s6ft-weights\dataset-metadata.json"
with open(dest, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
print("Wrote", dest)
