import json

nb = json.load(open('notebooks/candidate_A/final_max/_v2/rsna-knee-final-max-v2.ipynb', encoding='utf-8'))
lines = "".join(nb['cells'][12]['source']).splitlines()
for idx in range(66, min(140, len(lines))):
    print(f"{idx}: {lines[idx]}")
