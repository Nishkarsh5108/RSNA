import json

nb = json.load(open('notebooks/candidate_A/final_strict/rsna-knee-final-strict.ipynb', encoding='utf-8'))
src4 = "".join(nb['cells'][4]['source'])
for line in src4.splitlines():
    if any(k in line for k in ['compact_infer', 'WORKER', 'EMBEDDED', 'FILES', 'find_mount']):
        print(line[:100])
