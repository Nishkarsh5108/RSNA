import json

nb = json.load(open('notebooks/candidate_A/a_strict/rsna-knee-a-strict.ipynb', encoding='utf-8'))
src5 = "".join(nb['cells'][5]['source'])
for line in src5.splitlines()[-25:]:
    print(line)
