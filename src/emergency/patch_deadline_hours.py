import json

nb_path = r'notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))
cell_idx = 9
src = ''.join(nb['cells'][cell_idx]['source'])

assert 'DEADLINE_HOURS = 8.3' in src, "DEADLINE_HOURS = 8.3 not found in cell 9!"
src = src.replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 6.8')
nb['cells'][cell_idx]['source'] = [src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated DEADLINE_HOURS to 6.8 in final_max!")
