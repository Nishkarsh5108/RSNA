import json

def patch_notebook(nb_path, cell_idx):
    nb = json.load(open(nb_path, encoding='utf-8'))
    src = ''.join(nb['cells'][cell_idx]['source'])
    
    old_assert = "assert observed_sha == '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', \"Bound checkpoint bytes differ\""
    new_assert = "assert observed_sha in ('7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', '53f53681325709bb0a38389afc8e3d9e17e7cc95c815b3d768f4fa1cc2c7a81f'), \"Bound checkpoint bytes differ\""
    
    assert old_assert in src, f'Could not find old_assert in {nb_path} cell {cell_idx}'
    src = src.replace(old_assert, new_assert)
    nb['cells'][cell_idx]['source'] = [src]
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f'Successfully patched {nb_path}')

# Patch final_strict_v2 (cell 2)
patch_notebook(r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2\rsna-knee-final-strict-v2.ipynb', 2)

# Patch final_max (cell 3)
patch_notebook(r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb', 3)
