import json

nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2\rsna-knee-final-strict-v2.ipynb'
nb = json.load(open(nb_path, encoding='utf-8'))

c8_src = ''.join(nb['cells'][8]['source'])

old_block = """W_R384 = 0.35 if r224 is not None else 0.50
W_R224 = 0.15 if r224 is not None else 0.00
W_S6   = 0.25
W_FT2  = 0.15 if s6ft2 is not None else 0.00
W_FT1  = 0.10 if s6ft1 is not None else 0.00
W_S6F  = 0.00 if (s6ft2 is not None and s6ft1 is not None) else 0.25"""

new_block = """W_R384 = 0.35 if r224 is not None else 0.50
W_R224 = 0.15 if r224 is not None else 0.00
W_S6   = 0.25
W_FT_SPEC = 0.15 if s6ft_spec is not None else 0.00
W_FT1  = 0.10 if s6ft1 is not None else 0.00
W_S6F  = 0.00 if (s6ft_spec is not None and s6ft1 is not None) else 0.25"""

assert old_block in c8_src, "Could not find old_block in Cell 8"
c8_src = c8_src.replace(old_block, new_block)

old_receipt = "'weights': {'r384': W_R384, 'r224': W_R224, 's6': W_S6, 'ft2': W_FT2, 'ft1': W_FT1, 's6f': W_S6F}"
new_receipt = "'weights': {'r384': W_R384, 'r224': W_R224, 's6': W_S6, 'ft_spec': W_FT_SPEC, 'ft1': W_FT1, 's6f': W_S6F}"

assert old_receipt in c8_src, "Could not find old_receipt in Cell 8"
c8_src = c8_src.replace(old_receipt, new_receipt)

nb['cells'][8]['source'] = [c8_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("final_strict_v2 Cell 8 fixed successfully!")
