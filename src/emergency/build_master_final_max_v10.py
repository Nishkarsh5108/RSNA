import json
import shutil
import os

nb_path = r'notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb'
bak_path = r'notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb.v9.bak'
shutil.copy2(nb_path, bak_path)

nb = json.load(open(nb_path, encoding='utf-8'))

# 1. Update Cell 0 (Title / Description)
nb['cells'][0]['source'] = [
    "# rsna knee final max v10: master blend dual coatnet (pure vision 60% + public 40%) + heliosli r6 + s6fall + calibrated specialists\n",
    "\n",
    "master 4-way super-ensemble: dual coatnet-384 + r6 (384 px) + s6fall (all-data anchor) + calibrated specialists s6ft_specialist (0.9402) and s6ft_v1 (0.9864 MCL) with t2 transform."
]

# 2. Update Cell 9: DEADLINE_HOURS = 6.2, x-eval-stride = 2
c9_src = ''.join(nb['cells'][9]['source'])
assert 'DEADLINE_HOURS' in c9_src, "DEADLINE_HOURS not in Cell 9"
c9_src = c9_src.replace('DEADLINE_HOURS = 6.8', 'DEADLINE_HOURS = 6.2')
c9_src = c9_src.replace('DEADLINE_HOURS = 8.3', 'DEADLINE_HOURS = 6.2')

# update x-eval-stride to 2 if present as 1
if "'--x-eval-stride', '1'" in c9_src:
    c9_src = c9_src.replace("'--x-eval-stride', '1'", "'--x-eval-stride', '2'")
nb['cells'][9]['source'] = [c9_src]

# 3. Update Cell 12: Anchored Specialist Residual Routing
c12_src = ''.join(nb['cells'][12]['source'])

old_routing = """    # Abnormality routing matrix
    if c == 'MCL':
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.10, 0.30, 0.60, 0.00, 0.00
    elif c == 'Lateral Meniscus':
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.35, 0.30, 0.20, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis']:
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.40, 0.00, 0.45, 0.00
    elif c in ['Medial OA', 'Contusion', 'Fracture']:
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.35, 0.45, 0.00, 0.20, 0.00
    else: # ACL, Lateral OA, PF OA, Medial Meniscus
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.60, 0.00, 0.25, 0.00"""

new_routing = """    # Anchored Specialist Residual Routing (Master Blend v10)
    # Core Anchor: R6 (0.45-0.60) + Dual CoAtNet (0.20-0.25) + S6Fall (0.15)
    # Specialist Booster: 0.15-0.20 residual weight to validated finding champions
    if c == 'MCL':
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.20, 0.45, 0.15, 0.20, 0.00
    elif c == 'Lateral Meniscus':
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.20, 0.45, 0.15, 0.10, 0.10
    elif c in ['Effusion', "Baker's", 'Synovitis', 'Contusion']:
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.20, 0.45, 0.15, 0.00, 0.20
    elif c == 'Medial Meniscus':
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.20, 0.50, 0.15, 0.00, 0.15
    elif c in ['Medial OA', 'Lateral OA', 'PF OA', 'Fracture']:
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.25, 0.50, 0.15, 0.00, 0.10
    else: # ACL
        w_rap, w_r6, w_s6f, w_ft1, w_spec = 0.20, 0.60, 0.15, 0.00, 0.05"""

assert old_routing in c12_src, "old_routing not found in Cell 12"
c12_src = c12_src.replace(old_routing, new_routing)
nb['cells'][12]['source'] = [c12_src]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully built Master Final Max v10!")
