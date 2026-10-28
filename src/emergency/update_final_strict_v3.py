import json
import shutil
import os

def update_final_strict():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2\rsna-knee-final-strict-v2.ipynb'
    bak_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_strict_v2\rsna-knee-final-strict-v2.ipynb.bak'
    shutil.copy2(nb_path, bak_path)

    nb = json.load(open(nb_path, encoding='utf-8'))

    # 1. Update Cell 0 (Title / Description)
    nb['cells'][0]['source'] = [
        "# rsna knee final strict v3: dual coatnet (pure vision 60% + public 40%) + heliosli s6 + s6ft specialist (0.9402) + s6ft_v1 (0.9864 MCL)\n",
        "\n",
        "100% fastMRI-free super-ensemble: dual coatnet-384 + in-pass s6ft specialist (0.9402 Gold-58) + s6ft_v1 (0.9864 MCL) + heliosli s6 with T2 transform and specialist abnormality routing."
    ]

    # 2. Update Cell 2 (Raptor Arms: 60% New Pure Vision SWA + 40% Old SWA)
    c2_src = ''.join(nb['cells'][2]['source'])
    old_arms = 'ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean\n    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n]'
    new_arms = 'ARMS = [   # dual CoAtNet: 60% new pure-vision SWA + 40% public SWA (+0.0034 on Gold)\n    {"file": "raptor_ft_vision_pure.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.60},\n    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.40},\n]'
    assert old_arms in c2_src, 'Could not find old_arms in Cell 2'
    c2_src = c2_src.replace(old_arms, new_arms)
    nb['cells'][2]['source'] = [c2_src]

    # 3. Update Cell 6 (Mounts and in-pass models)
    c6_src = ''.join(nb['cells'][6]['source'])
    c6_src = c6_src.replace(
        "S6FT_DIR = find_mount('rsna-knee-s6ft-weights', 's6ft_v2_ema.pt')\nassert (S6FT_DIR / 's6ft_v2_ema.pt').is_file() and (S6FT_DIR / 's6ft_v1_ema.pt').is_file(), 's6ft weights missing'\nWORKER = 'compact_infer_x.py'\n_x_ckpts = [str(S6F_DIR / 'E_prior_s42_fall.pt'), str(S6FT_DIR / 's6ft_v2_ema.pt'), str(S6FT_DIR / 's6ft_v1_ema.pt')]",
        "S6FT_DIR = find_mount('rsna-knee-s6ft-weights', 's6ft_specialist_ema.pt')\nassert (S6FT_DIR / 's6ft_specialist_ema.pt').is_file() and (S6FT_DIR / 's6ft_v1_ema.pt').is_file(), 's6ft weights missing'\nWORKER = 'compact_infer_x.py'\n_x_ckpts = [str(S6F_DIR / 'E_prior_s42_fall.pt'), str(S6FT_DIR / 's6ft_specialist_ema.pt'), str(S6FT_DIR / 's6ft_v1_ema.pt')]"
    )
    nb['cells'][6]['source'] = [c6_src]

    # 4. Update Cell 7 (Collection of in-pass predictions)
    c7_src = ''.join(nb['cells'][7]['source'])
    c7_src = c7_src.replace(
        "S6F_PROBS = WORK / 's6fall_inpass.csv'\nS6FT2_PROBS = WORK / 's6ft_v2_inpass.csv'\nS6FT1_PROBS = WORK / 's6ft_v1_inpass.csv'\n_x, S6F_OK = _collect_x('worker*/x_E_prior_s42_fall.csv', S6F_PROBS)\n_x2, S6FT2_OK = _collect_x('worker*/x_s6ft_v2_ema.csv', S6FT2_PROBS)\n_x1, S6FT1_OK = _collect_x('worker*/x_s6ft_v1_ema.csv', S6FT1_PROBS)",
        "S6F_PROBS = WORK / 's6fall_inpass.csv'\nS6FT_SPEC_PROBS = WORK / 's6ft_specialist_inpass.csv'\nS6FT1_PROBS = WORK / 's6ft_v1_inpass.csv'\n_x, S6F_OK = _collect_x('worker*/x_E_prior_s42_fall.csv', S6F_PROBS)\n_x_spec, S6FT_SPEC_OK = _collect_x('worker*/x_s6ft_specialist_ema.csv', S6FT_SPEC_PROBS)\n_x1, S6FT1_OK = _collect_x('worker*/x_s6ft_v1_ema.csv', S6FT1_PROBS)"
    )
    c7_src = c7_src.replace("'s6ft2_ok': S6FT2_OK", "'s6ft_spec_ok': S6FT_SPEC_OK")
    nb['cells'][7]['source'] = [c7_src]

    # 5. Update Cell 8 (Blend weights with Specialist Routing)
    c8_src = ''.join(nb['cells'][8]['source'])
    c8_src = c8_src.replace(
        "s6ft2, s6ft2_fail, s6ft2_note = _load_prob_table(WORK / 's6ft_v2_inpass.csv', 'S6FT2_OK', 's6ft_v2')",
        "s6ft_spec, s6ft_spec_fail, s6ft_spec_note = _load_prob_table(S6FT_SPEC_PROBS, 'S6FT_SPEC_OK', 's6ft_specialist')"
    )
    c8_src = c8_src.replace("'s6ft2': s6ft2_note", "'s6ft_spec': s6ft_spec_note")

    # Replace the weight assignment block in loop
    old_weight_calc = """    # Combine with weights
    legs_r = [(r_r384, ok_r384, W_R384), (r_r224, ok_r224, W_R224),
              (r_s6, ok_s6, W_S6), (r_ft2, ok_ft2, W_FT2),
              (r_ft1, ok_ft1, W_FT1), (r_s6f, ok_s6f, W_S6F)]

    legs_z = [(z_r384, ok_r384, W_R384), (z_r224, ok_r224, W_R224),
              (z_s6, ok_s6, W_S6), (z_ft2, ok_ft2, W_FT2),
              (z_ft1, ok_ft1, W_FT1), (z_s6f, ok_s6f, W_S6F)]"""

    new_weight_calc = """    # Finding-specific specialist weights [final-strict v3: +0.0056 on Gold-58]
    if c == 'MCL':
        w_rap, w_s6, w_v1, w_spec, w_s6f = 0.25, 0.15, 0.55, 0.05, 0.00
    elif c == 'Lateral Meniscus':
        w_rap, w_s6, w_v1, w_spec, w_s6f = 0.25, 0.15, 0.35, 0.25, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis']:
        w_rap, w_s6, w_v1, w_spec, w_s6f = 0.25, 0.15, 0.10, 0.50, 0.00
    elif c in ['Medial OA', 'Contusion', 'Fracture']:
        w_rap, w_s6, w_v1, w_spec, w_s6f = 0.50, 0.20, 0.00, 0.30, 0.00
    else: # ACL, Lateral OA, PF OA, Medial Meniscus
        w_rap, w_s6, w_v1, w_spec, w_s6f = 0.40, 0.25, 0.00, 0.35, 0.00

    w_r384 = w_rap * 0.70 if r224 is not None else w_rap
    w_r224 = w_rap * 0.30 if r224 is not None else 0.00

    legs_r = [(r_r384, ok_r384, w_r384), (r_r224, ok_r224, w_r224),
              (r_s6, ok_s6, w_s6), (r_ft_spec, ok_ft_spec, w_spec),
              (r_ft1, ok_ft1, w_v1), (r_s6f, ok_s6f, w_s6f)]

    legs_z = [(z_r384, ok_r384, w_r384), (z_r224, ok_r224, w_r224),
              (z_s6, ok_s6, w_s6), (z_ft_spec, ok_ft_spec, w_spec),
              (z_ft1, ok_ft1, w_v1), (z_s6f, ok_s6f, w_s6f)]"""

    # Also adapt variables for s6ft_spec
    c8_src = c8_src.replace("p_ft2  = s6ft2[c].values if s6ft2 is not None else np.zeros(n)", "p_ft_spec = s6ft_spec[c].values if s6ft_spec is not None else np.zeros(n)")
    c8_src = c8_src.replace("ok_ft2  = ~s6ft2_fail if s6ft2 is not None else np.zeros(n, bool)", "ok_ft_spec = ~s6ft_spec_fail if s6ft_spec is not None else np.zeros(n, bool)")
    c8_src = c8_src.replace("r_ft2  = pct_on(p_ft2, ok_ft2) if s6ft2 is not None else np.zeros(n)", "r_ft_spec = pct_on(p_ft_spec, ok_ft_spec) if s6ft_spec is not None else np.zeros(n)")
    c8_src = c8_src.replace("z_ft2  = zl_on(p_ft2, ok_ft2) if s6ft2 is not None else np.zeros(n)", "z_ft_spec = zl_on(p_ft_spec, ok_ft_spec) if s6ft_spec is not None else np.zeros(n)")

    assert old_weight_calc in c8_src, 'Could not find old_weight_calc in Cell 8'
    c8_src = c8_src.replace(old_weight_calc, new_weight_calc)
    nb['cells'][8]['source'] = [c8_src]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('rsna-knee-final-strict-v2.ipynb updated successfully to v3!')

if __name__ == '__main__':
    update_final_strict()
