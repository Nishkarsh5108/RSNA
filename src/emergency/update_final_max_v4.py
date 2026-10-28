import json
import shutil
import os

def update_final_max():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    bak_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb.bak'
    shutil.copy2(nb_path, bak_path)

    nb = json.load(open(nb_path, encoding='utf-8'))

    # 1. Update Cell 0 (Title / Description)
    nb['cells'][0]['source'] = [
        "# rsna knee final max v4: nartaa coatnet 384 + 224, heliosli r6, custom s6ft specialist (0.9402) inpass\n",
        "\n",
        "single-decode pass: r6 (384 px) runs in 0.50 slot, in-pass s6ft specialist (0.9402 Gold-58) + s6ft_v1 (0.9864 MCL) in 0.35 slot, nartaa coatnet in 0.15 slot with t2 transform."
    ]

    # 2. Update Cell 9
    c9_src = ''.join(nb['cells'][9]['source'])
    c9_src = c9_src.replace(
        "S6FT_DIR = find_mount('rsna-knee-s6ft-weights', 's6ft_v2_ema.pt')\nassert (S6FT_DIR / 's6ft_v2_ema.pt').is_file() and (S6FT_DIR / 's6ft_v1_ema.pt').is_file(), 's6ft weights missing'\nWORKER = 'compact_infer_x.py'\n_x_ckpts = [str(S6F_DIR / 'E_prior_s42_fall.pt'), str(S6FT_DIR / 's6ft_v2_ema.pt'), str(S6FT_DIR / 's6ft_v1_ema.pt')]",
        "S6FT_DIR = find_mount('rsna-knee-s6ft-weights', 's6ft_specialist_ema.pt')\nassert (S6FT_DIR / 's6ft_specialist_ema.pt').is_file() and (S6FT_DIR / 's6ft_v1_ema.pt').is_file(), 's6ft weights missing'\nWORKER = 'compact_infer_x.py'\n_x_ckpts = [str(S6F_DIR / 'E_prior_s42_fall.pt'), str(S6FT_DIR / 's6ft_specialist_ema.pt'), str(S6FT_DIR / 's6ft_v1_ema.pt')]"
    )
    nb['cells'][9]['source'] = [c9_src]

    # 3. Update Cell 10
    c10_src = ''.join(nb['cells'][10]['source'])
    c10_src = c10_src.replace(
        "S6F_PROBS = WORK / 's6fall_inpass.csv'\nS6FT2_PROBS = WORK / 's6ft_v2_inpass.csv'\nS6FT1_PROBS = WORK / 's6ft_v1_inpass.csv'\n_x, S6F_OK = _collect_x('worker*/x_E_prior_s42_fall.csv', S6F_PROBS)\n_x2, S6FT2_OK = _collect_x('worker*/x_s6ft_v2_ema.csv', S6FT2_PROBS)\n_x1, S6FT1_OK = _collect_x('worker*/x_s6ft_v1_ema.csv', S6FT1_PROBS)",
        "S6F_PROBS = WORK / 's6fall_inpass.csv'\nS6FT_SPEC_PROBS = WORK / 's6ft_specialist_inpass.csv'\nS6FT1_PROBS = WORK / 's6ft_v1_inpass.csv'\n_x, S6F_OK = _collect_x('worker*/x_E_prior_s42_fall.csv', S6F_PROBS)\n_x_spec, S6FT_SPEC_OK = _collect_x('worker*/x_s6ft_specialist_ema.csv', S6FT_SPEC_PROBS)\n_x1, S6FT1_OK = _collect_x('worker*/x_s6ft_v1_ema.csv', S6FT1_PROBS)"
    )
    c10_src = c10_src.replace(
        "'s6ft2_ok': S6FT2_OK",
        "'s6ft_spec_ok': S6FT_SPEC_OK"
    )
    nb['cells'][10]['source'] = [c10_src]

    # 4. Update Cell 12
    c12_src = ''.join(nb['cells'][12]['source'])
    c12_src = c12_src.replace(
        "s6ft2, s6ft2_fail, s6ft2_note = _load_prob_table(WORK / 's6ft_v2_inpass.csv', 'S6FT2_OK', 's6ft_v2')",
        "s6ft_spec, s6ft_spec_fail, s6ft_spec_note = _load_prob_table(S6FT_SPEC_PROBS, 'S6FT_SPEC_OK', 's6ft_specialist')"
    )
    c12_src = c12_src.replace(
        "'s6ft2': s6ft2_note",
        "'s6ft_spec': s6ft_spec_note"
    )

    old_blend_block = """    # Base weights per finding family
    if c in ['MCL', 'Lateral Meniscus']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.55, 0.30, 0.15, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.55, 0.10, 0.35, 0.00
    elif c in ['Medial OA', 'Lateral OA', 'PF OA', 'Fracture']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.60, 0.10, 0.20, 0.10
    else: # ACL, Contusion
        w_r6, w_ft1, w_ft2, w_s6f = 0.60, 0.00, 0.25, 0.15

    # Weights zeroed if model not present or study failed
    wo = np.where(our_fail, 0.0, w_r6)
    w1 = np.where(s6ft1_fail if s6ft1 is not None else True, 0.0, w_ft1)
    w2 = np.where(s6ft2_fail if s6ft2 is not None else True, 0.0, w_ft2)
    ws = np.where(s6f_fail if s6f is not None else True, 0.0, w_s6f)

    r1 = f(s6ft1[c].values, ~s6ft1_fail) if s6ft1 is not None else np.zeros(n)
    r2 = f(s6ft2[c].values, ~s6ft2_fail) if s6ft2 is not None else np.zeros(n)
    rs = f(s6f[c].values, ~s6f_fail) if s6f is not None else np.zeros(n)

    den = wo + w1 + w2 + ws
    num = wo * np.nan_to_num(ro) + w1 * np.nan_to_num(r1) + w2 * np.nan_to_num(r2) + ws * np.nan_to_num(rs)"""

    new_blend_block = """    # Base weights per finding family [final-max v4: s6ft_specialist 0.9402 Gold]
    if c in ['MCL', 'Lateral Meniscus']:
        w_r6, w_ft1, w_spec, w_s6f = 0.50, 0.35, 0.15, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis', 'Contusion']:
        w_r6, w_ft1, w_spec, w_s6f = 0.50, 0.05, 0.45, 0.00
    elif c in ['Medial Meniscus']:
        w_r6, w_ft1, w_spec, w_s6f = 0.50, 0.10, 0.40, 0.00
    elif c in ['Medial OA', 'Lateral OA', 'PF OA', 'Fracture']:
        w_r6, w_ft1, w_spec, w_s6f = 0.55, 0.05, 0.30, 0.10
    else: # ACL
        w_r6, w_ft1, w_spec, w_s6f = 0.55, 0.05, 0.30, 0.10

    # Weights zeroed if model not present or study failed
    wo = np.where(our_fail, 0.0, w_r6)
    w1 = np.where(s6ft1_fail if s6ft1 is not None else True, 0.0, w_ft1)
    w_sp = np.where(s6ft_spec_fail if s6ft_spec is not None else True, 0.0, w_spec)
    ws = np.where(s6f_fail if s6f is not None else True, 0.0, w_s6f)

    r1 = f(s6ft1[c].values, ~s6ft1_fail) if s6ft1 is not None else np.zeros(n)
    rsp = f(s6ft_spec[c].values, ~s6ft_spec_fail) if s6ft_spec is not None else np.zeros(n)
    rs = f(s6f[c].values, ~s6f_fail) if s6f is not None else np.zeros(n)

    den = wo + w1 + w_sp + ws
    num = wo * np.nan_to_num(ro) + w1 * np.nan_to_num(r1) + w_sp * np.nan_to_num(rsp) + ws * np.nan_to_num(rs)"""

    assert old_blend_block in c12_src, 'Could not find old_blend_block in Cell 12'
    c12_src = c12_src.replace(old_blend_block, new_blend_block)
    nb['cells'][12]['source'] = [c12_src]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('rsna-knee-final-max.ipynb updated successfully to v4!')

if __name__ == '__main__':
    update_final_max()
