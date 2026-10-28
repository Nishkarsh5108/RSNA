import json
import shutil
import os

def update_final_max_v5():
    nb_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb'
    bak_path = r'd:\Coding\kaggle\RSNA\notebooks\candidate_A\final_max\rsna-knee-final-max.ipynb.v4.bak'
    shutil.copy2(nb_path, bak_path)

    nb = json.load(open(nb_path, encoding='utf-8'))

    # 1. Update Cell 0 (Title / Description)
    nb['cells'][0]['source'] = [
        "# rsna knee final max v5: dual coatnet (pure vision 60% + public 40%) + heliosli r6 + s6ft specialist (0.9402) + s6ft_v1 (0.9864 MCL)\n",
        "\n",
        "ultimate 4-way super-ensemble: dual coatnet-384 + r6 (384 px) + in-pass s6ft specialist (0.9402 Gold-58) + s6ft_v1 (0.9864 MCL) + s6fall with t2 transform and specialist abnormality routing."
    ]

    # 2. Update Cell 3 (Raptor Arms: 60% New Pure Vision SWA + 40% Old SWA)
    c3_src = ''.join(nb['cells'][3]['source'])
    old_arms = 'ARMS = [   # our 5-fold CoAtNet (public recipe reproduced on our folds/labels), equal-weight rank-mean\n    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 1.0},\n]'
    new_arms = 'ARMS = [   # dual CoAtNet: 60% new pure-vision SWA + 40% public SWA (+0.0034 on Gold)\n    {"file": "raptor_ft_vision_pure.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.60},\n    {"file": "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt", "arch": "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", "res": 384, "w": 0.40},\n]'
    assert old_arms in c3_src, 'Could not find old_arms in Cell 3'
    c3_src = c3_src.replace(old_arms, new_arms)
    nb['cells'][3]['source'] = [c3_src]

    # 3. Update Cell 12 (Direct 4-way Specialist Routing)
    c12_src = ''.join(nb['cells'][12]['source'])
    
    old_slot_blend = """def slot_blend(f, c, neutral):
    # [max v3] Raptor slot (384 + 35 % 224) + specialist compact slot (R6 + s6ft_v2 + s6ft_v1 + s6fall)
    rr = f(raw[c].values, ~rap_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * f(r224[c].values, ~rap_fail)
    ro = f(ours[c].values, ~our_fail)

    # Base weights per finding family [final-max v4: s6ft_specialist 0.9402 Gold]
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
    num = wo * np.nan_to_num(ro) + w1 * np.nan_to_num(r1) + w_sp * np.nan_to_num(rsp) + ws * np.nan_to_num(rs)
    rc = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    comp_fail = (den == 0)

    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * rc
    b[rap_fail] = rc[rap_fail]
    b[comp_fail] = rr[comp_fail]
    b[rap_fail & comp_fail] = neutral
    return b"""

    new_slot_blend = """def slot_blend(f, c, neutral):
    # [final-max v5] Direct 4-way finding specialist routing (+0.0039 on Gold-58)
    rr = f(raw[c].values, ~rap_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * f(r224[c].values, ~rap_fail)
    ro = f(ours[c].values, ~our_fail)
    r1 = f(s6ft1[c].values, ~s6ft1_fail) if s6ft1 is not None else np.zeros(n)
    rsp = f(s6ft_spec[c].values, ~s6ft_spec_fail) if s6ft_spec is not None else np.zeros(n)
    rs = f(s6f[c].values, ~s6f_fail) if s6f is not None else np.zeros(n)

    # Abnormality routing matrix
    if c == 'MCL':
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.10, 0.30, 0.60, 0.00, 0.00
    elif c == 'Lateral Meniscus':
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.35, 0.30, 0.20, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis']:
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.40, 0.00, 0.45, 0.00
    elif c in ['Medial OA', 'Contusion', 'Fracture']:
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.35, 0.45, 0.00, 0.20, 0.00
    else: # ACL, Lateral OA, PF OA, Medial Meniscus
        w_rap, w_r6, w_ft1, w_spec, w_s6f = 0.15, 0.60, 0.00, 0.25, 0.00

    wr = np.where(rap_fail, 0.0, w_rap)
    wo = np.where(our_fail, 0.0, w_r6)
    w1 = np.where(s6ft1_fail if s6ft1 is not None else True, 0.0, w_ft1)
    w_sp = np.where(s6ft_spec_fail if s6ft_spec is not None else True, 0.0, w_spec)
    ws = np.where(s6f_fail if s6f is not None else True, 0.0, w_s6f)

    den = wr + wo + w1 + w_sp + ws
    num = (wr * np.nan_to_num(rr) +
           wo * np.nan_to_num(ro) +
           w1 * np.nan_to_num(r1) +
           w_sp * np.nan_to_num(rsp) +
           ws * np.nan_to_num(rs))
    b = np.where(den > 0, num / np.where(den > 0, den, 1.0), neutral)
    return b"""

    assert old_slot_blend in c12_src, 'Could not find old_slot_blend in Cell 12'
    c12_src = c12_src.replace(old_slot_blend, new_slot_blend)
    nb['cells'][12]['source'] = [c12_src]

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('rsna-knee-final-max.ipynb updated successfully to v5!')

if __name__ == '__main__':
    update_final_max_v5()
