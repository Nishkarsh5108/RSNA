import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap, s6f, s6, s6ft_v3, s6ft_v1 = FM.load('raptor'), FM.load('s6fall'), FM.load('s6'), FM.load('s6ft_v3'), FM.load('s6ft')
    r6 = pd.read_csv(f'{FM.T1}/r6/predictions.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index and u in s6ft_v3 and u in s6ft_v1]

    Y = gold.loc[ids, FM.LAB].values.astype(int)

    # Load newly trained Raptor (both normal and SWA)
    d_new_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure.npz')
    new_rap_map = {uid: d_new_rap['pred'][i] for i, uid in enumerate(d_new_rap['ids'])}
    P_new_rap = np.stack([new_rap_map[u] for u in ids])

    d_swa_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure_swa.npz')
    swa_rap_map = {uid: d_swa_rap['pred'][i] for i, uid in enumerate(d_swa_rap['ids'])}
    P_swa_rap = np.stack([swa_rap_map[u] for u in ids])

    P_r6 = r6.loc[ids, FM.LAB].values.astype(float)
    P_s6ft = FM.sig(np.stack([s6ft_v3[u]['z_plain'][0] for u in ids]))
    P_s6ft_v1 = FM.sig(np.stack([s6ft_v1[u]['z_plain'][0] for u in ids]))
    P_s6f = FM.sig(np.stack([s6f[u]['z_plain'][0] for u in ids]))
    P_s6 = FM.sig(np.stack([s6[u]['z_plain'][0] for u in ids]))
    P_old_rap = np.stack([rap[u]['p384_pair'] for u in ids])

    print("=" * 65)
    print("GOLD-58 STANDALONE AUC PER FINDING (TOP MODELS)")
    print("=" * 65)
    header = f"{'Finding':20s} | {'R6':6s} | {'OldRap':6s} | {'SwaRap':6s} | {'S6F':6s} | {'S6FT_v1':7s} | {'S6FT_v3':7s} | {'Best':10s}"
    print(header)
    print("-" * 65)
    
    models = {
        'R6': P_r6,
        'OldRap': P_old_rap,
        'SwaRap': P_swa_rap,
        'S6F': P_s6f,
        'S6FT_v1': P_s6ft_v1,
        'S6FT_v3': P_s6ft
    }
    
    macro_aucs = {m: [] for m in models}
    for j, lab in enumerate(FM.LAB):
        row = f"{lab:20s} | "
        scores = {}
        for m in models:
            auc = roc_auc_score(Y[:, j], models[m][:, j])
            macro_aucs[m].append(auc)
            scores[m] = auc
            row += f"{auc:.4f} | "
        best_m = max(scores, key=scores.get)
        row += f"{best_m} ({scores[best_m]:.4f})"
        print(row)
    print("-" * 65)
    macro_str = f"{'MACRO AUC':20s} | "
    for m in models:
        macro_str += f"{np.mean(macro_aucs[m]):.4f} | "
    print(macro_str)
    print("=" * 65)

    # 1. Evaluate FastMRI-Free Blends (For Pick 2: final-strict-v3)
    print("\n>>> FAST-MRI FREE BLENDS (PICK 2 OPPORTUNITY) <<<")
    # Baseline final-strict (OldRap 0.50 + S6 0.25 + S6F 0.25)
    p_strict_base = 0.50 * P_old_rap + 0.25 * P_s6 + 0.25 * P_s6f
    auc_strict_base = np.mean([roc_auc_score(Y[:, j], p_strict_base[:, j]) for j in range(12)])
    print(f"Current final-strict baseline (0.50 OldRap + 0.25 S6 + 0.25 S6F): {auc_strict_base:.4f} (LB: 0.956)")

    for w_rap, w_s6ft, w_v1, w_s6 in [
        (0.50, 0.35, 0.15, 0.00),
        (0.40, 0.40, 0.10, 0.10),
        (0.45, 0.35, 0.10, 0.10),
        (0.35, 0.45, 0.10, 0.10),
    ]:
        p_strict_new = w_rap * P_swa_rap + w_s6ft * P_s6ft + w_v1 * P_s6ft_v1 + w_s6 * P_s6
        auc_s = np.mean([roc_auc_score(Y[:, j], p_strict_new[:, j]) for j in range(12)])
        print(f"FastMRI-free ({w_rap:.2f} SwaRap + {w_s6ft:.2f} S6FT_v3 + {w_v1:.2f} v1 + {w_s6:.2f} s6): {auc_s:.4f} (delta = {auc_s - auc_strict_base:+.4f})")

    # 2. Evaluate 4-way Finding-Specific Super-Blend (For Pick 1: final-max v5)
    print("\n>>> 4-WAY SUPER-BLEND WITH SPECIALIST ROUTING (PICK 1 OPPORTUNITY) <<<")
    # Finding-specific weights routing each finding to its champions
    P_super = np.zeros_like(P_r6)
    for j, c in enumerate(FM.LAB):
        if c == 'MCL':
            # S6FT_v1 is dominant on MCL (0.9864)
            P_super[:, j] = 0.35 * P_r6[:, j] + 0.15 * P_swa_rap[:, j] + 0.45 * P_s6ft_v1[:, j] + 0.05 * P_s6ft[:, j]
        elif c in ['Effusion', "Baker's", 'Synovitis']:
            # S6FT_v3 is dominant on Effusion (0.9789) and Synovitis (0.8600)
            P_super[:, j] = 0.40 * P_r6[:, j] + 0.15 * P_swa_rap[:, j] + 0.40 * P_s6ft[:, j] + 0.05 * P_s6ft_v1[:, j]
        elif c in ['Medial OA', 'Lateral OA', 'PF OA', 'Contusion', 'Fracture']:
            # Raptor is monster on Medial OA (0.9907) and Contusion (0.9636)
            P_super[:, j] = 0.45 * P_r6[:, j] + 0.30 * P_swa_rap[:, j] + 0.20 * P_s6ft[:, j] + 0.05 * P_s6f[:, j]
        else: # ACL, Lateral Meniscus, Medial Meniscus
            P_super[:, j] = 0.50 * P_r6[:, j] + 0.20 * P_swa_rap[:, j] + 0.25 * P_s6ft[:, j] + 0.05 * P_s6ft_v1[:, j]

    auc_super = np.mean([roc_auc_score(Y[:, j], P_super[:, j]) for j in range(12)])
    p_base = 0.50 * P_r6 + 0.25 * P_old_rap + 0.25 * P_s6f
    auc_base = np.mean([roc_auc_score(Y[:, j], p_base[:, j]) for j in range(12)])
    print(f"Public Baseline Blend: {auc_base:.4f} (LB: 0.958)")
    print(f"4-Way Finding-Specialist Super-Blend: {auc_super:.4f} (delta = {auc_super - auc_base:+.4f} / +0.0031 on Gold-58!)")

if __name__ == '__main__':
    main()
