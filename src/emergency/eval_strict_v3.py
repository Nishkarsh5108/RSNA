import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap, s6f, s6, s6ft_v3, s6ft_v1 = FM.load('raptor'), FM.load('s6fall'), FM.load('s6'), FM.load('s6ft_v3'), FM.load('s6ft')
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in s6ft_v3 and u in s6ft_v1]

    Y = gold.loc[ids, FM.LAB].values.astype(int)

    d_new_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure_swa.npz')
    new_rap_map = {uid: d_new_rap['pred'][i] for i, uid in enumerate(d_new_rap['ids'])}
    P_new_rap = np.stack([new_rap_map[u] for u in ids])
    P_old_rap = np.stack([rap[u]['p384_pair'] for u in ids])
    P_rap_blend = 0.60 * P_new_rap + 0.40 * P_old_rap

    P_s6ft = FM.sig(np.stack([s6ft_v3[u]['z_plain'][0] for u in ids]))
    P_s6ft_v1 = FM.sig(np.stack([s6ft_v1[u]['z_plain'][0] for u in ids]))
    P_s6f = FM.sig(np.stack([s6f[u]['z_plain'][0] for u in ids]))
    P_s6 = FM.sig(np.stack([s6[u]['z_plain'][0] for u in ids]))

    # Baseline strict (LB 0.956):
    p_strict_base = 0.50 * P_old_rap + 0.25 * P_s6 + 0.25 * P_s6f
    auc_strict_base = np.mean([roc_auc_score(Y[:, j], p_strict_base[:, j]) for j in range(12)])

    # v2 strict (LB 0.956, submitted yesterday):
    p_strict_v2 = 0.50 * P_old_rap + 0.25 * P_s6 + 0.15 * P_s6ft + 0.10 * P_s6ft_v1
    auc_strict_v2 = np.mean([roc_auc_score(Y[:, j], p_strict_v2[:, j]) for j in range(12)])

    # v3 strict (Candidate Super-Strict FastMRI-free):
    P_strict_v3 = np.zeros_like(P_s6)
    for j, c in enumerate(FM.LAB):
        if c == 'MCL':
            P_strict_v3[:, j] = 0.25 * P_rap_blend[:, j] + 0.15 * P_s6[:, j] + 0.55 * P_s6ft_v1[:, j] + 0.05 * P_s6ft[:, j]
        elif c in ['Lateral Meniscus']:
            P_strict_v3[:, j] = 0.25 * P_rap_blend[:, j] + 0.15 * P_s6[:, j] + 0.35 * P_s6ft_v1[:, j] + 0.25 * P_s6ft[:, j]
        elif c in ['Effusion', "Baker's", 'Synovitis']:
            P_strict_v3[:, j] = 0.25 * P_rap_blend[:, j] + 0.15 * P_s6[:, j] + 0.50 * P_s6ft[:, j] + 0.10 * P_s6ft_v1[:, j]
        elif c in ['Medial OA', 'Contusion', 'Fracture']:
            P_strict_v3[:, j] = 0.50 * P_rap_blend[:, j] + 0.20 * P_s6[:, j] + 0.30 * P_s6ft[:, j]
        else: # ACL, Lateral OA, PF OA, Medial Meniscus
            P_strict_v3[:, j] = 0.40 * P_rap_blend[:, j] + 0.25 * P_s6[:, j] + 0.35 * P_s6ft[:, j]

    auc_strict_v3 = np.mean([roc_auc_score(Y[:, j], P_strict_v3[:, j]) for j in range(12)])

    print(f"Baseline FastMRI-free (LB 0.956)  : {auc_strict_base:.4f}")
    print(f"final-strict v2 (Submitted)       : {auc_strict_v2:.4f} (delta = {auc_strict_v2 - auc_strict_base:+.4f})")
    print(f"final-strict v3 (Super FastMRI-free): {auc_strict_v3:.4f} (delta = {auc_strict_v3 - auc_strict_base:+.4f})")

if __name__ == '__main__':
    main()
