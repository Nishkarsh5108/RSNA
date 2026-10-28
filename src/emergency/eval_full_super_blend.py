import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap, s6f, s6ft_v3, s6ft_v1 = FM.load('raptor'), FM.load('s6fall'), FM.load('s6ft_v3'), FM.load('s6ft')
    r6 = pd.read_csv(f'{FM.T1}/r6/predictions.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = [u for u in gold.index if u in rap and u in s6f and u in r6.index and u in s6ft_v3 and u in s6ft_v1]

    Y = gold.loc[ids, FM.LAB].values.astype(int)

    d_new_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure_swa.npz')
    new_rap_map = {uid: d_new_rap['pred'][i] for i, uid in enumerate(d_new_rap['ids'])}
    P_new_rap = np.stack([new_rap_map[u] for u in ids])
    P_old_rap = np.stack([rap[u]['p384_pair'] for u in ids])
    P_rap_blend = 0.60 * P_new_rap + 0.40 * P_old_rap

    P_r6 = r6.loc[ids, FM.LAB].values.astype(float)
    P_s6ft = FM.sig(np.stack([s6ft_v3[u]['z_plain'][0] for u in ids]))
    P_s6ft_v1 = FM.sig(np.stack([s6ft_v1[u]['z_plain'][0] for u in ids]))
    P_s6f = FM.sig(np.stack([s6f[u]['z_plain'][0] for u in ids]))

    # Public baseline (0.958 LB):
    p_base = 0.50 * P_r6 + 0.25 * P_old_rap + 0.25 * P_s6f
    auc_base = np.mean([roc_auc_score(Y[:, j], p_base[:, j]) for j in range(12)])

    # v4 blend (currently running on LB):
    p_v4 = 0.50 * P_r6 + 0.25 * P_old_rap + 0.20 * P_s6ft + 0.05 * P_s6ft_v1
    auc_v4 = np.mean([roc_auc_score(Y[:, j], p_v4[:, j]) for j in range(12)])

    # Candidate v5 super-blend (dual Raptor 60/40 + finding-routed compact slot):
    P_v5 = np.zeros_like(P_r6)
    for j, c in enumerate(FM.LAB):
        if c == 'MCL':
            P_v5[:, j] = 0.30 * P_r6[:, j] + 0.10 * P_rap_blend[:, j] + 0.60 * P_s6ft_v1[:, j]
        elif c in ['Lateral Meniscus']:
            P_v5[:, j] = 0.35 * P_r6[:, j] + 0.15 * P_rap_blend[:, j] + 0.30 * P_s6ft_v1[:, j] + 0.20 * P_s6ft[:, j]
        elif c in ['Effusion', "Baker's", 'Synovitis']:
            P_v5[:, j] = 0.40 * P_r6[:, j] + 0.15 * P_rap_blend[:, j] + 0.45 * P_s6ft[:, j]
        elif c in ['Medial OA', 'Contusion', 'Fracture']:
            P_v5[:, j] = 0.45 * P_r6[:, j] + 0.35 * P_rap_blend[:, j] + 0.20 * P_s6ft[:, j]
        else: # ACL, Lateral OA, PF OA, Medial Meniscus
            P_v5[:, j] = 0.60 * P_r6[:, j] + 0.15 * P_rap_blend[:, j] + 0.25 * P_s6ft[:, j]

    auc_v5 = np.mean([roc_auc_score(Y[:, j], P_v5[:, j]) for j in range(12)])

    print(f"Baseline Public Ensemble (LB 0.958)     : {auc_base:.4f}")
    print(f"final-max v4 (Currently Scoring)        : {auc_v4:.4f} (delta = {auc_v4 - auc_base:+.4f})")
    print(f"final-max v5 (Candidate Super-Ensemble) : {auc_v5:.4f} (delta = {auc_v5 - auc_base:+.4f})")

if __name__ == '__main__':
    main()
