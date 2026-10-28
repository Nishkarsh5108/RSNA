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

    W_RAPTOR_C = {
        'Medial OA': 0.45,
        'Lateral OA': 0.20,
        'PF OA': 0.35,
        'Synovitis': 0.25,
        "Baker's": 0.25,
        'Fracture': 0.55,
        'Contusion': 0.40,
        'MCL': 0.15,
        'Lateral Meniscus': 0.20,
        'ACL': 0.20,
        'Effusion': 0.20,
        'Medial Meniscus': 0.30,
    }

    P_v5 = np.zeros_like(P_r6)
    for j, c in enumerate(FM.LAB):
        # Compact slot
        if c == 'MCL':
            w_r6, w_ft1, w_spec, w_s6f = 0.35, 0.60, 0.05, 0.00
        elif c == 'Lateral Meniscus':
            w_r6, w_ft1, w_spec, w_s6f = 0.40, 0.35, 0.25, 0.00
        elif c in ['Effusion', "Baker's", 'Synovitis']:
            w_r6, w_ft1, w_spec, w_s6f = 0.45, 0.05, 0.50, 0.00
        elif c in ['Medial OA', 'Contusion', 'Fracture']:
            w_r6, w_ft1, w_spec, w_s6f = 0.50, 0.05, 0.35, 0.10
        else: # ACL, Lateral OA, PF OA, Medial Meniscus
            w_r6, w_ft1, w_spec, w_s6f = 0.65, 0.05, 0.30, 0.00
        
        rc = w_r6 * P_r6[:, j] + w_ft1 * P_s6ft_v1[:, j] + w_spec * P_s6ft[:, j] + w_s6f * P_s6f[:, j]
        rr = P_rap_blend[:, j]
        wr = W_RAPTOR_C.get(c, 0.30)
        P_v5[:, j] = wr * rr + (1.0 - wr) * rc

    auc_v5 = np.mean([roc_auc_score(Y[:, j], P_v5[:, j]) for j in range(12)])
    p_base = 0.50 * P_r6 + 0.25 * P_old_rap + 0.25 * P_s6f
    auc_base = np.mean([roc_auc_score(Y[:, j], p_base[:, j]) for j in range(12)])

    print(f"Baseline Public Ensemble (LB 0.958)     : {auc_base:.4f}")
    print(f"final-max v5 (2-Tier Slot Formulation)  : {auc_v5:.4f} (delta = {auc_v5 - auc_base:+.4f})")

if __name__ == '__main__':
    main()
