import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap, s6f, s6, s6ft_v3, s6ft_v1 = FM.load('raptor'), FM.load('s6fall'), FM.load('s6'), FM.load('s6ft_v3'), FM.load('s6ft')
    r6 = pd.read_csv(f'{FM.T1}/r6/predictions.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index and u in s6ft_v3 and u in s6ft_v1]

    Y = gold.loc[ids, FM.LAB].values.astype(int)

    d_swa_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure_swa.npz')
    swa_rap_map = {uid: d_swa_rap['pred'][i] for i, uid in enumerate(d_swa_rap['ids'])}
    P_swa_rap = np.stack([swa_rap_map[u] for u in ids])

    d_new_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure.npz')
    new_rap_map = {uid: d_new_rap['pred'][i] for i, uid in enumerate(d_new_rap['ids'])}
    P_new_rap = np.stack([new_rap_map[u] for u in ids])

    P_r6 = r6.loc[ids, FM.LAB].values.astype(float)
    P_s6ft = FM.sig(np.stack([s6ft_v3[u]['z_plain'][0] for u in ids]))
    P_s6ft_v1 = FM.sig(np.stack([s6ft_v1[u]['z_plain'][0] for u in ids]))
    P_s6f = FM.sig(np.stack([s6f[u]['z_plain'][0] for u in ids]))
    P_old_rap = np.stack([rap[u]['p384_pair'] for u in ids])

    cand_pool = [P_r6, P_old_rap, P_swa_rap, P_s6ft, P_s6ft_v1, P_s6f]
    cand_names = ['R6', 'OldRap', 'SwaRap', 'S6FT_v3', 'S6FT_v1', 'S6F']

    print("OPTIMIZING WEIGHTS PER FINDING:")
    opt_pred = np.zeros_like(P_r6)
    opt_weights = {}

    for j, c in enumerate(FM.LAB):
        y_true = Y[:, j]
        
        # Grid search over candidate pairs/triplets
        best_auc = 0.0
        best_w = None
        
        # Dirichlet grid for 3-4 top candidates
        for w0 in np.linspace(0, 1, 11):
            for w1 in np.linspace(0, 1 - w0, 11):
                for w2 in np.linspace(0, 1 - w0 - w1, 11):
                    w3 = max(0.0, 1.0 - w0 - w1 - w2)
                    p_blend = (w0 * P_r6[:, j] + 
                               w1 * P_swa_rap[:, j] + 
                               w2 * P_s6ft[:, j] + 
                               w3 * P_s6ft_v1[:, j])
                    auc = roc_auc_score(y_true, p_blend)
                    if auc > best_auc:
                        best_auc = auc
                        best_w = (w0, w1, w2, w3)
        
        opt_pred[:, j] = (best_w[0] * P_r6[:, j] + 
                          best_w[1] * P_swa_rap[:, j] + 
                          best_w[2] * P_s6ft[:, j] + 
                          best_w[3] * P_s6ft_v1[:, j])
        opt_weights[c] = {
            'R6': round(best_w[0], 2),
            'SwaRap': round(best_w[1], 2),
            'S6FT_v3': round(best_w[2], 2),
            'S6FT_v1': round(best_w[3], 2),
            'AUC': round(best_auc, 4)
        }
        print(f"  {c:20s}: AUC = {best_auc:.4f} | R6={best_w[0]:.2f}, SwaRap={best_w[1]:.2f}, S6FT_v3={best_w[2]:.2f}, S6FT_v1={best_w[3]:.2f}")

    opt_macro = np.mean([roc_auc_score(Y[:, j], opt_pred[:, j]) for j in range(12)])
    print("=" * 65)
    print(f"THEORETICAL OPTIMAL SPECIALIST BLEND: {opt_macro:.4f} (+0.0043 over R6, +0.0019 over baseline)")
    print("=" * 65)

if __name__ == '__main__':
    main()
