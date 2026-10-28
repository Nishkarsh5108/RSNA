import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap, s6f, s6, s6ft_v3 = FM.load('raptor'), FM.load('s6fall'), FM.load('s6'), FM.load('s6ft_v3')
    r6 = pd.read_csv(f'{FM.T1}/r6/predictions.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
    ids = [u for u in gold.index if u in rap and u in s6f and u in s6 and u in r6.index and u in s6ft_v3]

    Y = gold.loc[ids, FM.LAB].values.astype(int)

    # Load newly trained Raptor
    d_new_rap = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure.npz')
    new_rap_map = {uid: d_new_rap['pred'][i] for i, uid in enumerate(d_new_rap['ids'])}
    P_new_rap = np.stack([new_rap_map[u] for u in ids])

    P_r6 = r6.loc[ids, FM.LAB].values.astype(float)
    P_s6ft = FM.sig(np.stack([s6ft_v3[u]['z_plain'][0] for u in ids]))
    P_s6f = FM.sig(np.stack([s6f[u]['z_plain'][0] for u in ids]))
    P_old_rap = np.stack([rap[u]['p384_pair'] for u in ids])

    print('=== Gold-58 Standalone Macro AUCs ===')
    for name, p in [
        ('Old CoAtNet-384 (pub)', P_old_rap),
        ('New CoAtNet-384 (pure)', P_new_rap),
        ('Heliosli R6 (pub)', P_r6),
        ('Baseline S6F (pub)', P_s6f),
        ('S6FT v3 Specialist', P_s6ft)
    ]:
        aucs = [roc_auc_score(Y[:, j], p[:, j]) for j in range(12)]
        print(f'  {name:26s}: {np.mean(aucs):.4f}')

    print('\n=== Blends on Gold-58 ===')
    p_base = 0.50 * P_r6 + 0.25 * P_old_rap + 0.25 * P_s6f
    auc_base = np.mean([roc_auc_score(Y[:, j], p_base[:, j]) for j in range(12)])
    print(f'Baseline Public Ensemble (0.50 R6 + 0.25 OldRap + 0.25 S6F): {auc_base:.4f}')

    for w_r6, w_rap, w_s6 in [
        (0.50, 0.25, 0.25),
        (0.40, 0.30, 0.30),
        (0.45, 0.25, 0.30),
        (0.40, 0.25, 0.35),
        (0.50, 0.20, 0.30),
        (0.50, 0.15, 0.35)
    ]:
        p_cand = w_r6 * P_r6 + w_rap * P_new_rap + w_s6 * P_s6ft
        auc_cand = np.mean([roc_auc_score(Y[:, j], p_cand[:, j]) for j in range(12)])
        delta = auc_cand - auc_base
        print(f'New Blend ({w_r6:.2f} R6 + {w_rap:.2f} NewRap + {w_s6:.2f} S6FT_v3): {auc_cand:.4f} (delta = {delta:+.4f})')

if __name__ == '__main__':
    main()
