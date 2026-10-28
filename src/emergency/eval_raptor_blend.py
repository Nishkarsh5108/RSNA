import sys, os, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final_max_gold_eval as FM

def main():
    gold = pd.read_csv(f'{FM.R}/data/labels_v3/gold_labels_original.csv').set_index('StudyInstanceUID')
    rap = FM.load('raptor')
    ids = [u for u in gold.index if u in rap]
    Y = gold.loc[ids, FM.LAB].values.astype(int)

    d_new = np.load(rf'{FM.R}/output/emergency_research/raptor_output/raptor_gold_raptor_vision_pure_swa.npz')
    new_map = {uid: d_new['pred'][i] for i, uid in enumerate(d_new['ids'])}
    P_new = np.stack([new_map[u] for u in ids])
    P_old = np.stack([rap[u]['p384_pair'] for u in ids])

    for w_new in [1.0, 0.7, 0.6, 0.5, 0.4, 0.0]:
        p = w_new * P_new + (1 - w_new) * P_old
        auc = np.mean([roc_auc_score(Y[:, j], p[:, j]) for j in range(12)])
        print(f"Raptor Slot: {w_new*100:3.0f}% New SWA + {(1-w_new)*100:3.0f}% Old -> Gold AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
