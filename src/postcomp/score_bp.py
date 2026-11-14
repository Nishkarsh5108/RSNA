"""Score box-pooling training runs from their per-epoch validation files (fold{F}_epochNN.csv).

Rulers (decision P3): gold-58 is a reference only; models are compared on the held-out fold against a label source
they were NOT trained on ("cross-source"), with a paired bootstrap over studies.

  python score_bp.py --runs a0=D:/rsna_work/kaggle_runs/knee-bp-a0-f0/out a1=D:/rsna_work/kaggle_runs/knee-bp-a1-f0/out
                     [--pair a1 a0] [--partner pt=<our run dir with val_preds_best.csv + gold_preds_SEALED.npz>]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

R = "d:/Coding/kaggle/RSNA"
E = "D:/rsna_work/ext_labels"
L = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
     'Synovitis', "Baker's", 'Contusion', 'Fracture']


def auc_cols(y, p):
    """Mann-Whitney AUC per column; NaN where a column has one class only."""
    out = np.full(y.shape[1], np.nan)
    for j in range(y.shape[1]):
        n1 = y[:, j].sum(); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(p[:, j])
        out[j] = (r[y[:, j] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return out


def macro(y, p):
    return float(np.nanmean(auc_cols(y, p)))


def paired(y, pa, pb, B, seed=0):
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for i in range(B):
        b = rng.integers(0, len(y), len(y))
        d[i] = np.nanmean(auc_cols(y[b], pa[b]) - auc_cols(y[b], pb[b]))
    return macro(y, pa) - macro(y, pb), np.percentile(d, 2.5), np.percentile(d, 97.5), float(np.mean(d > 0))


def rk(a):
    return np.apply_along_axis(rankdata, 0, a) / len(a)


def fmt(t):
    return "%+.4f  95%% CI [%+.4f, %+.4f]  P(>0)=%.3f" % t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=output_dir")
    ap.add_argument("--pair", nargs=2, action="append", default=[], help="paired bootstrap A minus B (last epoch)")
    ap.add_argument("--partner", default="", help="name=run dir of our study-level model held out on the same fold")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")[L]
    ref = {"7th": pd.read_csv(f"{E}/songqizhou/8_final_label.csv").set_index("StudyInstanceUID")[L],
           "1st": pd.read_csv(f"{E}/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID")[L]}
    last = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        files = sorted(glob.glob(os.path.join(d, "fold*_epoch*.csv")))
        files = [f for f in files if "_series_" not in os.path.basename(f)]
        assert files, f"no epoch files in {d}"
        print(f"\n=== {name}: {d} ({len(files)} epochs) ===")
        print("epoch  gold-58   fold vs 7th   fold vs 1st   n_fold  n_gold")
        for f in files:
            p = pd.read_csv(f).set_index("StudyInstanceUID")[L]
            g = p.loc[p.index.isin(gold.index)]
            v = p.loc[~p.index.isin(gold.index)]
            row = [macro(gold.loc[g.index].values, g.values)] + [macro((ref[k].loc[v.index].values >= 0.5).astype(int), v.values) for k in ("7th", "1st")]
            ep = int(re.search(r"epoch(\d+)", os.path.basename(f)).group(1))
            print(f"{ep:5d}  {row[0]:.4f}    {row[1]:.4f}        {row[2]:.4f}       {len(v):5d}  {len(g):5d}")
        last[name] = (g.reindex(gold.index), v.sort_index())
    names = list(last)
    print("\n=== last epoch, per finding ===")
    t = {}
    for n in names:
        g, v = last[n]
        t[f"{n} gold"] = auc_cols(gold.values, g.values)
        for k in ("7th", "1st"):
            t[f"{n} f.v.{k}"] = auc_cols((ref[k].loc[v.index].values >= 0.5).astype(int), v.values)
    t = pd.DataFrame(t, index=L)
    t.loc["MACRO"] = t.mean()
    print(t.round(3).to_string())

    for x, y in a.pair:
        (gx, vx), (gy, vy) = last[x], last[y]
        assert (vx.index == vy.index).all(), "the two runs hold out different studies"
        print(f"\n=== paired bootstrap, {x} minus {y} (last epoch) ===")
        print("gold-58          :", fmt(paired(gold.values, gx.values, gy.values, a.boot * 2)))
        for k in ("7th", "1st"):
            print(f"fold vs {k} labels:", fmt(paired((ref[k].loc[vx.index].values >= 0.5).astype(int), vx.values, vy.values, a.boot)))
        ens_g, ens_v = rk(gx.values) + rk(gy.values), rk(vx.values) + rk(vy.values)
        print(f"rank-mean of both: gold {macro(gold.values, ens_g):.4f} | fold vs 7th "
              f"{macro((ref['7th'].loc[vx.index].values >= 0.5).astype(int), ens_v):.4f} | fold vs 1st "
              f"{macro((ref['1st'].loc[vx.index].values >= 0.5).astype(int), ens_v):.4f}")

    if a.partner:
        pn, pd_ = a.partner.split("=", 1)
        z = np.load(f"{pd_}/gold_preds_SEALED.npz", allow_pickle=True)
        pg = pd.DataFrame(z["p"], index=z["uids"], columns=L).loc[gold.index]
        pv = pd.read_csv(f"{pd_}/val_preds_best.csv").set_index("StudyInstanceUID")[L]
        print(f"\n=== partner {pn} (our study-level model) ===")
        print(f"alone: gold {macro(gold.values, pg.values):.4f}")
        for n in names:
            g, v = last[n]
            common = v.index.intersection(pv.index)
            print(f"-- with {n}: {len(common)} shared held-out studies")
            vv, pp = v.loc[common].values, pv.loc[common].values
            for w in (0.2, 0.33, 0.5):
                eg = (1 - w) * rk(g.values) + w * rk(pg.values)
                ev = (1 - w) * rk(vv) + w * rk(pp)
                line = f"   weight {w:.2f}: gold {macro(gold.values, eg):.4f} ({macro(gold.values, eg) - macro(gold.values, g.values):+.4f})"
                for k in ("7th", "1st"):
                    y = (ref[k].loc[common].values >= 0.5).astype(int)
                    line += f" | fold vs {k} {macro(y, ev):.4f} ({macro(y, ev) - macro(y, vv):+.4f}; partner alone {macro(y, pp):.4f})"
                print(line)
            y1 = (ref["1st"].loc[common].values >= 0.5).astype(int)
            print("   paired, (0.67 model + 0.33 partner) minus model, fold vs 1st:", fmt(paired(y1, 0.67 * rk(vv) + 0.33 * rk(pp), rk(vv), a.boot)))
            print("   paired, same, gold-58                                      :",
                  fmt(paired(gold.values, 0.67 * rk(g.values) + 0.33 * rk(pg.values), rk(g.values), a.boot * 2)))


if __name__ == "__main__":
    main()
