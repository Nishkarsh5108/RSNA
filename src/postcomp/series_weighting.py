"""Own idea N4b / N6: combine a study's series with per-finding weights for each series type instead of the plain mean.

The box-pooling model predicts every series on its own; the study score is the mean over its series. A series type
that cannot show a finding (e.g. an axial series for a meniscal tear) pulls that mean toward the prior. Here each
finding gets one non-negative weight per series type (plane x fluid-sensitive = 6 types):

    study score = sum_s w[k, type(s)] * p[s, k] / sum_s w[k, type(s)]

Weights are fitted on held-out predictions (BCE of the study score, shrunk toward equal weights) and judged honestly:
  * 5-fold cross-fitting inside the 870 held-out studies (weights never see the studies they are scored on);
  * gold-58 with weights fitted on all 870 held-out studies.
Fixed aggregators that need no fitting (mean of logits, max, mean of the top 2) are scored as well.

  python series_weighting.py --preds D:/rsna_work/sharp_f0/series_sharp.csv [--l2 1.0] [--save weights.json]
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from score_bp import E, L, R, auc_cols, fmt, macro, paired

TYPES = [(pl, fs) for pl in ("Sagittal", "Coronal", "Axial") for fs in (1, 0)]
TNAME = [f"{pl[:3]}{'-FS' if fs else ''}" for pl, fs in TYPES]


def tensors(d, studies):
    """p [S, 12], study index [S], type one-hot [S, 6] for the series of `studies` (in that order)."""
    pos = {s: i for i, s in enumerate(studies)}
    d = d[d.StudyInstanceUID.isin(pos)]
    t = np.array([TYPES.index((pl, int(fs))) for pl, fs in zip(d.Anatomical_Plane, d.Fluid_Sensitive)])
    return (torch.tensor(d[L].to_numpy(np.float64)), torch.tensor(d.StudyInstanceUID.map(pos).to_numpy()),
            torch.nn.functional.one_hot(torch.tensor(t), len(TYPES)).double())


def combine(p, si, onehot, theta, n):
    w = onehot @ torch.exp(theta).T                       # [S, 12]: weight of each series for each finding
    num = torch.zeros(n, p.shape[1], dtype=p.dtype).index_add(0, si, w * p)
    den = torch.zeros(n, p.shape[1], dtype=p.dtype).index_add(0, si, w)
    return num / den.clamp(min=1e-9)


def fit(p, si, onehot, y, l2):
    theta = torch.zeros(len(L), len(TYPES), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")
    yt = torch.tensor(y, dtype=torch.float64)

    def closure():
        opt.zero_grad()
        q = combine(p, si, onehot, theta, len(y)).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(q, yt) + l2 * (theta - theta.mean(1, keepdim=True)).pow(2).mean()
        loss.backward()
        return loss

    opt.step(closure)
    th = theta.detach()
    return th - th.mean(1, keepdim=True)


def fixed(d, studies, how):
    g = d[d.StudyInstanceUID.isin(set(studies))].groupby("StudyInstanceUID")[L]
    if how == "mean":
        out = g.mean()
    elif how == "logit":
        z = d[L].clip(1e-6, 1 - 1e-6)
        z = np.log(z / (1 - z))
        z["StudyInstanceUID"] = d.StudyInstanceUID
        out = 1 / (1 + np.exp(-z[z.StudyInstanceUID.isin(set(studies))].groupby("StudyInstanceUID")[L].mean()))
    elif how == "max":
        out = g.max()
    else:                                                  # mean of the two highest series
        out = g.agg(lambda c: np.sort(c.to_numpy())[-2:].mean())
    return out.loc[studies].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="series_sharp.csv (uses sharp == 1) or a fold*_series_epoch*.csv with type columns")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    df = pd.read_csv(a.preds)
    if "sharp" in df.columns:
        df = df[df.sharp == 1.0]
    if "model" not in df.columns:
        df["model"] = "m"
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")[L]
    ref = {"7th": pd.read_csv(f"{E}/songqizhou/8_final_label.csv").set_index("StudyInstanceUID")[L],
           "1st": pd.read_csv(f"{E}/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID")[L]}
    saved = {}
    for name, d in df.groupby("model"):
        studies = sorted(set(d.StudyInstanceUID) - set(gold.index))
        gi = [u for u in gold.index if u in set(d.StudyInstanceUID)]
        print(f"\n================ model {name}: {len(studies)} held-out studies, {len(gi)} gold studies ================")
        yg = gold.loc[gi].to_numpy()
        base_g = fixed(d, gi, "mean")
        for how in ("mean", "logit", "max", "top2"):
            line = f"{how:6s} gold {macro(yg, fixed(d, gi, how)):.4f}"
            for k in ("7th", "1st"):
                line += f" | fold vs {k} {macro((ref[k].loc[studies].to_numpy() >= 0.5).astype(int), fixed(d, studies, how)):.4f}"
            print(line)
        rng = np.random.default_rng(0)
        part = rng.permutation(len(studies)) % 5
        for k in ("7th", "1st"):                           # labels the weights are fitted on (and the fold is scored on)
            y = (ref[k].loc[studies].to_numpy() >= 0.5).astype(float)
            base = fixed(d, studies, "mean")
            oof = np.zeros_like(base)
            for f in range(5):
                tr = [s for s, q in zip(studies, part) if q != f]
                te = [s for s, q in zip(studies, part) if q == f]
                th = fit(*tensors(d, tr), y[part != f], a.l2)
                p, si, oh = tensors(d, te)
                oof[part == f] = combine(p, si, oh, th, len(te)).numpy()
            th_all = fit(*tensors(d, studies), y, a.l2)
            p, si, oh = tensors(d, gi)
            wg = combine(p, si, oh, th_all, len(gi)).numpy()
            print(f"\n-- weights fitted on the {k} labels (l2 {a.l2})")
            print(f"fold, cross-fitted, vs {k}: weighted {macro(y.astype(int), oof):.4f} | plain mean {macro(y.astype(int), base):.4f} | paired",
                  fmt(paired(y.astype(int), oof, base, 1000)))
            other = "1st" if k == "7th" else "7th"
            yo = (ref[other].loc[studies].to_numpy() >= 0.5).astype(int)
            print(f"fold, cross-fitted, vs {other}: weighted {macro(yo, oof):.4f} | plain mean {macro(yo, base):.4f} | paired", fmt(paired(yo, oof, base, 1000)))
            print(f"gold-58 (weights from all held-out studies): weighted {macro(yg, wg):.4f} | plain mean {macro(yg, base_g):.4f} | paired",
                  fmt(paired(yg, wg, base_g, 4000)))
            t = pd.DataFrame(np.exp(th_all.numpy()), index=L, columns=TNAME)
            t["fold gain"] = auc_cols(y.astype(int), oof) - auc_cols(y.astype(int), base)
            t["gold gain"] = auc_cols(yg, wg) - auc_cols(yg, base_g)
            print(t.round(3).to_string())
            saved[f"{name}/{k}"] = {lab: dict(zip(TNAME, np.exp(th_all.numpy()[i]).round(4).tolist())) for i, lab in enumerate(L)}
    if a.save:
        json.dump(saved, open(a.save, "w"), indent=1)


if __name__ == "__main__":
    main()
