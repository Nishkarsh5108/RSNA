"""Task 3 (plan section 12, upgrade E): a gold-calibrated residual adapter on the frozen S6 / s6fall class tokens.

  z_adapted[c] = z_frozen[c] + <w_c, h_c>      w_c in R^H per finding, zero-initialised (12 x H parameters per leg)
  h_c          = class token of finding c (Net.out_norm output; S6: mean over its 4 fold models), plain pass
  loss         = 0.5 * pairwise logistic AUC loss on the training gold studies
               + 0.5 * Bernoulli KL( sigmoid(z_adapted) || sigmoid(z_frozen) ) on the 300 anchor studies,  + 10 * ||w||^2
Training is fixed in advance (Adam, lr 1e-3, 400 full-batch steps, no early stopping, nothing selected on gold).
Evaluation: 5-fold stratified CV over gold_58 (seed 0, stratified by the number of positive findings): every gold study
is scored by the adapter that did not see it. Compared statistic (the brief's rule):
  out-of-fold gold macro AUC of rank(S6_adapted) + rank(S6F_adapted)  vs  rank(S6) + rank(S6F)  (frozen),
  adopt only if delta >= +0.004 with paired-bootstrap (10,000) lower bound > 0, AND the full-gold adapters keep the anchor
  Spearman correlation >= 0.97 (mean over findings, adapted vs frozen, both legs).
The final v1-blend effect (with Raptor) is also reported, descriptive only.
Usage: python -I src/emergency/task3_gold_adapter.py
"""
import glob
import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

R = "D:/Coding/kaggle/RSNA"
T1 = f"{R}/output/emergency_research/task1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]
LAMBDA, LR, STEPS = 10.0, 1e-3, 400


def load_leg(folder):
    out = {}
    for f in glob.glob(f"{T1}/{folder}/*.npz"):
        z = np.load(f, allow_pickle=True)
        out[str(z["uid"])] = (z["z_plain"].mean(0).astype(np.float32), z["tok"].astype(np.float32).mean(0))
    return out


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s, method="average")
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(Y, S, idx=None):
    if idx is not None:
        Y, S = Y[idx], S[idx]
    return np.nanmean([auc(Y[:, j], S[:, j]) for j in range(12)])


def pct_cols(S):
    return np.column_stack([pd.Series(S[:, j]).rank(pct=True).values for j in range(S.shape[1])])


def train_adapter(zg, hg, yg, za, ha):
    """zg [n,12], hg [n,12,H], yg [n,12] gold; za/ha anchors. Returns w [12,H]."""
    torch.manual_seed(0)
    zg, hg, yg = (torch.tensor(x) for x in (zg, hg, yg.astype(np.float32)))
    za, ha = torch.tensor(za), torch.tensor(ha)
    w = torch.zeros(12, hg.shape[-1], requires_grad=True)
    opt = torch.optim.Adam([w], lr=LR)
    pos = [(yg[:, c] == 1).nonzero().flatten() for c in range(12)]
    neg = [(yg[:, c] == 0).nonzero().flatten() for c in range(12)]
    pa = torch.sigmoid(za)
    for _ in range(STEPS):
        s = zg + (hg * w).sum(-1)
        terms = [torch.nn.functional.softplus(-(s[pos[c], c][:, None] - s[neg[c], c][None, :])).mean()
                 for c in range(12) if len(pos[c]) and len(neg[c])]
        l_auc = torch.stack(terms).mean()
        q = torch.sigmoid(za + (ha * w).sum(-1)).clamp(1e-6, 1 - 1e-6)
        kl = (q * torch.log(q / pa.clamp(1e-6, 1 - 1e-6)) + (1 - q) * torch.log((1 - q) / (1 - pa).clamp(1e-6, 1 - 1e-6))).mean()
        loss = 0.5 * l_auc + 0.5 * kl + LAMBDA * (w ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return w.detach().numpy()


def main():
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")
    st = pd.read_csv(f"{T1}/task1_studies.csv", dtype={"StudyInstanceUID": str})
    legs = {"S6": load_leg("s6"), "S6F": load_leg("s6fall")}
    gid = [u for u in gold.index if all(u in L for L in legs.values())]
    aid = [u for u in st[st["set"] == "oof_300"].StudyInstanceUID if all(u in L for L in legs.values())]
    print(f"gold {len(gid)} | anchors {len(aid)}")
    Y = gold.loc[gid, LAB].values.astype(int)
    rng = np.random.default_rng(0)
    order = np.argsort(Y.sum(1) + rng.random(len(gid)) * 0.1)        # stratify by the number of positive findings
    fold = np.empty(len(gid), int)
    fold[order] = np.arange(len(gid)) % 5
    oof = {k: np.zeros((len(gid), 12)) for k in legs}
    frozen = {k: np.stack([legs[k][u][0] for u in gid]) for k in legs}
    for k, L in legs.items():
        zg = frozen[k]; hg = np.stack([L[u][1] for u in gid])
        za = np.stack([L[u][0] for u in aid]); ha = np.stack([L[u][1] for u in aid])
        for f in range(5):
            tr, te = fold != f, fold == f
            w = train_adapter(zg[tr], hg[tr], Y[tr], za, ha)
            oof[k][te] = zg[te] + (hg[te] * w).sum(-1)
        w_all = train_adapter(zg, hg, Y, za, ha)
        za_ad = za + (ha * w_all).sum(-1)
        rho = np.mean([spearmanr(za[:, j], za_ad[:, j]).correlation for j in range(12)])
        np.save(f"{T1}/adapter_{k}_w.npy", w_all)
        print(f"{k}: anchor Spearman adapted vs frozen (full-gold adapter) {rho:.4f} | |w| {np.abs(w_all).max():.4f}")
        legs[k] = (L, rho)
    base = pct_cols(frozen["S6"]) + pct_cols(frozen["S6F"])
    adapt = pct_cols(oof["S6"]) + pct_cols(oof["S6F"])
    d = macro(Y, adapt) - macro(Y, base)
    B = [rng.integers(0, len(gid), len(gid)) for _ in range(10000)]
    boot = np.array([macro(Y, adapt, b) - macro(Y, base, b) for b in B])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    rho_ok = all(legs[k][1] >= 0.97 for k in legs)
    adopt = d >= 0.004 and lo > 0 and rho_ok
    res = {"gold_n": len(gid), "anchors_n": len(aid), "frozen_S6+S6F_macro": round(macro(Y, base), 4),
           "adapted_oof_macro": round(macro(Y, adapt), 4), "delta": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "anchor_spearman": {k: round(float(legs[k][1]), 4) for k in legs}, "adopt": bool(adopt),
           "per_finding_delta": dict(zip(LAB, [round(auc(Y[:, j], adapt[:, j]) - auc(Y[:, j], base[:, j]), 4) for j in range(12)]))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{T1}/task3_adapter.json", "w"), indent=1)


if __name__ == "__main__":
    main()
