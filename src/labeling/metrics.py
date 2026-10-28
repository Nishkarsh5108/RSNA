"""Evaluation utilities shared by the label audit, the new label pipeline and the
downstream experiments.

All functions take plain numpy arrays. ROC-AUC is reported as NaN (never imputed)
when the evaluation subset contains a single class. Confidence intervals come
from a study-level bootstrap; for macro metrics the same resampled studies are
used for every class, so the interval reflects the joint uncertainty.
"""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


def roc_auc_score(y, s):
    """Mann-Whitney AUC with average ranks for ties (same value as sklearn)."""
    y = np.asarray(y)
    r = rankdata(s)
    n_pos = int((y == 1).sum())
    n_neg = len(y) - n_pos
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def safe_auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def ece(y, p, n_bins=5):
    """Expected calibration error with equal-width bins (few bins: n is small)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def binary_report(y, s, threshold=0.5, n_boot=2000, seed=0):
    """Metrics for one target. `s` may be a probability or any monotone score;
    calibration metrics are only meaningful when it is a probability in [0,1]."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    m = ~(np.isnan(y) | np.isnan(s))
    y, s = y[m], s[m]
    n = len(y)
    pos = int(y.sum())
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (tp and not np.isnan(prec) and not np.isnan(rec)) else (0.0 if tp + fp + fn else np.nan)
    auc = safe_auc(y, s)
    lo = hi = np.nan
    if not np.isnan(auc) and n_boot:
        rng = np.random.default_rng(seed)
        pos_i, neg_i = np.where(y == 1)[0], np.where(y == 0)[0]
        boots = []
        for _ in range(n_boot):  # stratified: keep class counts fixed
            bi = np.concatenate([rng.choice(pos_i, len(pos_i)), rng.choice(neg_i, len(neg_i))])
            boots.append(roc_auc_score(y[bi], s[bi]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
    is_prob = bool(s.min() >= 0 and s.max() <= 1)
    pc = np.clip(s, 1e-4, 1 - 1e-4)
    return {
        "n": n, "n_pos": pos, "n_neg": n - pos,
        "auc": auc, "auc_lo": lo, "auc_hi": hi,
        "ap": float(average_precision_score(y, s)) if 0 < pos < n else np.nan,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
        "balanced_acc": np.nanmean([rec, spec]) if not (np.isnan(rec) and np.isnan(spec)) else np.nan,
        "accuracy": (tp + tn) / n if n else np.nan,
        "brier": float(np.mean((s - y) ** 2)) if is_prob else np.nan,
        "log_loss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))) if is_prob else np.nan,
        "ece5": ece(y, s) if is_prob else np.nan,
        "mean_score_pos": float(s[y == 1].mean()) if pos else np.nan,
        "mean_score_neg": float(s[y == 0].mean()) if n - pos else np.nan,
    }


def macro_auc_ci(Y, S, n_boot=2000, seed=0):
    """Macro AUC over columns with both classes, with a joint study bootstrap CI.

    Y, S: (n_studies, n_targets) arrays; NaN in S marks a missing score and is
    excluded per column.
    """
    Y = np.asarray(Y, dtype=float)
    S = np.asarray(S, dtype=float)
    point = np.nanmean([safe_auc(Y[:, j], S[:, j]) for j in range(Y.shape[1])])
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    boots = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        vals = [safe_auc(Y[bi, j], S[bi, j]) for j in range(Y.shape[1])]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_macro_auc_diff(Y, S1, S2, n_boot=2000, seed=0):
    """Bootstrap distribution of macro-AUC(S1) - macro-AUC(S2) on the same studies.
    Returns (point difference, 2.5%, 97.5%, fraction of bootstrap draws <= 0)."""
    Y = np.asarray(Y, dtype=float)
    def mac(S, idx):
        v = [safe_auc(Y[idx, j], S[idx, j]) for j in range(Y.shape[1])]
        v = [x for x in v if not np.isnan(x)]
        return np.mean(v) if v else np.nan
    n = Y.shape[0]
    full = np.arange(n)
    point = mac(S1, full) - mac(S2, full)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        bi = rng.integers(0, n, n)
        d = mac(S1, bi) - mac(S2, bi)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    return float(point), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float((diffs <= 0).mean())
