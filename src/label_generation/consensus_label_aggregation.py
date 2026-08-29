"""
Consensus label aggregation across independent pseudo-label sources.

You now have 5 usable, distinct sources for the 12 knee pathologies (a 6th,
labels_llm_gpt56sol.csv, is dropped here because it is byte-identical to
report_labels_gpt56sol.csv on every row checked -- same source, would silently
double its vote if included twice):

  gpt56sol       -> report_labels_gpt56sol.csv   (LLM verdict + confidence, GENUINE
                                                    on the 58 GT rows: ~80% cell match,
                                                    not a copy of ground truth)
  report_v2      -> report_labels_v2.csv          (rule-based verdict/confidence
                                                    extractor, continuous scores,
                                                    also GENUINE on the 58 GT rows)
  qwen14b_awq    -> train_folds_qwen14b_awq_v2.csv (LEAKED on the 58 GT rows: those
  qwen7_5b       -> train_folds_sota_qwen25.csv     58 rows are ground truth copied in
  public_pseudo  -> train_folds_with_pseudo.csv     via combine_first, not real votes)

Why not just average all 5: naive ensembling treats every source as equally
trustworthy and independent of the 58 GT rows, which wastes the one thing you
actually know for sure (which sources are right how often) and can't tell you
which rows are genuinely uncertain vs confidently agreed on.

Primary label = a 50/50 blend of cleanlab's CROWDLAB and the validated-AUC-weighted
average, both run on the full 5-source annotator matrix. Why a blend rather than
picking one outright: with only 2 genuine (non-leaked) sources to test against,
CROWDLAB clearly won the honest comparison (7/12 pathologies). After qwen14b_awq
and qwen7_5b were upgraded from leaked to genuine (see
src/label_generation/regenerate_gt58_qwen14b.py /
regenerate_gt58_qwen7b.py / patch_gt58_genuine_predictions.py), giving 4 genuine
sources instead of 2, the race tightened to a near-even split: CROWDLAB wins 5/12,
the plain mean wins 4/12 (one exact tie), and -- notably -- Dawid-Skene EM now wins
2/12, where it won zero with only 2 raters. That's a direct confirmation that EM's
earlier failure was partly a too-few-raters problem, not purely the
correlated-source problem (see point 1 below, which is still real: the leaked
public_pseudo source's Dawid-Skene sensitivity is STILL a suspiciously flat 0.98
across all 12 classes, unaffected by the other two sources being fixed). With no
single method clearly dominant on this data, blending CROWDLAB and the weighted
mean is the more statistically prudent default than hard-committing to whichever
one happens to win the noisy 58-row race in either direction.

Dawid-Skene (Dawid & Skene, 1979) and a logistic-regression stacker are still
computed and kept as `{col}__dawid_skene` and `{col}__stacker` for comparison/QA,
not for direct use. `{col}__weighted_mean` and `{col}__crowdlab` are the two inputs
to the primary blend, also kept explicit. Two reasons Dawid-Skene still underperforms
overall despite now winning 2/12 classes:

1. Dawid-Skene assumes sources make INDEPENDENT errors. qwen14b_awq/qwen7_5b are
   the same model family, and public_pseudo may itself be LLM-derived -- if they're
   correlated, EM rewards a source for agreeing with a consensus it helped build,
   which inflates its estimated reliability regardless of true accuracy. The
   smoking gun: public_pseudo's estimated sensitivity comes out at a suspiciously
   flat 0.98 across every single pathology (real sources vary a lot by class, e.g.
   gpt56sol ranges 0.66-0.98) -- despite docs/label_generation_walkthrough.md
   documenting ~60% real-world accuracy for that source.
2. Even with 4 genuine sources now, that's still a small number of independent
   raters for EM to estimate per-source confusion matrices from reliably.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from cleanlab.multiannotator import get_majority_vote_label, get_label_quality_multiannotator

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
DATA_DIR = r"D:\Coding\kaggle\RSNA\data\main_data\to_use"
GT_PATH = os.path.join(DATA_DIR, "available_labels.csv")
OUT_PATH = os.path.join(DATA_DIR, "train_folds_consensus.csv")

TARGET_COLS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

SOURCES = {
    "gpt56sol":      {"path": os.path.join(DATA_DIR, "report_labels_gpt56sol.csv"), "col_prefix": ""},
    "report_v2":     {"path": os.path.join(DATA_DIR, "report_labels_v2.csv"),        "col_prefix": ""},
    "qwen14b_awq":   {"path": os.path.join(DATA_DIR, "train_folds_qwen14b_awq_v2.csv"), "col_prefix": ""},
    "qwen7_5b":      {"path": os.path.join(DATA_DIR, "train_folds_sota_qwen25.csv"),    "col_prefix": ""},
    "public_pseudo": {"path": os.path.join(DATA_DIR, "train_folds_with_pseudo.csv"),    "col_prefix": "pseudo_"},
}

LEAKAGE_TOLERANCE = 1e-6
VOTE_THRESHOLD = 0.5     # continuous score -> hard vote, for the Dawid-Skene step
EM_ITERS = 30
RELIABILITY_CLIP = (0.02, 0.98)  # keep sensitivity/specificity away from 0/1 to avoid log(0)
N_CALIBRATION_FOLDS = 5

# Fixed trust weights for LEAKED sources, used in the final weighted-average consensus.
# Their reliability can't be measured from the 58 GT rows here (those rows are ground
# truth copied in, not real predictions), and Dawid-Skene's self-estimate for them isn't
# trustworthy either (see module docstring). Plug in real historical numbers if you have
# them -- e.g. qwen7_5b was separately benchmarked at ~78.7% macro accuracy / 0.80 macro
# AUC in an earlier GT validation run (docs/multilingual_llm_medical_classification_chat.md).
# public_pseudo is kept low: docs/label_generation_walkthrough.md documents ~60% accuracy.
LEAKED_SOURCE_TRUST = {
    "qwen14b_awq": 0.35,
    "qwen7_5b": 0.35,
    "public_pseudo": 0.05,
}
MIN_GENUINE_SOURCE_WEIGHT = 0.05  # floor so a source never gets literally zero say


# -------------------------------------------------------------
# LOAD
# -------------------------------------------------------------
def load_source(spec):
    df = pd.read_csv(spec["path"])
    prefix = spec["col_prefix"]
    src_cols = [f"{prefix}{col}" for col in TARGET_COLS]
    missing = [c for c in src_cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns {missing} in {spec['path']}")
    df = df.set_index("StudyInstanceUID")
    # Select the exact prefixed columns first, then rename positionally -- avoids collisions
    # when a file (e.g. public_pseudo) also carries a bare-named column like "ACL" alongside
    # "pseudo_ACL"; a dict-based rename() would leave two same-named "ACL" columns behind.
    sub = df[src_cols].copy()
    sub.columns = TARGET_COLS
    return sub.astype(float)


print("=" * 100)
print(" LOADING GROUND TRUTH + SOURCES")
print("=" * 100)

gt_df = pd.read_csv(GT_PATH).set_index("StudyInstanceUID")[TARGET_COLS].astype(float)
print(f"Ground truth: {len(gt_df)} studies")

base_meta = pd.read_csv(SOURCES["qwen14b_awq"]["path"])[["StudyInstanceUID", "Report"]].set_index("StudyInstanceUID")
universe = base_meta.index

source_dfs = {}
for name, spec in SOURCES.items():
    df = load_source(spec).reindex(universe)  # NaN where a source doesn't cover a row
    source_dfs[name] = df
    n_missing = df[TARGET_COLS[0]].isna().sum()
    print(f"[{name}] {len(df) - n_missing}/{len(df)} studies covered"
          f"{'' if n_missing == 0 else f' ({n_missing} missing)'}")


# -------------------------------------------------------------
# LEAKAGE DETECTION on the 58 GT rows
# -------------------------------------------------------------
print("\n" + "=" * 100)
print(" LEAKAGE CHECK")
print("=" * 100)

leaked = {}
for name, df in source_dfs.items():
    common = gt_df.index.intersection(df.dropna().index)
    if len(common) == 0:
        leaked[name] = False
        print(f"[{name}] no overlap with GT rows to check.")
        continue
    diff = (gt_df.loc[common] - df.loc[common]).abs().max(axis=1)
    identical_rate = (diff < LEAKAGE_TOLERANCE).mean()
    leaked[name] = identical_rate > 0.98
    tag = "LEAKED (ground truth copied in, votes on these rows are not real)" if leaked[name] else "genuine"
    print(f"[{name}] {identical_rate*100:.1f}% identical to GT on {len(common)} rows -> {tag}")


# -------------------------------------------------------------
# DAWID-SKENE STYLE EM, per class
# -------------------------------------------------------------
def fit_dawid_skene(votes, known_mask, true_labels, source_leaked, prior_pos, n_iters=EM_ITERS):
    """
    votes: dict[source_name] -> np.array of {0.0, 1.0, np.nan}, aligned to `universe` order.
    known_mask: bool array, True for the 58 GT studies.
    true_labels: float array, true 0/1 label for known rows (ignored elsewhere).
    source_leaked: dict[source_name] -> bool, whether that source's known-row votes are fake.
    prior_pos: P(true=1) prior, estimated from GT prevalence.
    Returns: q (posterior P(true=1) per row, with known rows fixed to true_labels),
             reliability (dict[source] -> (sensitivity, specificity)).
    """
    n = len(known_mask)
    q = np.where(known_mask, true_labels, np.nanmean([v for v in votes.values()], axis=0))
    q = np.nan_to_num(q, nan=prior_pos)

    names = list(votes.keys())
    reliability = {name: (0.8, 0.8) for name in names}  # init

    for _ in range(n_iters):
        # M-step: estimate sensitivity/specificity per source from rows where it has a vote,
        # excluding known rows for sources whose known-row votes are leaked (uninformative).
        for name in names:
            v = votes[name]
            has_vote = ~np.isnan(v)
            usable = has_vote & (~known_mask if source_leaked[name] else np.ones(n, dtype=bool))
            if usable.sum() == 0:
                continue
            w_pos = q[usable]
            w_neg = 1.0 - w_pos
            denom_pos = w_pos.sum()
            denom_neg = w_neg.sum()
            sens = float((v[usable] * w_pos).sum() / denom_pos) if denom_pos > 1e-9 else 0.8
            spec = float(((1 - v[usable]) * w_neg).sum() / denom_neg) if denom_neg > 1e-9 else 0.8
            sens = float(np.clip(sens, *RELIABILITY_CLIP))
            spec = float(np.clip(spec, *RELIABILITY_CLIP))
            reliability[name] = (sens, spec)

        # E-step: update q for the unknown rows only (known rows stay fixed to true labels).
        log_odds = np.full(n, np.log(prior_pos / (1 - prior_pos)))
        for name in names:
            v = votes[name]
            sens, spec = reliability[name]
            has_vote = ~np.isnan(v)
            contrib = np.zeros(n)
            pos_vote = has_vote & (v == 1)
            neg_vote = has_vote & (v == 0)
            contrib[pos_vote] = np.log(sens / (1 - spec))
            contrib[neg_vote] = np.log((1 - sens) / spec)
            log_odds += contrib
        new_q_unknown = 1.0 / (1.0 + np.exp(-log_odds))
        q = np.where(known_mask, true_labels, new_q_unknown)

    return q, reliability


def run_crowdlab(votes_subset, continuous_subset, source_names):
    """
    votes_subset: dict[source_name] -> np.array of {0.0, 1.0, np.nan} (hard votes), aligned to
                  `universe` order -- this is what CROWDLAB treats as the annotator matrix.
    continuous_subset: dict[source_name] -> np.array of raw [0,1] scores (pre-threshold) --
                  used only as features for the auxiliary classifier below, so it sees more
                  nuance than the thresholded votes it's also being compared against.
    Returns: soft consensus probability per row (P(true=1), derived from CROWDLAB's
             hard consensus_label + consensus_quality_score), and the annotator_stats table.

    CROWDLAB needs pred_probs from a classifier with no missing values, so an auxiliary
    LogisticRegression is trained via cross_val_predict on each source's raw continuous score,
    using the (unsupervised) majority-vote label as its training target -- no ground truth
    required to run this. Its out-of-sample predictions then feed into CROWDLAB alongside the
    raw annotator votes.
    """
    ma_labels = pd.DataFrame({name: votes_subset[name] for name in source_names})
    majority_labels = get_majority_vote_label(ma_labels)

    X_features = np.column_stack([np.nan_to_num(continuous_subset[name], nan=0.5) for name in source_names])
    pred_probs = cross_val_predict(LogisticRegression(), X_features, majority_labels,
                                    cv=5, method="predict_proba")

    results = get_label_quality_multiannotator(ma_labels, pred_probs, verbose=False)
    consensus_label = results["label_quality"]["consensus_label"].to_numpy()
    consensus_quality = results["label_quality"]["consensus_quality_score"].to_numpy()
    soft = np.where(consensus_label == 1, consensus_quality, 1 - consensus_quality)
    return soft, results["annotator_stats"]


def cross_validated_stacker_auc(X, y, n_splits=N_CALIBRATION_FOLDS):
    """Honest (out-of-fold) AUC for a logistic-regression stack of source scores."""
    min_class = int(np.bincount(y.astype(int)).min())
    if min_class < 2:
        return np.nan, np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=min(n_splits, min_class), shuffle=True, random_state=42)
    oof = np.full(len(y), np.nan)
    for train_idx, val_idx in skf.split(X, y):
        if len(np.unique(y[train_idx])) < 2:
            oof[val_idx] = X[val_idx].mean(axis=1)
            continue
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        oof[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    auc = roc_auc_score(y, oof) if len(np.unique(y)) > 1 else np.nan
    return auc, oof


# -------------------------------------------------------------
# RUN PER CLASS
# -------------------------------------------------------------
print("\n" + "=" * 100)
print(" DAWID-SKENE CONSENSUS + STACKING (per pathology)")
print("=" * 100)

known_mask = np.asarray(universe.isin(gt_df.index))
genuine_sources = [name for name in SOURCES if not leaked[name]]

consensus_cols = {}       # primary: 50/50 blend of weighted_mean_cols + crowdlab_cols (see docstring)
weighted_mean_cols = {}   # input to the blend, also kept explicit as "{col}__weighted_mean"
dawid_skene_cols = {}     # secondary/diagnostic
stacker_cols = {}         # secondary/diagnostic
crowdlab_cols = {}        # input to the blend, also kept explicit as "{col}__crowdlab"
disagreement_cols = {}
reliability_report = []
weights_report = []
crowdlab_quality_report = []

for col in TARGET_COLS:
    true_labels_full = np.full(len(universe), np.nan)
    for uid, val in gt_df[col].items():
        if uid in universe:
            true_labels_full[universe.get_loc(uid)] = val
    true_labels_full = np.nan_to_num(true_labels_full, nan=0.0)  # only read where known_mask=True

    prior_pos = float(gt_df[col].mean())
    prior_pos = float(np.clip(prior_pos, 0.02, 0.98))

    votes = {}
    continuous = {}
    for name, df in source_dfs.items():
        raw = df[col].to_numpy()
        continuous[name] = raw
        votes[name] = np.where(np.isnan(raw), np.nan, (raw >= VOTE_THRESHOLD).astype(float))

    q, reliability = fit_dawid_skene(votes, known_mask, true_labels_full, leaked, prior_pos)
    dawid_skene_cols[col] = q

    for name, (sens, spec) in reliability.items():
        reliability_report.append({"pathology": col, "source": name, "sensitivity": sens, "specificity": spec})

    # Disagreement diagnostic: std dev of raw (continuous) scores across genuine + leaked sources,
    # computed for ALL rows (informative even where we can't validate against GT).
    all_scores = np.vstack([source_dfs[name][col].to_numpy() for name in SOURCES])
    disagreement_cols[col] = np.nanstd(all_scores, axis=0)

    # Stacker: logistic regression on genuine sources' raw scores, trained/validated on the 58 GT rows.
    # A missing source vote (e.g. report_v2's 1 uncovered study) is filled with a neutral 0.5 prior --
    # sklearn's LogisticRegression can't accept NaN natively.
    X_known = np.column_stack([source_dfs[name][col].to_numpy()[known_mask] for name in genuine_sources])
    X_known = np.nan_to_num(X_known, nan=0.5)
    y_known = true_labels_full[known_mask].astype(int)
    stacker_auc, oof_preds = cross_validated_stacker_auc(X_known, y_known)

    # Fit on all 58 for deployment on the unknown rows.
    final_clf = LogisticRegression()
    if len(np.unique(y_known)) > 1:
        final_clf.fit(X_known, y_known)
        X_all = np.column_stack([source_dfs[name][col].to_numpy() for name in genuine_sources])
        X_all = np.nan_to_num(X_all, nan=0.5)
        stacker_pred = final_clf.predict_proba(X_all)[:, 1]
    else:
        stacker_pred = np.full(len(universe), prior_pos)
    stacker_pred = np.where(known_mask, true_labels_full, stacker_pred)
    stacker_cols[col] = stacker_pred

    # CROWDLAB (cleanlab), full run across all 5 sources -- used for the {col}__crowdlab output column.
    crowdlab_soft_full, crowdlab_annotator_stats = run_crowdlab(votes, continuous, list(SOURCES.keys()))
    crowdlab_cols[col] = np.where(known_mask, true_labels_full, crowdlab_soft_full)
    for name in SOURCES:
        if name in crowdlab_annotator_stats.index:
            crowdlab_quality_report.append({
                "pathology": col, "source": name,
                "annotator_quality": crowdlab_annotator_stats.loc[name, "annotator_quality"],
            })

    # Honest EM quality check: re-run EM using ONLY genuine (non-leaked) sources, so its
    # score on the 58 GT rows isn't contaminated by sources that already know the answer there.
    blind_votes = {name: votes[name] for name in genuine_sources}
    blind_leaked = {name: False for name in genuine_sources}
    q_blind, _ = fit_dawid_skene(blind_votes, np.zeros(len(universe), dtype=bool),  # nothing "known" during this pass
                                  np.zeros(len(universe)), blind_leaked, prior_pos)
    em_blind_auc = roc_auc_score(y_known, q_blind[known_mask]) if len(np.unique(y_known)) > 1 else np.nan

    # Same idea for CROWDLAB: run it again using only genuine sources, so the 58 GT rows aren't
    # trivially "solved" by leaked sources whose votes there are just copied ground truth.
    blind_continuous = {name: continuous[name] for name in genuine_sources}
    crowdlab_soft_blind, _ = run_crowdlab(blind_votes, blind_continuous, genuine_sources)
    crowdlab_blind_auc = roc_auc_score(
        y_known, crowdlab_soft_blind[known_mask]
    ) if len(np.unique(y_known)) > 1 else np.nan

    mean_genuine_probs = np.nanmean(X_known, axis=1)
    mean_genuine_auc = roc_auc_score(
        y_known, mean_genuine_probs
    ) if len(np.unique(y_known)) > 1 else np.nan

    blend_blind_probs = 0.5 * mean_genuine_probs + 0.5 * crowdlab_soft_blind[known_mask]
    blend_blind_auc = roc_auc_score(
        y_known, blend_blind_probs
    ) if len(np.unique(y_known)) > 1 else np.nan

    print(f"\n{col}")
    print(f"  honest AUC on 58 GT rows (genuine sources only, not contaminated by leaked sources):")
    print(f"    simple mean of genuine sources : {mean_genuine_auc:.4f}" if not np.isnan(mean_genuine_auc) else "    simple mean: N/A (single class)")
    print(f"    EM fusion of genuine sources    : {em_blind_auc:.4f}" if not np.isnan(em_blind_auc) else "    EM fusion: N/A")
    print(f"    CROWDLAB fusion of genuine sources: {crowdlab_blind_auc:.4f}" if not np.isnan(crowdlab_blind_auc) else "    CROWDLAB fusion: N/A")
    print(f"    50/50 blend (mean + CROWDLAB)   : {blend_blind_auc:.4f}" if not np.isnan(blend_blind_auc) else "    blend: N/A")
    print(f"    logistic-regression stacker     : {stacker_auc:.4f}" if not np.isnan(stacker_auc) else "    stacker: N/A")

    source_aucs = {}
    for name in genuine_sources:
        s = source_dfs[name][col].to_numpy()[known_mask]
        valid = ~np.isnan(s)
        if valid.sum() > 0 and len(np.unique(y_known[valid])) > 1:
            auc = roc_auc_score(y_known[valid], s[valid])
            source_aucs[name] = auc
            print(f"    source '{name}' alone            : {auc:.4f}"
                  f"{'' if valid.all() else f'  ({(~valid).sum()} row(s) skipped, not covered by this source)'}")

    # Primary consensus: weighted average. Genuine sources are weighted by their own measured
    # AUC-above-chance (so a source with AUC~0.5 contributes almost nothing); leaked sources get
    # the fixed, conservative LEAKED_SOURCE_TRUST weight since we can't measure them honestly here.
    weights = {}
    for name in genuine_sources:
        skill = max(source_aucs.get(name, 0.5) - 0.5, 0.0)
        weights[name] = max(skill, MIN_GENUINE_SOURCE_WEIGHT)
    for name in SOURCES:
        if leaked[name]:
            weights[name] = LEAKED_SOURCE_TRUST.get(name, 0.05)
    weight_sum = sum(weights.values())
    for name, w in weights.items():
        weights_report.append({"pathology": col, "source": name, "weight": w / weight_sum})

    weighted_scores = np.zeros(len(universe))
    for name in SOURCES:
        raw = np.nan_to_num(source_dfs[name][col].to_numpy(), nan=0.5)
        weighted_scores += raw * (weights[name] / weight_sum)
    weighted_mean_cols[col] = np.where(known_mask, true_labels_full, weighted_scores)

    # Primary consensus = 50/50 blend of the weighted mean and CROWDLAB, both full 5-source runs.
    # Neither method is clearly dominant now that there are 4 genuine sources (see docstring) --
    # blending is the more statistically prudent default than fully committing to one.
    consensus_cols[col] = 0.5 * weighted_mean_cols[col] + 0.5 * crowdlab_cols[col]

print("\n" + "=" * 100)
print(" PER-SOURCE PER-CLASS RELIABILITY (from Dawid-Skene, diagnostic only -- see caveats above)")
print("=" * 100)
reliability_df = pd.DataFrame(reliability_report)
print(reliability_df.pivot(index="source", columns="pathology", values="sensitivity").round(2).to_string())
print("\n(specificity table omitted for brevity -- see reliability_df in-memory if running interactively)")

print("\n" + "=" * 100)
print(" PER-SOURCE PER-CLASS WEIGHTS USED IN THE WEIGHTED-AVERAGE CONSENSUS (diagnostic, '{col}__weighted_mean')")
print("=" * 100)
weights_df = pd.DataFrame(weights_report)
print(weights_df.pivot(index="source", columns="pathology", values="weight").round(3).to_string())

print("\n" + "=" * 100)
print(" CROWDLAB PER-SOURCE PER-CLASS ANNOTATOR QUALITY (from the full 5-source run, diagnostic only)")
print("=" * 100)
crowdlab_quality_df = pd.DataFrame(crowdlab_quality_report)
print(crowdlab_quality_df.pivot(index="source", columns="pathology", values="annotator_quality").round(2).to_string())


# -------------------------------------------------------------
# ASSEMBLE FINAL OUTPUT
# -------------------------------------------------------------
final_df = base_meta.copy()
for col in TARGET_COLS:
    final_df[col] = consensus_cols[col]
    final_df[f"{col}__dawid_skene"] = dawid_skene_cols[col]
    final_df[f"{col}__stacker"] = stacker_cols[col]
    final_df[f"{col}__crowdlab"] = crowdlab_cols[col]  # one of the two inputs blended into the primary column
    final_df[f"{col}__weighted_mean"] = weighted_mean_cols[col]
    final_df[f"{col}__disagreement"] = disagreement_cols[col]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
strat_target = (final_df[TARGET_COLS] >= 0.5).sum(axis=1)
final_df["fold"] = -1
for fold, (t_idx, v_idx) in enumerate(sgkf.split(final_df, strat_target, groups=final_df.index)):
    final_df.iloc[v_idx, final_df.columns.get_loc("fold")] = fold

final_df = final_df.reset_index()
final_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved consensus dataset to: {OUT_PATH}")
print("Primary label columns (e.g. 'ACL') are a 50/50 blend of '{col}__weighted_mean' and '{col}__crowdlab'")
print("(both full 5-source runs) -- neither method was clearly dominant once qwen14b_awq/qwen7_5b became")
print("genuine sources (see module docstring for the honest per-pathology comparison that motivated this).")
print("'{col}__dawid_skene', '{col}__stacker' are further diagnostic alternatives -- kept for QA/comparison,")
print("not blended into the primary.")
print("'{col}__disagreement' columns flag rows where sources disagree most -- consider down-weighting")
print("or manually reviewing the highest-disagreement rows before trusting them as soft labels.")
