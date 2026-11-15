"""Collect the aggregate numbers behind the report's figures and tables.

Reads the project's own outputs (run folders, Kaggle training logs, label files, EDA tables) and writes only
aggregate tables (counts, rates, AUCs, confidence intervals) to ../data/. No per-study rows, report text or
images are written, so the report folder carries no competition data.

Run with the project's Python environment:
    python collect_results.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path("D:/Coding/kaggle/RSNA")
WORK = Path("D:/rsna_work")
RUNS = REPO / "output/imaging_v5/runs"
EDA = REPO / "reports/eda"
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
RNG = np.random.default_rng(2026)
N_BOOT = 2000


def auc(y, s):
    """Exact ROC-AUC by the rank-sum (Mann-Whitney) formula, ties counted as 1/2; equal to sklearn's roc_auc_score
    (checked below) but fast enough for thousands of bootstrap resamples."""
    r = rankdata(s)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def macro_auc(y, p):
    vals = [auc(y[:, k], p[:, k]) for k in range(y.shape[1]) if 0 < y[:, k].sum() < len(y)]
    return float(np.mean(vals))


def per_class_auc(y, p):
    return [auc(y[:, k], p[:, k]) if 0 < y[:, k].sum() < len(y) else np.nan for k in range(y.shape[1])]


_y, _s = np.random.default_rng(0).integers(0, 2, 200), np.round(np.random.default_rng(1).random(200), 2)
assert abs(auc(_y, _s) - roc_auc_score(_y, _s)) < 1e-12, "rank AUC must equal sklearn's"


def boot_ci(y, p, n=N_BOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        try:
            vals.append(macro_auc(y[s], p[s]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta(y, p_new, p_ref, n=N_BOOT):
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        s = RNG.choice(idx, len(idx), replace=True)
        try:
            d.append(macro_auc(y[s], p_new[s]) - macro_auc(y[s], p_ref[s]))
        except ValueError:
            continue
    return macro_auc(y, p_new) - macro_auc(y, p_ref), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---------------------------------------------------------------- gold labels (58 studies, 0/1)
train = pd.read_csv(REPO / "data/Raw_dicom/train.csv", usecols=["StudyInstanceUID", *LABELS])
gold = train.dropna(subset=LABELS).set_index("StudyInstanceUID")[LABELS].astype(int)
GOLD_UIDS = list(gold.index)
Y_GOLD = gold.to_numpy()
print("gold studies", len(gold), "positives per finding", dict(zip(LABELS, Y_GOLD.sum(0))))

# ---------------------------------------------------------------- 1. dataset composition
inv = pd.read_csv(EDA / "study_inventory.csv")
lang = inv.groupby("lang_final").agg(studies=("StudyInstanceUID", "size"), gold=("is_gold", "sum"),
                                     median_chars=("n_chars", "median")).sort_values("studies", ascending=False)
lang.to_csv(OUT / "dataset_language.csv")


def vendor(name):
    s = str(name).lower()
    for key, v in (("siemens", "Siemens"), ("philips", "Philips"), ("ge ", "GE"), ("gehc", "GE"), ("general electric", "GE"),
                   ("toshiba", "Canon/Toshiba"), ("canon", "Canon/Toshiba"), ("fuji", "Fujifilm"), ("hitachi", "Hitachi")):
        if key in s + " ":
            return v
    return "Other"


inv["vendor"] = inv.manufacturer.map(vendor)
inv.groupby("vendor").size().sort_values(ascending=False).rename("studies").to_csv(OUT / "dataset_vendor.csv")
inv.groupby("field_T").size().rename("studies").to_csv(OUT / "dataset_field.csv")
summary = {
    "studies": int(len(inv)), "gold": int(inv.is_gold.sum()), "series": int(inv.n_series.sum()),
    "slices": int(inv.n_slices.sum()), "bytes_GB": float(inv.bytes.sum() / 1e9),
    "series_per_study_median": float(inv.n_series.median()), "series_per_study_min": int(inv.n_series.min()),
    "series_per_study_max": int(inv.n_series.max()), "slices_per_study_median": float(inv.n_slices.median()),
    "no_fs_sag": int((~inv.has_fluid_sag).sum()), "no_fs_cor": int((~inv.has_fluid_cor).sum()),
    "no_fs_ax": int((~inv.has_fluid_ax).sum()), "laterality_tag_present": int(inv.laterality_tag.notna().sum()),
}
ser = pd.read_csv(EDA / "metadata/series_table.csv",
                  usecols=["Anatomical_Plane", "Fluid_Sensitive", "n_files", "fov_mm", "SliceThickness", "PixelSpacing_row"])
q = [0.05, 0.25, 0.5, 0.75, 0.95, 1.0]
rows = []
for plane, g in ser.groupby("Anatomical_Plane"):
    for col in ("n_files", "fov_mm", "SliceThickness", "PixelSpacing_row"):
        qs = g[col].quantile(q).to_numpy()
        rows.append([plane, col, len(g), *qs])
pd.DataFrame(rows, columns=["plane", "quantity", "n_series", "q05", "q25", "q50", "q75", "q95", "max"]).to_csv(
    OUT / "series_geometry_quantiles.csv", index=False)
# histogram counts of slices per series by plane (aggregate bins only)
bins = np.array([0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 120, 200, 400])
hist = {plane: np.histogram(g.n_files.clip(upper=399), bins)[0] for plane, g in ser.groupby("Anatomical_Plane")}
pd.DataFrame(hist, index=[f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])]).to_csv(OUT / "slices_per_series_hist.csv")
json.dump(summary, open(OUT / "dataset_summary.json", "w"), indent=1)
print("dataset", summary)

# ---------------------------------------------------------------- 2. label prevalence by source
prev = pd.read_csv(EDA / "reports_labels/prevalence_gold_vs_reports.csv", index_col=0)
v3 = pd.read_csv(REPO / "data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
v3 = v3.loc[~v3.index.isin(GOLD_UIDS), LABELS]
l7 = pd.read_csv(WORK / "ext_labels/songqizhou/8_final_label.csv").set_index("StudyInstanceUID")[LABELS]
l1 = pd.read_csv(WORK / "ext_labels/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID")[LABELS]
l1 = l1.loc[~l1.index.isin(GOLD_UIDS)]
fused = pd.read_csv(WORK / "ext_labels/fused_mean_7th_1st.csv").set_index("StudyInstanceUID")[LABELS]
p = pd.DataFrame({
    "gold (58, image read)": Y_GOLD.mean(0),
    "public report labels (P1)": prev["gpt56sol_pos_rate_unlabeled"].reindex(LABELS).to_numpy(),
    "our report-derived labels": (v3 >= 0.5).mean().to_numpy(),
    "7th-place image-read labels": (l7 >= 0.5).mean().to_numpy(),
    "1st-place image-read labels": (l1 >= 0.5).mean().to_numpy(),
}, index=LABELS)
p.to_csv(OUT / "prevalence_by_source.csv")
print(p.round(3))
# disagreement between the two image-read sets (share of non-gold studies where the 0.5 calls differ)
common = l7.index.intersection(l1.index)
dis = ((l7.loc[common] >= 0.5) != (l1.loc[common] >= 0.5)).mean()
dis.rename("disagreement_7th_vs_1st").to_csv(OUT / "imageread_disagreement.csv")

# ---------------------------------------------------------------- 3. label-source agreement with gold
ms = pd.read_csv(EDA / "label_validation/macro_summary.csv")
oa = pd.read_csv(EDA / "label_validation/order_avg_macro.csv")
rows = []
name_map = {"qwen14b_old": "Qwen2.5-14B-AWQ (first pipeline)", "qwen7b_old": "Qwen2.5-7B (first pipeline)",
            "rules_v1.1": "Rule extractor (9 languages)", "gpt56sol": "Public report labels P1",
            "llm_s0": "Our labeller (Qwen3-4B), one order", "report_v2": "Public report labels P2",
            "ens_gpt56+v2 (public only)": "Mean of P1 and P2"}
for _, r in ms.iterrows():
    if r.source in name_map:
        rows.append([name_map[r.source], "report", r.macro_auc, r.lo, r.hi])
r = oa.set_index("source")
rows.append(["Our labeller, two orders averaged", "report", *r.loc["llm_order_avg", ["macro_auc", "lo", "hi"]]])
rows.append(["Training labels (ours + P1 + P2)", "report", *r.loc["ens(llm_order_avg+gpt56+v2)", ["macro_auc", "lo", "hi"]]])
l7g = pd.read_csv(WORK / "ext_labels/songqizhou/9_final_label_with_gold58_studies.csv").set_index("StudyInstanceUID").loc[GOLD_UIDS, LABELS].to_numpy()
l1g = pd.read_csv(WORK / "ext_labels/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID").loc[GOLD_UIDS, LABELS].to_numpy()
for nm, pr in (("7th-place image-read labels", l7g), ("1st-place image-read labels (in-sample)", l1g)):
    rows.append([nm, "image", macro_auc(Y_GOLD, pr), *boot_ci(Y_GOLD, pr)])
lab = pd.DataFrame(rows, columns=["source", "kind", "macro_auc", "lo", "hi"]).sort_values("macro_auc")
lab.to_csv(OUT / "label_sources_gold_auc.csv", index=False)
print(lab.round(3))

# ---------------------------------------------------------------- 4. training curves
runs = {"E1": "E1_cnxtT_s3_k8_attn12", "E1 seed 2": "E1s2_cnxtT_s3_k8_attn12_seed2", "E1p (position fix)": "E1p_cnxtT_s3_k8_attn12_posfix",
        "E2a mean pooling": "E2a_cnxtT_s3_k8_mean", "E6 EfficientNetV2-S": "E6_effv2s_s3_k8_attn12",
        "E1L 12 epochs": "E1L_cnxtT_s3_k8_attn12_ep12", "E0 frozen features": "E0_frozen_effb0_v2head"}
cur = []
for nm, d in runs.items():
    h = pd.read_csv(RUNS / d / "history.csv")
    if "seed" in h.columns:  # the frozen-feature baseline E0 logs 5 seeds x 12 epochs: mean over seeds per epoch
        h = h.groupby("epoch", as_index=False)[["val_macro_auc"]].mean()
    for _, r in h.iterrows():
        cur.append([nm, int(r.epoch) + 1, r.val_macro_auc, r.get("train_loss", np.nan)])
pd.DataFrame(cur, columns=["run", "epoch", "pseudo_val_macro_auc", "train_loss"]).to_csv(OUT / "curves_mil.csv", index=False)

bp_runs = {"A0 (1st-place labels, fold 0)": "knee-bp-a0-f0", "A1 (7th-place labels, fold 0)": "knee-bp-a1-f0",
           "A2 (fused labels, fold 0)": "knee-bp-a2-f0", "C1 (fused, fold 1)": "knee-bp-c-f1",
           "C2 (fused, fold 2)": "knee-bp-c-f2", "C3 (fused, fold 3)": "knee-bp-c-f3"}
pat = re.compile(r"^epoch (\d+): agent labels soft (\S+) hard (\S+) \((\d+) studies\) \| gold 58 (\S+) .*training (\d+) s, validation (\d+) s")
cur = []
for nm, d in bp_runs.items():
    recs = json.load(open(WORK / f"kaggle_runs/{d}/out/{d}.log", encoding="utf-8"))
    seen = set()
    for rec in recs:
        for line in rec.get("data", "").splitlines():
            m = pat.match(line)
            if m and line not in seen:
                seen.add(line)
                e, soft, hard, n, g, tr, va = m.groups()
                cur.append([nm, int(e), float(g), float(hard), float(soft), int(tr), int(va)])
bp = pd.DataFrame(cur, columns=["run", "epoch", "gold58_macro_auc", "fold_hard_auc", "fold_soft_auc", "train_s", "valid_s"])
bp.to_csv(OUT / "curves_boxpool.csv", index=False)
print(bp.groupby("run").agg(last_gold=("gold58_macro_auc", "last"), hours=("train_s", lambda s: s.sum() / 3600)))

# ---------------------------------------------------------------- 5. ablations (project's paired bootstrap table)
pd.read_csv(EDA / "imaging/experiment_paired_vs_E1_cnxtT_s3_k8_attn12.csv").to_csv(OUT / "ablation_paired_vs_E1.csv", index=False)
pd.read_csv(EDA / "imaging/attention_E1_cnxtT_s3_k8_attn12.csv").to_csv(OUT / "attention_plane_share.csv", index=False)


# ---------------------------------------------------------------- 6. gold-58 per finding
def sealed(run):
    z = np.load(RUNS / run / "gold_preds_SEALED.npz", allow_pickle=True)
    return pd.DataFrame(z["p"], index=list(z["uids"]), columns=LABELS).loc[GOLD_UIDS].to_numpy()


def bp_gold(d, fold):
    f = pd.read_csv(WORK / f"kaggle_runs/{d}/out/fold{fold}_epoch08.csv", index_col=0)
    return f.loc[GOLD_UIDS, LABELS].to_numpy()


P = {
    "E0 frozen EfficientNet-B0": sealed("E0_frozen_effb0_v2head"),
    "2.5D MIL ensemble (F1-F3)": np.mean([sealed(r) for r in ("F1_cnxtT_s3_k8_attn12_pf_all_s42", "F2_cnxtT_s3_k8_attn12_pf_all_s7",
                                                              "F3_cnxtT_s3_k8_attn12_pf_all_s11")], 0),
    "Box-pooling, six-model mean": np.mean([bp_gold("knee-bp-a0-f0", 0), bp_gold("knee-bp-a1-f0", 0), bp_gold("knee-bp-a2-f0", 0),
                                             bp_gold("knee-bp-c-f1", 1), bp_gold("knee-bp-c-f2", 2), bp_gold("knee-bp-c-f3", 3)], 0),
}
rows = []
for nm, pr in P.items():
    rows.append([nm, "macro", macro_auc(Y_GOLD, pr), *boot_ci(Y_GOLD, pr)])
    for k, a in zip(LABELS, per_class_auc(Y_GOLD, pr)):
        rows.append([nm, k, a, np.nan, np.nan])
gp = pd.DataFrame(rows, columns=["model", "finding", "auc", "lo", "hi"])
gp.to_csv(OUT / "gold_per_finding.csv", index=False)
print(gp[gp.finding == "macro"].round(4))
pd.Series(Y_GOLD.sum(0), index=LABELS, name="gold_positives").to_csv(OUT / "gold_positives.csv")

# ---------------------------------------------------------------- 7. label-swap ablation on gold-58
base = sealed("E1p_cnxtT_s3_k8_attn12_posfix")
ours = {"Our report-derived labels": base, "1st-place labels": sealed("LS1_cnxtT_s3_k8_attn12_pf_lab1st"),
        "7th-place labels": sealed("LS7_cnxtT_s3_k8_attn12_pf_lab7th"), "Fused 1st + 7th": sealed("LSavg_cnxtT_s3_k8_attn12_pf_labfused")}
bpm = {"1st-place labels": bp_gold("knee-bp-a0-f0", 0), "7th-place labels": bp_gold("knee-bp-a1-f0", 0),
       "Fused 1st + 7th": bp_gold("knee-bp-a2-f0", 0)}
rows = []
for nm, pr in ours.items():
    d = paired_delta(Y_GOLD, pr, base) if nm != "Our report-derived labels" else (0.0, np.nan, np.nan)
    rows.append(["2.5D MIL (our model, 224 px)", nm, macro_auc(Y_GOLD, pr), *boot_ci(Y_GOLD, pr), *d])
for nm, pr in bpm.items():
    d = paired_delta(Y_GOLD, pr, bpm["1st-place labels"]) if nm != "1st-place labels" else (0.0, np.nan, np.nan)
    rows.append(["Box-pooling ConvNeXt-T (256 px)", nm, macro_auc(Y_GOLD, pr), *boot_ci(Y_GOLD, pr), *d])
ls = pd.DataFrame(rows, columns=["model", "labels", "gold_auc", "lo", "hi", "delta_vs_ref", "d_lo", "d_hi"])
ls.to_csv(OUT / "label_swap_gold.csv", index=False)
print(ls.round(4))
# high-resolution check of our model (same labels, fold, seed)
hr = sealed("LS7hr_cnxtT_s3_k6_attn12_c320_pf_lab7th")
d = paired_delta(Y_GOLD, hr, ours["7th-place labels"])
pd.DataFrame([["LS7 224 px", macro_auc(Y_GOLD, ours["7th-place labels"])], ["LS7hr 320 crop", macro_auc(Y_GOLD, hr)],
              ["delta", d[0]], ["delta_lo", d[1]], ["delta_hi", d[2]]], columns=["item", "value"]).to_csv(OUT / "resolution_check_gold.csv", index=False)

# ---------------------------------------------------------------- 8. attention sharpening sweep (held-out fold 0)
sh = pd.read_csv(WORK / "sharp_f0/series_sharp.csv")
study = sh.groupby(["model", "sharp", "StudyInstanceUID"])[LABELS].mean().reset_index()
fold_uids = sorted(set(study.StudyInstanceUID) - set(GOLD_UIDS))
refs = {"1st-place labels": pd.read_csv(WORK / "ext_labels/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID").loc[fold_uids, LABELS],
        "7th-place labels": pd.read_csv(WORK / "ext_labels/songqizhou/8_final_label.csv").set_index("StudyInstanceUID").loc[fold_uids, LABELS]}
rows, prows = [], []
for model in ("a0", "a1"):
    pm = {s: g.set_index("StudyInstanceUID").loc[fold_uids, LABELS].to_numpy() for s, g in study[study.model == model].groupby("sharp")}
    for rname, ref in refs.items():
        y = (ref.to_numpy() >= 0.5).astype(int)
        for s, pr in pm.items():
            d = paired_delta(y, pr, pm[1.0], n=1000) if s != 1.0 else (0.0, 0.0, 0.0)
            rows.append([model.upper(), rname, s, macro_auc(y, pr), *d])
            for k, (a, a1) in enumerate(zip(per_class_auc(y, pr), per_class_auc(y, pm[1.0]))):
                prows.append([model.upper(), rname, s, LABELS[k], a - a1])
pd.DataFrame(rows, columns=["model", "reference", "s", "fold_macro_auc", "delta", "d_lo", "d_hi"]).to_csv(OUT / "sharpening_sweep.csv", index=False)
pd.DataFrame(prows, columns=["model", "reference", "s", "finding", "delta"]).to_csv(OUT / "sharpening_per_finding.csv", index=False)
print(pd.read_csv(OUT / "sharpening_sweep.csv").round(4))
print("held-out fold studies:", len(fold_uids))

print("done ->", OUT)
