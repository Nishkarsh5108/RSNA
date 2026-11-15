"""Draw the report's data figures from the aggregate tables in ../data (written by collect_results.py).

Only aggregate numbers are plotted. Output: ../figures/*.pdf (vector, white background, Times-like font).
    python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA, FIG = HERE.parent / "data", HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
          "Synovitis", "Baker's", "Contusion", "Fracture"]
# validated categorical slots (blue, orange, aqua, violet, red) + neutral grey; secondary encoding by hatch/marker
BLUE, ORANGE, AQUA, VIOLET, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#dddcd8", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "hatch.linewidth": 0.6, "pdf.fonttype": 42,
})
WIDTH = 6.3  # inches: the text block is about 16.25 cm wide


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", color=INK)


# ------------------------------------------------------------------ dataset composition
def fig_dataset():
    lang = pd.read_csv(DATA / "dataset_language.csv", index_col=0)
    ven = pd.read_csv(DATA / "dataset_vendor.csv", index_col=0)["studies"]
    fld = pd.read_csv(DATA / "dataset_field.csv", index_col=0)["studies"]
    hist = pd.read_csv(DATA / "slices_per_series_hist.csv", index_col=0)
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 4.9), gridspec_kw={"hspace": 0.55, "wspace": 0.45})
    ax = axs[0, 0]
    l = lang["studies"][::-1]
    ax.barh(l.index, l.values, color=BLUE, height=0.65)
    for y, v in enumerate(l.values):
        ax.text(v + 25, y, f"{v:,}", va="center", fontsize=7, color=MUTED)
    ax.set_xlabel("studies")
    ax.set_xlim(0, l.max() * 1.18)
    ax.grid(axis="y", visible=False)
    panel(ax, "a", "Report language")
    ax = axs[0, 1]
    v = ven.sort_values()
    ax.barh(v.index, v.values, color=BLUE, height=0.65)
    for y, val in enumerate(v.values):
        ax.text(val + 25, y, f"{val:,}", va="center", fontsize=7, color=MUTED)
    ax.set_xlim(0, v.max() * 1.2)
    ax.set_xlabel("studies")
    ax.grid(axis="y", visible=False)
    panel(ax, "b", "Scanner vendor")
    ax = axs[1, 0]
    f = fld.sort_index()
    names = [f"{x:g} T" for x in f.index]
    ax.bar(names, f.values, color=BLUE, width=0.6)
    for x, val in enumerate(f.values):
        ax.text(x, val + 40, f"{val:,}", ha="center", fontsize=7, color=MUTED)
    ax.set_ylabel("studies")
    ax.set_ylim(0, f.max() * 1.15)
    ax.grid(axis="x", visible=False)
    panel(ax, "c", "Magnetic field strength")
    ax = axs[1, 1]
    x = np.arange(len(hist))
    w = 0.28
    for i, (plane, col, hatch) in enumerate((("Sagittal", BLUE, ""), ("Coronal", ORANGE, "///"), ("Axial", AQUA, "..."))):
        ax.bar(x + (i - 1) * w, hist[plane].values, w, color=col, label=plane, hatch=hatch, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x, [s.replace("-", "–") for s in hist.index], rotation=60, ha="right")
    ax.set_xlabel("slices per series")
    ax.set_ylabel("series")
    ax.legend(loc="upper right", handlelength=1.2)
    ax.grid(axis="x", visible=False)
    panel(ax, "d", "Series length by plane")
    save(fig, "fig_dataset")


# ------------------------------------------------------------------ label prevalence and label quality
def fig_labels():
    p = pd.read_csv(DATA / "prevalence_by_source.csv", index_col=0)
    lab = pd.read_csv(DATA / "label_sources_gold_auc.csv")
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.9), gridspec_kw={"width_ratios": [1.2, 0.8], "wspace": 1.15})
    ax = axs[0]
    cols = [("gold (58, image read)", GREY, "xxxx", "Gold-58 (enriched, image read)"),
            ("our report-derived labels", BLUE, "", "Our report-derived labels"),
            ("7th-place image-read labels", ORANGE, "///", "7th-place image-read labels"),
            ("1st-place image-read labels", AQUA, "...", "1st-place image-read labels")]
    y = np.arange(len(LABELS))[::-1]
    h = 0.2
    for i, (c, col, hatch, name) in enumerate(cols):
        ax.barh(y + (1.5 - i) * h, p[c].reindex(LABELS).values, h, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=name)
    ax.set_yticks(y, LABELS)
    ax.set_xlabel("share of studies called positive (score ≥ 0.5)")
    ax.set_xlim(0, 0.75)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, columnspacing=0.8, handlelength=1.2)
    panel(ax, "a", "Positive rate by label source")
    ax = axs[1]
    lab = lab.sort_values("macro_auc")
    yy = np.arange(len(lab))
    for k, (kind, col, mk) in enumerate((("report", BLUE, "o"), ("image", ORANGE, "s"))):
        m = (lab.kind == kind).to_numpy()
        ax.errorbar(lab.macro_auc[m], yy[m], xerr=[(lab.macro_auc - lab.lo)[m], (lab.hi - lab.macro_auc)[m]], fmt=mk, color=col,
                    ms=4.5, elinewidth=1, capsize=2, label="from reports" if kind == "report" else "from images (published)")
    ax.set_yticks(yy, lab.source, fontsize=7)
    ax.set_xlabel("macro AUC on gold-58, 95% CI")
    ax.set_xlim(0.75, 1.0)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.13), ncol=1, handlelength=1.2)
    panel(ax, "b", "Agreement with gold-58")
    save(fig, "fig_labels")


# ------------------------------------------------------------------ training curves
def fig_curves():
    mil = pd.read_csv(DATA / "curves_mil.csv")
    bp = pd.read_csv(DATA / "curves_boxpool.csv")
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.9), gridspec_kw={"wspace": 0.3})
    ax = axs[0]
    styles = {"E1": (BLUE, "o", "-"), "E2a mean pooling": (ORANGE, "s", "-"), "E6 EfficientNetV2-S": (AQUA, "^", "-"),
              "E1L 12 epochs": (VIOLET, "D", "--"), "E0 frozen features": (GREY, "v", ":")}
    for run, (col, mk, ls) in styles.items():
        d = mil[mil.run == run]
        ax.plot(d.epoch, d.pseudo_val_macro_auc, ls, color=col, marker=mk, ms=3.5, lw=1.4,
                label=run.replace("E0 frozen features", "E0 frozen (3-seed mean)"))
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro AUC, held-out fold 0 (report labels)")
    ax.set_ylim(0.64, 0.90)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2, columnspacing=0.8, handlelength=1.8)
    panel(ax, "a", "2.5D MIL model (our design)")
    ax = axs[1]
    styles = {"A0 (1st-place labels, fold 0)": (BLUE, "o", "-"), "A1 (7th-place labels, fold 0)": (ORANGE, "s", "-"),
              "A2 (fused labels, fold 0)": (AQUA, "^", "-"), "C1 (fused, fold 1)": (GREY, "v", "--"),
              "C2 (fused, fold 2)": (GREY, "D", ":"), "C3 (fused, fold 3)": (GREY, "x", "-.")}
    for run, (col, mk, ls) in styles.items():
        d = bp[bp.run == run]
        ax.plot(d.epoch, d.gold58_macro_auc, ls, color=col, marker=mk, ms=3.5, lw=1.3, label=run)
    ax.set_xlabel("epoch")
    ax.set_ylabel("macro AUC on gold-58")
    ax.set_ylim(0.895, 0.95)
    ax.legend(loc="lower right", fontsize=6.5)
    panel(ax, "b", "Box-pooling model (post-competition)")
    save(fig, "fig_curves")


# ------------------------------------------------------------------ ablations of the 2.5D model
def fig_ablation():
    t = pd.read_csv(DATA / "ablation_paired_vs_E1.csv")
    names = {"E1s2_cnxtT_s3_k8_attn12_seed2": "Seed 2 (same recipe)", "E1p_cnxtT_s3_k8_attn12_posfix": "Slice-position fix (E1p)",
             "E2c_cnxtT_s3_k8_tfm": "Transformer + attention (E2c)", "E1L_cnxtT_s3_k8_attn12_ep12": "12 epochs (E1L)",
             "E8_cnxtT_s3_k8_attn12_v6r256": "0.6 mm/px, 256 px (E8)", "E4_cnxtT_s6_k8_attn12": "Six series (E4)",
             "E5w_cnxtT_s3_k8_attn12_confw1": "Confidence-weighted labels (E5w)", "E2a_cnxtT_s3_k8_mean": "Mean pooling (E2a)",
             "E6_effv2s_s3_k8_attn12": "EfficientNetV2-S (E6)"}
    t = t[t.run.isin(names)].copy()
    t["name"] = t.run.map(names)
    t = t.set_index("name").loc[list(names.values())[::-1]]
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(t))
    for ax, pre, letter, title in ((axs[0], "pseudo_val", "a", "Held-out fold 0, n = 870"), (axs[1], "gold", "b", "Gold-58, n = 58")):
        d, lo, hi = t[f"{pre}_delta"], t[f"{pre}_lo"], t[f"{pre}_hi"]
        seed = t.index == "Seed 2 (same recipe)"
        ax.errorbar(d[~seed], y[~seed], xerr=[(d - lo)[~seed], (hi - d)[~seed]], fmt="o", color=BLUE, ms=4, elinewidth=1, capsize=2)
        ax.errorbar(d[seed], y[seed], xerr=[(d - lo)[seed], (hi - d)[seed]], fmt="D", color=ORANGE, ms=4, elinewidth=1, capsize=2)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("Δ macro ROC-AUC vs E1 (95% CI)")
        ax.grid(axis="y", visible=False)
        panel(ax, letter, title)
    axs[0].set_yticks(y, t.index)
    save(fig, "fig_ablation")


# ------------------------------------------------------------------ where the attention goes
def fig_attention():
    a = pd.read_csv(DATA / "attention_plane_share.csv").set_index("target").reindex(LABELS)
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True, gridspec_kw={"width_ratios": [1.4, 1], "wspace": 0.08})
    y = np.arange(len(LABELS))[::-1]
    ax = axs[0]
    left = np.zeros(len(LABELS))
    for col, c, hatch, name in (("mass_sag_fs", BLUE, "", "Sagittal"), ("mass_cor_fs", ORANGE, "///", "Coronal"), ("mass_axi_fs", AQUA, "...", "Axial")):
        v = a[col].to_numpy()
        ax.barh(y, v, 0.62, left=left, color=c, hatch=hatch, edgecolor="white", linewidth=0.8, label=name)
        left += v
    ax.set_yticks(y, LABELS)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of attention mass")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.2)
    panel(ax, "a", "Attention by plane")
    ax = axs[1]
    ax.scatter(a.meanpos_sag_fs, y, color=BLUE, s=22, marker="o", label="sagittal")
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0.3, 0.75)
    ax.set_xlabel("mean position (0 lateral, 1 medial)")
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(axis="y", visible=False)
    panel(ax, "b", "Position in sagittal series")
    save(fig, "fig_attention")


# ------------------------------------------------------------------ per-finding gold-58
def fig_gold():
    g = pd.read_csv(DATA / "gold_per_finding.csv")
    pos = pd.read_csv(DATA / "gold_positives.csv", index_col=0)["gold_positives"]
    models = [("E0 frozen EfficientNet-B0", GREY, "xxxx", "Frozen EfficientNet-B0 (E0)"),
              ("2.5D MIL ensemble (F1-F3)", BLUE, "", "Our 2.5D MIL ensemble (F1–F3)"),
              ("Box-pooling, six-model mean", ORANGE, "///", "Box-pooling, six-model mean")]
    rows = ["macro"] + LABELS
    names = ["Macro (all 12)"] + [f"{k} ({pos[k]})" for k in LABELS]
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 4.6))
    y = np.arange(len(rows))[::-1].astype(float)
    y[0] += 0.4
    h = 0.26
    for i, (m, col, hatch, nm) in enumerate(models):
        v = g[g.model == m].set_index("finding").auc.reindex(rows).to_numpy()
        ax.barh(y + (1 - i) * h, v, h, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=nm)
    ax.set_yticks(y, names)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC on gold-58 (number of positive studies in brackets)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, handlelength=1.2)
    save(fig, "fig_gold_per_finding")


# ------------------------------------------------------------------ label swap
def fig_labelswap():
    t = pd.read_csv(DATA / "label_swap_gold.csv")
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True, gridspec_kw={"width_ratios": [4, 3], "wspace": 0.08})
    colmap = {"Our report-derived labels": GREY, "1st-place labels": AQUA, "7th-place labels": ORANGE, "Fused 1st + 7th": BLUE}
    hatchmap = {"Our report-derived labels": "xxxx", "1st-place labels": "...", "7th-place labels": "///", "Fused 1st + 7th": ""}
    for ax, model, letter in ((axs[0], "2.5D MIL (our model, 224 px)", "a"), (axs[1], "Box-pooling ConvNeXt-T (256 px)", "b")):
        d = t[t.model == model].reset_index(drop=True)
        x = np.arange(len(d))
        ax.bar(x, d.gold_auc, 0.6, color=[colmap[l] for l in d.labels], hatch=[hatchmap[l] for l in d.labels], edgecolor="white", linewidth=0.3)
        ax.errorbar(x, d.gold_auc, yerr=[d.gold_auc - d.lo, d.hi - d.gold_auc], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2.5)
        for xi, v, top in zip(x, d.gold_auc, d.hi):
            ax.text(xi, top + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x, [l.replace("Our report-derived labels", "Our report-\nderived labels").replace("Fused 1st + 7th", "Fused\n1st + 7th").replace("-place labels", "-place\nlabels") for l in d.labels], fontsize=7)
        ax.set_ylim(0.83, 0.98)
        ax.grid(axis="x", visible=False)
        panel(ax, letter, model)
    axs[0].set_ylabel("macro ROC-AUC on gold-58 (95% CI)")
    save(fig, "fig_label_swap")


# ------------------------------------------------------------------ attention sharpening
def fig_sharpen():
    s = pd.read_csv(DATA / "sharpening_sweep.csv")
    pf = pd.read_csv(DATA / "sharpening_per_finding.csv")
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.35})
    ax = axs[0]
    for model, col, mk in (("A0", BLUE, "o"), ("A1", ORANGE, "s")):
        for ref, ls in (("1st-place labels", "-"), ("7th-place labels", "--")):
            d = s[(s.model == model) & (s.reference == ref)].sort_values("s")
            ax.plot(d.s, d.delta * 1000, ls, color=col, marker=mk, ms=3.5, lw=1.2, label=f"{model} vs {ref.split()[0]} labels")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0.5, 1, 1.5, 2, 3], ["0.5", "1", "1.5", "2", "3"])
    ax.set_xlim(0.35, 3.15)
    ax.set_xlabel("sharpening factor s")
    ax.set_ylabel("Δ macro ROC-AUC vs s = 1 (×10⁻³)")
    ax.legend(fontsize=6.5, loc="lower center")
    panel(ax, "a", "Macro effect, published fold 0")
    ax = axs[1]
    d = pf[(pf.model == "A1") & (pf.reference == "1st-place labels")]
    y = np.arange(len(LABELS))[::-1]
    h = 0.26
    for i, (sv, col, hatch) in enumerate(((1.25, "#9cc3ef", ""), (2.0, BLUE, ""), (3.0, "#123f73", ""))):
        v = d[d.s == sv].set_index("finding").delta.reindex(LABELS).to_numpy() * 1000
        ax.barh(y + (1 - i) * h, v, h, color=col, label=f"s = {sv:g}")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, LABELS, fontsize=7)
    ax.set_xlabel("Δ ROC-AUC vs s = 1 (×10⁻³)")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7, loc="lower left")
    panel(ax, "b", "Per finding, model A1")
    save(fig, "fig_sharpen")


# ------------------------------------------------------------------ leaderboard history
def fig_leaderboard():
    t = pd.read_csv(DATA / "submissions_owner.csv")
    t = t.dropna(subset=["public"]).reset_index(drop=True)
    stage_col = {"Stage 1": GREY, "Stage 3": BLUE, "Stage 4": ORANGE, "Stage 4 (last day)": RED, "Stage 5": AQUA}
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.25})
    x = t.order.to_numpy()
    for ax, ylim, xlim, letter, title in ((axs[0], (0.45, 1.0), (0, 23), "a", "All scored submissions"),
                                          (axs[1], (0.84, 0.975), (2.4, 22.6), "b", "Submissions 3 to 22, above 0.84")):
        ax.plot(x, t.public, color=MUTED, lw=0.6, alpha=0.6)
        ax.plot(x, t.private, color=MUTED, lw=0.6, alpha=0.6, ls="--")
        for st, col in stage_col.items():
            m = (t.stage == st).to_numpy()
            ax.scatter(x[m], t.public[m], color=col, marker="o", s=20, zorder=3)
            ax.scatter(x[m], t.private[m], facecolor="white", edgecolor=col, marker="s", s=20, lw=1.1, zorder=3)
        ax.set_ylim(*ylim)
        ax.set_xlim(*xlim)
        ax.set_xlabel("submission number (order made)")
        ax.set_xticks(range(2, 23, 2) if letter == "a" else range(3, 23))
        ax.tick_params(axis="x", labelsize=6.5)
        panel(ax, letter, title)
    axs[0].set_ylabel("macro ROC-AUC")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=c, ms=4.5, label=s) for s, c in stage_col.items()]
    handles += [Line2D([], [], marker="o", ls="-", lw=0.6, color=MUTED, ms=4.5, label="filled, solid: public LB"),
                Line2D([], [], marker="s", ls="--", lw=0.6, mfc="white", mec=MUTED, color=MUTED, ms=4.5, label="open, dashed: private LB")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=7)
    save(fig, "fig_leaderboard")


if __name__ == "__main__":
    fig_dataset()
    fig_labels()
    fig_curves()
    fig_ablation()
    fig_attention()
    fig_gold()
    fig_labelswap()
    fig_sharpen()
    if (DATA / "submissions_owner.csv").exists():
        fig_leaderboard()
