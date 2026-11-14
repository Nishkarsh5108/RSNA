"""Build a private Kaggle notebook that trains the box-pooling classifier (series-level ConvNeXt-T, ROI box pooling,
lesion + report auxiliary heads) with our switches.

Base code: the public notebook "RSNA 2026 Knee Box-Pooling Classifier" by tascj0 (1st place, Team Kangaroo), used
with the public dataset tascj0/rsna-2026-knee-agent-labels (CC BY 4.0). This script patches a local copy of that
classifier.py as text (it is never executed locally) and writes <out>/<slug>.ipynb + kernel-metadata.json.

Switches added (environment variables read by classifier.py):
  LABEL_FILE     training targets CSV (default: the dataset's labels_astra_trained_guide.csv)
  FOLD, SEED     fold of the plain 5-fold split / seed
  FULL_FIT=1     train on every non-gold study (the fold's studies are still predicted, but are then in-sample)
  EPOCHS         default 8
  TIME_BUDGET_H  stop cleanly after an epoch when the next one would not fit (weights are saved every epoch)
"""
import argparse
import json
import os

BASE = "D:/rsna_work/public_code/tascj/cls_output/classifier.py"


def sub(src, old, new, count=1):
    assert src.count(old) == count, f"expected {count} x {old!r}, found {src.count(old)}"
    return src.replace(old, new)


def patch(src):
    src = sub(src, 'OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))',
              'OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))\n'
              'LABEL_FILE = Path(os.environ.get("LABEL_FILE", DATA / "labels_astra_trained_guide.csv"))\n'
              'FULL_FIT = os.environ.get("FULL_FIT", "0") == "1"\n'
              'TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))')
    src = sub(src, "FOLD, N_FOLDS, SEED = 0, 5, 3",
              'FOLD, N_FOLDS, SEED = int(os.environ.get("FOLD", 0)), 5, int(os.environ.get("SEED", 3))')
    src = sub(src, "EPOCHS = 1 if DEBUG else 8", 'EPOCHS = 1 if DEBUG else int(os.environ.get("EPOCHS", 8))')
    src = sub(src, 'labels = pd.read_csv(DATA / "labels_astra_trained_guide.csv").StudyInstanceUID',
              "labels = pd.read_csv(LABEL_FILE).StudyInstanceUID")
    src = sub(src, "    return set(studies[folds != FOLD]), set(studies[folds == FOLD]) | set(gold)",
              "    return set(studies if FULL_FIT else studies[folds != FOLD]), set(studies[folds == FOLD]) | set(gold)")
    src = sub(src, 'labels = pd.read_csv(DATA / "labels_astra_trained_guide.csv").set_index("StudyInstanceUID")[LABELS].astype(np.float32)',
              'labels = pd.read_csv(LABEL_FILE).set_index("StudyInstanceUID")[LABELS].astype(np.float32)')
    src = sub(src, "    step = 0\n    for epoch in range(1, EPOCHS + 1):",
              "    if rank == 0:\n"
              '        print(f"config: labels {LABEL_FILE} | fold {FOLD} seed {SEED} full_fit {FULL_FIT} epochs {EPOCHS} | crop {CROP_MM} mm at {CROP_PX} px | "\n'
              '              f"backbone {BACKBONE} | steps roi_pool {ROI_POOLING} roi_crop {ROI_CROP} attention {ATTENTION} lesion {LESION_HEAD} report {REPORT_HEAD}", flush=True)\n'
              "    step, run_start = 0, time.time()\n    for epoch in range(1, EPOCHS + 1):")
    src = sub(src, "        dist.barrier()\n    dist.destroy_process_group()",
              "        dist.barrier()\n"
              "        stop = [bool((time.time() - run_start) + 1.15 * (time.time() - epoch_start) > TIME_BUDGET_H * 3600)] if rank == 0 else [None]\n"
              "        dist.broadcast_object_list(stop, src=0)  # rank 0 decides, so every rank leaves the loop together\n"
              "        if stop[0] and epoch < EPOCHS:\n"
              "            if rank == 0:\n"
              '                print(f"time budget {TIME_BUDGET_H} h: stopping after epoch {epoch}", flush=True)\n'
              "            break\n"
              "    dist.destroy_process_group()")
    # Inference-time switch (own idea N4): sharpen the in-box attention. 1.0 = unchanged, so training is not affected.
    src = sub(src, 'TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))',
              'TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))\n'
              '_sharp = [float(v) for v in os.environ.get("ATT_SHARP", "1").split(",")]\n'
              "ATT_SHARP = _sharp[0] if len(_sharp) == 1 else _sharp  # one value, or one per label (in LABELS order)")
    src = sub(src, "            weight = weight * torch.exp(score - top[slice_series][..., None, None])",
              "            sharp = torch.tensor(ATT_SHARP, device=score.device).view(1, -1, 1, 1) if isinstance(ATT_SHARP, (list, tuple)) else ATT_SHARP\n"
              "            weight = weight * torch.exp(sharp * (score - top[slice_series][..., None, None]))")
    return patch_series_preds(src)


def patch_series_preds(src):
    """Also write the validation predictions per series (fold{F}_series_epoch{NN}.csv): needed to test other ways of
    combining a study's series than the plain mean, without spending GPU time on a second inference pass."""
    src = sub(src, "study=study, lesion_boxes=torch.from_numpy(lesion),", "study=study, series=uid, lesion_boxes=torch.from_numpy(lesion),")
    src = sub(src, '        study=[s["study"] for s in items],\n    )',
              '        study=[s["study"] for s in items],\n        series=[s["series"] for s in items],\n    )')
    src = sub(src, "@torch.no_grad()\ndef predict(model, loader, device):",
              "SERIES_ROWS = []  # (series, study, 12 probabilities) of this rank's last predict() call\n\n\n"
              "@torch.no_grad()\ndef predict(model, loader, device):")
    src = sub(src, "    model.eval()\n    out = {}\n    for x in loader:", "    model.eval()\n    out = {}\n    SERIES_ROWS.clear()\n    for x in loader:")
    src = sub(src, '        for study, p in zip(x["study"], torch.sigmoid(logits.float()).cpu().numpy(), strict=True):\n'
                   "            out.setdefault(study, []).append(p)\n",
              '        for study, uid, p in zip(x["study"], x["series"], torch.sigmoid(logits.float()).cpu().numpy(), strict=True):\n'
              "            out.setdefault(study, []).append(p)\n"
              "            SERIES_ROWS.append((uid, study, *p.tolist()))\n")
    # VALID_FROM: before that epoch only the gold studies are predicted (a cheap sanity number). The fold's validation
    # costs 15% of an epoch (measured on A0/A1: 460 s of 2,980 s) and only the last epoch's predictions are used.
    src = sub(src, 'TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))\n',
              'TIME_BUDGET_H = float(os.environ.get("TIME_BUDGET_H", 11.0))\nVALID_FROM = int(os.environ.get("VALID_FROM", 1))\n')
    src = sub(src, "    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(valid_set, mine), batch_size=8, **loader_args)\n",
              "    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(valid_set, mine), batch_size=8, **loader_args)\n"
              "    gold_mine = [i for i in mine if valid_set.series.StudyInstanceUID.iloc[i] in gold.index]\n"
              "    gold_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(valid_set, gold_mine), batch_size=8, **loader_args)\n")
    src = sub(src, "        dist.all_gather_object(parts, predict(ema, valid_loader, device))\n",
              "        dist.all_gather_object(parts, predict(ema, valid_loader if epoch >= VALID_FROM else gold_loader, device))\n"
              "        series_parts = [None] * world\n"
              "        dist.all_gather_object(series_parts, list(SERIES_ROWS))\n")
    src = sub(src, '            prob.to_csv(OUT / f"fold{FOLD}_epoch{epoch:02d}.csv", index_label="StudyInstanceUID")\n',
              '            prob.to_csv(OUT / f"fold{FOLD}_epoch{epoch:02d}.csv", index_label="StudyInstanceUID")\n'
              '            pd.DataFrame([r for part in series_parts for r in part], columns=["SeriesInstanceUID", "StudyInstanceUID", *LABELS]).to_csv(\n'
              '                OUT / f"fold{FOLD}_series_epoch{epoch:02d}.csv", index=False)\n')
    return src


LAUNCH = '''import os, subprocess, sys
def first(*ps):
    return next((p for p in ps if os.path.exists(p)), None)
env = dict(os.environ, DEBUG="0", **{ENV})
def resolve(spec):
    owner, slug, name = spec.split("/")
    p = first(f"/kaggle/input/datasets/{owner}/{slug}/{name}", f"/kaggle/input/{slug}/{name}")
    assert p, f"label file {spec} is not mounted: {os.listdir('/kaggle/input')}"
    return p
lab, lab2 = {LABEL!r}, {LABEL2!r}
if lab and lab2:  # train on the mean of two label sets (same studies, same columns)
    import pandas as pd
    a, b = (pd.read_csv(resolve(s)).set_index("StudyInstanceUID") for s in (lab, lab2))
    cols = [c for c in a.columns if c in b.columns]
    common = a.index.intersection(b.index)
    fused = (a.loc[common, cols] + b.loc[common, cols]) / 2
    fused.to_csv("/tmp/fused_labels.csv", index_label="StudyInstanceUID")
    print("fused labels:", len(fused), "studies,", len(cols), "columns, mean", round(float(fused.mean().mean()), 4), flush=True)
    env["LABEL_FILE"] = "/tmp/fused_labels.csv"
elif lab:
    env["LABEL_FILE"] = resolve(lab)
subprocess.run("nvidia-smi --query-gpu=index,name,memory.total --format=csv; nproc; free -g | head -2; df -h /kaggle/working | tail -1", shell=True)
r = subprocess.run("torchrun --standalone --nproc_per_node 2 classifier.py", shell=True, env=env)
print("exit code", r.returncode, flush=True)
sys.exit(r.returncode)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="kernel slug, e.g. knee-bp-a0-f0")
    ap.add_argument("--user", default="anujchaubey")
    ap.add_argument("--label", default="", help="owner/dataset-slug/file.csv of the training targets ('' = the base default)")
    ap.add_argument("--label2", default="", help="second label file (same form): train on the mean of --label and --label2")
    ap.add_argument("--env", default="", help="comma list KEY=VALUE passed to classifier.py (FOLD, SEED, FULL_FIT, EPOCHS, VALID_FROM, ...)")
    ap.add_argument("--out-root", default="D:/rsna_work/kaggle_runs")
    a = ap.parse_args()
    src = patch(open(BASE, encoding="utf-8").read())
    env = dict(kv.split("=", 1) for kv in a.env.split(",") if kv)
    launch = LAUNCH.replace("{ENV}", repr(env)).replace("{LABEL!r}", repr(a.label)).replace("{LABEL2!r}", repr(a.label2))
    d = f"{a.out_root}/{a.slug}"
    os.makedirs(d, exist_ok=True)

    def cell(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}

    nb = {"cells": [cell("%%writefile classifier.py\n" + src), cell(launch)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"{d}/{a.slug}.ipynb", "w", encoding="utf-8"), indent=1)
    open(f"{d}/classifier_patched.py", "w", encoding="utf-8").write(src)
    datasets = ["tascj0/rsna-2026-knee-agent-labels"]
    for lab in (a.label, a.label2):
        if lab:
            datasets.append("/".join(lab.split("/")[:2]))
    meta = {"id": f"{a.user}/{a.slug}", "title": a.slug.replace("-", " "), "code_file": f"{a.slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": sorted(set(datasets)), "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print("wrote", d, "| env", env, "| label", a.label or "(base default)", ("+ " + a.label2) if a.label2 else "")


if __name__ == "__main__":
    main()
