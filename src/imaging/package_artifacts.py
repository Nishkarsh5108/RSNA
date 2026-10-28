"""Package local experiment + model artefacts for analysis and for the Kaggle upload.

Creates (all git-ignored; derived from competition data -> keep private):
  output/imaging_v5/rsna_artifacts/            structured folder (models/, experiments/, analysis/, code/)
  output/imaging_v5/rsna_model_artifacts.zip   everything incl. model weights
  output/imaging_v5/rsna_artifacts_light.zip   no weights (reports, configs, metrics, predictions, code)
  D:/rsna_work/kaggle_upload/rsna-knee-image-model/   folder to upload as a PRIVATE Kaggle dataset
                                                      (knee_*.pt + wheels/ + README)
Usage: python package_artifacts.py RUN_ID[,RUN_ID...]   (runs whose ckpt_best.pt become knee_<run>.pt)
"""
import glob
import json
import os
import platform
import shutil
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
RUNS = f"{R}/output/imaging_v5/runs"
A = f"{R}/output/imaging_v5/rsna_artifacts"
UP = "D:/rsna_work/kaggle_upload/rsna-knee-image-model"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion",
           "Synovitis", "Baker's", "Contusion", "Fracture"]


def main(final_runs):
    shutil.rmtree(A, ignore_errors=True)
    for d in ("models", "experiments", "analysis", "code", "code/src_imaging", "code/notebooks"):
        os.makedirs(f"{A}/{d}", exist_ok=True)
    os.makedirs(UP, exist_ok=True)
    # 2026-10-18 guard: this folder holds the 0.927 submission (fallback C). Never delete it; new releases are packaged
    # by src/emergency/package_release.py into their own folders.
    released = {"knee_F1_cnxtT_s3_k8_attn12_pf_all_s42.pt", "knee_F2_cnxtT_s3_k8_attn12_pf_all_s7.pt",
                "knee_F3_cnxtT_s3_k8_attn12_pf_all_s11.pt"}
    if released & {os.path.basename(f) for f in glob.glob(f"{UP}/knee_*.pt")} and "--force-overwrite-release" not in sys.argv:
        raise SystemExit(f"{UP} holds the released 0.927 checkpoints; refusing to overwrite (see src/emergency/package_release.py)")
    for f in glob.glob(f"{UP}/knee_*.pt"):     # the inference notebook ensembles EVERY .pt it finds: keep only this release
        os.remove(f)
    # ---------------- models
    pp_used, args_used = {}, {}
    for rid in final_runs:
        ck = torch.load(f"{RUNS}/{rid}/ckpt_best.pt", map_location="cpu", weights_only=False)
        torch.save(ck, f"{A}/models/knee_{rid}.pt")
        shutil.copy(f"{A}/models/knee_{rid}.pt", f"{UP}/knee_{rid}.pt")
        json.dump(ck["args"], open(f"{A}/models/knee_{rid}_config.json", "w"), indent=1)
        pp_used[rid] = ck["args"].get("preproc", "v2")
        args_used[rid] = ck["args"]
    import timm
    json.dump({"targets": TARGETS, "label_columns_order": TARGETS,
               "preprocessing": {"source": "DICOM", "slice_order": "ImagePositionPatient . slice normal",
                                 "intensity": "per-series 1st-99th percentile -> uint8",
                                 "resize_per_model": {r: ("v6: PIL bilinear to 0.6 mm/px (aspect kept), centred 256x256 crop/zero-pad"
                                                          if v == "v6" else "v2: PIL bilinear 224x224") for r, v in pp_used.items()},
                                 "laterality": "tag majority else coronal/axial geometry; cor/ax mirror columns, sagittal reverse slices (right knees)",
                                 "windows": "3 slices [c-s,c,c+s], s=round(3.5mm/spacing), centres 4-96%"},
               "inference": {"aggregation": "mean of sigmoid probabilities over models (equal weights)",
                             "tta": "tta_shift=True: mean of 2 window bags (training setting + shifted by half a window step)"},
               "models": {r: {k: a.get(k) for k in ("backbone", "slots", "K", "agg", "preproc", "sag_pos_fix", "epochs", "seed",
                                                    "train_all_folds", "index", "label_transform", "conf_weight")}
                          for r, a in args_used.items()},
               "versions": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__}},
              open(f"{A}/models/inference_config.json", "w"), indent=1)
    # ---------------- experiments
    for rd in sorted(glob.glob(f"{RUNS}/*/")):
        rid = os.path.basename(rd.rstrip("/\\"))
        if rid == "smoke":
            continue
        os.makedirs(f"{A}/experiments/{rid}", exist_ok=True)
        for f in ("config.json", "history.csv", "summary.json", "val_preds_best.csv", "gold_preds_SEALED.npz"):
            if os.path.exists(f"{rd}/{f}"):
                shutil.copy(f"{rd}/{f}", f"{A}/experiments/{rid}/{f}")
    for f in glob.glob(f"{R}/reports/eda/imaging/*"):
        shutil.copy(f, f"{A}/analysis/")
    # curves
    hs = {os.path.basename(os.path.dirname(p)): pd.read_csv(p) for p in glob.glob(f"{RUNS}/*/history.csv")
          if "smoke" not in p and "train_loss" in pd.read_csv(p).columns}
    if hs:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        for k, h in hs.items():
            ax[0].plot(h.epoch, h.train_loss, label=k); ax[1].plot(h.epoch, h.val_macro_auc, label=k)
        ax[0].set_title("train loss"); ax[1].set_title("pseudo-val macro AUC (fold 0)")
        ax[1].legend(fontsize=6); plt.tight_layout(); plt.savefig(f"{A}/analysis/training_curves.png", dpi=110); plt.close()
    # prediction correlations between runs (pseudo-val)
    vp = {k: pd.read_csv(f"{RUNS}/{k}/val_preds_best.csv").set_index("StudyInstanceUID")[TARGETS]
          for k in hs if os.path.exists(f"{RUNS}/{k}/val_preds_best.csv")}
    if len(vp) > 1:
        keys = list(vp)
        common = sorted(set.intersection(*[set(v.index) for v in vp.values()]))
        C = np.array([[np.mean([np.corrcoef(vp[a].loc[common, t], vp[b].loc[common, t])[0, 1] for t in TARGETS])
                       for b in keys] for a in keys])
        pd.DataFrame(C, index=keys, columns=keys).round(3).to_csv(f"{A}/analysis/val_prediction_correlation_between_runs.csv")
        for k, v in vp.items():
            v.describe().T.to_csv(f"{A}/analysis/score_distribution_{k}.csv")
    shutil.copy(f"{R}/data/imaging_v5/study_index_train_v2.csv", f"{A}/analysis/split_and_laterality_index_v2.csv")
    # ---------------- code + docs
    for f in glob.glob(f"{R}/src/imaging/*.py"):
        shutil.copy(f, f"{A}/code/src_imaging/")
    shutil.copy(f"{R}/src/labeling/metrics.py", f"{A}/code/src_imaging/metrics.py")
    for f in glob.glob(f"{R}/notebooks/rsna_knee_image_model_*.ipynb"):
        shutil.copy(f, f"{A}/code/notebooks/")
    for f in ("image_model_experiments.md", "FINAL_IMAGE_MODEL_REPORT.md", "image_pipeline_audit.md", "image_model_literature_review.md"):
        if os.path.exists(f"{R}/reports/{f}"):
            shutil.copy(f"{R}/reports/{f}", A)
    reqs = {"torch": torch.__version__, "timm": timm.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    import pydicom, PIL
    reqs.update({"pydicom": pydicom.__version__, "Pillow": PIL.__version__})
    open(f"{A}/requirements.txt", "w").write("\n".join(f"{k}=={v}" for k, v in reqs.items()) + "\n")
    open(f"{A}/README.md", "w", encoding="utf-8").write(README.format(runs=", ".join(final_runs)))
    a0 = args_used[final_runs[0]]
    open(f"{A}/MODEL_CARD.md", "w", encoding="utf-8").write(MODEL_CARD.format(
        n=len(final_runs), backbone=a0["backbone"], K=a0["K"], agg=a0["agg"], slots=a0["slots"],
        res="224 px squash (v2)" if a0.get("preproc", "v2") == "v2" else "0.6 mm/px, 256 px (v6)",
        data="all 5 non-gold folds (4,349 studies)" if a0.get("train_all_folds") else "folds 1-4 (3,479 studies)",
        epochs=a0["epochs"], posfix="yes" if a0.get("sag_pos_fix") else "no (original mirrored input)"))
    open(f"{UP}/README.md", "w", encoding="utf-8").write(UPLOAD_README.format(runs=", ".join(f"knee_{r}.pt" for r in final_runs)))
    # ---------------- archives
    out_full, out_light = f"{R}/output/imaging_v5/rsna_model_artifacts.zip", f"{R}/output/imaging_v5/rsna_artifacts_light.zip"
    for out, with_w in ((out_full, True), (out_light, False)):
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(A):
                for f in files:
                    if not with_w and f.endswith(".pt"):
                        continue
                    p = os.path.join(root, f)
                    z.write(p, os.path.relpath(p, A))
        with zipfile.ZipFile(out) as z:
            bad = z.testzip()
            print(out, f"{os.path.getsize(out)/1e6:.1f} MB", "files", len(z.namelist()), "integrity", "OK" if bad is None else bad)
    print("kaggle upload folder:", UP, os.listdir(UP))


README = """# RSNA knee image-model artefacts (local run)

Final model checkpoint(s): {runs} (`models/knee_<run>.pt`; dict with `model` state_dict (EMA weights) and `args`).

- `experiments/<run>/`: config, per-epoch history, summary, pseudo-val predictions (fold 0), sealed gold predictions
- `analysis/`: experiment tables (gold scored once, after selection), training curves, between-run prediction correlations,
  score distributions, orientation checks, split and laterality index
- `code/`: exact code used (src/imaging, metrics) and the Kaggle notebooks

Reproduce a run: `python code/src_imaging/train.py --run-id X <args from config.json>` (needs the v2 preprocessing cache).
Local analysis: load `val_preds_best.csv` per run and join on StudyInstanceUID with your labels.
Keep this archive private: it contains study identifiers and label-derived data from the competition.
"""
MODEL_CARD = """# Model card: RSNA knee 2.5D multi-slot MIL ({backbone})

- **Task.** 12 binary knee-MRI findings per study (competition metric: macro ROC-AUC). Research and competition use only;
  not a medical device.
- **Ensemble.** {n} checkpoint(s), equal-weight probability average, each with 2 window bags (shift TTA).
- **Inputs.** Slots {slots} (0,1,2 = sagittal/coronal/axial fluid-sensitive series, with the plane's non-FS series as fallback).
  Each series: K = {K} windows of 3 adjacent slices (about 3.5 mm apart), {res}, laterality-normalised:
  right-knee coronal/axial mirrored, sagittal slice order reversed; correct sagittal position input: {posfix}.
  No metadata or report text is used as model input.
- **Architecture.** {backbone} encoder (ImageNet weights, fine-tuned) → 384-d projection + slot and slice-position
  embeddings → {agg} aggregation → 12 logits.
- **Training.** {data}; never on the 58 gold studies; {epochs} epochs; AdamW + cosine; EMA weights; soft report-derived
  labels (labels_v3 teacher, about 0.89 macro AUC vs gold).
- **Validation.** Recipe selected on pseudo-labelled fold 0 (870 studies; see image_model_experiments.md). Gold-58 scored
  once after the selection was locked (FINAL_IMAGE_MODEL_REPORT.md).
- **Known limitations.**
  - Report-derived labels are noisy and site-biased.
  - Gold-58 is tiny (CI about ±0.05).
  - 70 untagged near-midline studies may be mis-oriented.
  - The hidden test's site and prevalence mix is unknown.
"""
UPLOAD_README = """Private Kaggle dataset for the RSNA knee inference notebook. Do NOT make it public.
Contents: {runs} (model checkpoints) and wheels/ (pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0 and pylibjpeg-openjpeg 2.6.0
for CPython 3.10-3.14 manylinux x86_64, pydicom 3.0.2, timm 1.0.30; the Kaggle image runs Python 3.13 as of 2026-09).
Attach this dataset plus the competition data to notebooks/rsna_knee_image_model_inference.ipynb, internet OFF.
"""

if __name__ == "__main__":
    main(sys.argv[1].split(","))
