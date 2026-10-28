"""Generate the Kaggle training and inference notebooks from the tested modules in src/imaging,
so the notebooks run exactly the code that was validated locally (no hand-copied drift)."""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "imaging")
LAB = os.path.join(HERE, "..", "src", "labeling")


def module_body(path, drop_main=True):
    s = open(path, encoding="utf-8").read()
    if drop_main and '\nif __name__ == "__main__":' in s:
        s = s.split('\nif __name__ == "__main__":')[0]
    # local-package imports become no-ops: everything lives in the notebook namespace
    s = re.sub(r"^from (study_index|infer_lib|data|model|predict|metrics|train) import .*$", "", s, flags=re.M)
    s = re.sub(r"^sys\.path\.insert\(0, .*$", "", s, flags=re.M)
    return s


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}


DISCOVER = r'''
import glob, os, subprocess, sys, json, time, shutil
def find_dir(pattern_files, roots=("/kaggle/input",)):
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", pattern_files[0]), recursive=True):
            d = os.path.dirname(p)
            if all(os.path.exists(os.path.join(d, f)) for f in pattern_files):
                return d
    return None
def pip_offline(pkgs):
    whl_dirs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)})
    if not whl_dirs:
        print("no wheel directories attached"); return False
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "-q"] + sum([["--find-links", d] for d in whl_dirs], []) + pkgs
    print("installing", pkgs, "from", whl_dirs); return subprocess.run(cmd).returncode == 0
'''

DEPS = r'''
# Offline dependencies. Install BEFORE pydicom is first imported: pixel-decoder plugins are discovered
# at import time. Wheels in the attached dataset cover CPython 3.10-3.14 (Kaggle image: Python 3.13).
from importlib.metadata import version, PackageNotFoundError
def _ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None
print("python", sys.version.split()[0], "| pydicom", _ver("pydicom"), "| pylibjpeg", _ver("pylibjpeg"),
      "| libjpeg", _ver("pylibjpeg-libjpeg"), "| openjpeg", _ver("pylibjpeg-openjpeg"), "| gdcm", _ver("python-gdcm"), "| timm", _ver("timm"))
assert "pydicom" not in sys.modules, "pydicom was imported before the dependency cell: restart the kernel and run from the top"
PYDICOM_VERSION, TIMM_VERSION = "3.0.2", "1.0.30"          # versions used locally (timm: training)
if _ver("pydicom") is None or int(_ver("pydicom").split(".")[0]) < 3:
    pip_offline([f"pydicom=={PYDICOM_VERSION}", "--no-deps"])
missing = [p for p in ("pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg") if _ver(p) is None]
if missing:
    pip_offline(missing)
if _ver("timm") != TIMM_VERSION:
    print(f"timm {_ver('timm')} != {TIMM_VERSION}: installing the bundled wheel")
    pip_offline([f"timm=={TIMM_VERSION}", "--no-deps"])
import pydicom, timm
from pydicom.pixels import get_decoder
from pydicom.uid import JPEGLosslessSV1, JPEGLossless, JPEG2000Lossless, JPEG2000
ok = {u.name: get_decoder(u).is_available for u in (JPEGLosslessSV1, JPEGLossless, JPEG2000Lossless, JPEG2000)}
print("pydicom", pydicom.__version__, "| timm", timm.__version__, "| decoders", ok)
assert all(ok.values()), "JPEG-Lossless / JPEG-2000 decoders unavailable: attach the dataset with the wheels (internet is off)"
assert timm.__version__ == TIMM_VERSION, f"timm {timm.__version__} loaded; restart the kernel after installation"
'''

INFER_CONFIG = r'''
# ---------------- CONFIG (edit only if auto-discovery fails) ----------------
COMP_DIR = find_dir(["test.csv", "test_series.csv", "sample_submission.csv"]) or "/kaggle/input/rsna-knee-abnormality-detection"
MODEL_DIR = None        # folder with the trained *.pt checkpoints (auto: any attached folder containing knee_*.pt)
TTA_SHIFT = True        # adopted: +0.0012 (E1) and +0.0017 [+0.0001, +0.0033] (E1p) pseudo-val; see experiments report
FAST_INDEX = True       # one header per series: identical study index on 450 checked studies, preprocessing -42% (efficiency report)
SELECT_MODELS = None    # None = every knee_*.pt in MODEL_DIR (weights as packaged); or {"knee_<run>.pt": weight, ...} to use a subset
if MODEL_DIR is None:
    cands = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/knee_*.pt", recursive=True)})
    MODEL_DIR = cands[0] if cands else None
CFG = dict(test_csv=f"{COMP_DIR}/test.csv", test_series_csv=f"{COMP_DIR}/test_series.csv",
           test_dicom_root=f"{COMP_DIR}/test_series", sample_submission=f"{COMP_DIR}/sample_submission.csv",
           model_dir=MODEL_DIR, cache_dir="/tmp/knee_cache", out_path="/kaggle/working/submission.csv",
           workers=max(1, (os.cpu_count() or 2)), batch_size=4, device="cuda",
           tta_shift=TTA_SHIFT,     # average a 2nd window bag shifted by half a window step (see the experiments report)
           fast_index=FAST_INDEX, select_models=SELECT_MODELS)
print(json.dumps(CFG, indent=1))
assert MODEL_DIR is not None, "attach the dataset that contains the knee_*.pt checkpoints"
'''


def build_inference():
    cells = [
        md("""
# RSNA Knee: image-model inference (submission notebook)

Runs the model trained by `src/imaging/train.py` (2.5D ConvNeXt multi-slot MIL) on the hidden test set.
**Internet must be OFF.** Accelerator: **GPU T4** (bf16 is used on Ampere+ GPUs, fp16 otherwise; recent PyTorch builds may not
support the P100). Attach two inputs:
1. the competition data;
2. a private dataset containing the `knee_*.pt` checkpoint(s) and `wheels/` (pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg for
   CPython 3.10–3.14, pydicom 3.0.2, timm 1.0.30). Every `knee_*.pt` in that folder is ensembled with equal weight.

Preprocessing is the training preprocessing recorded in each checkpoint (v2 224 px or v6 256 px at 0.6 mm/px; v2 verified bit-identical
to the training cache on 51 local series, v6 is produced by the same function that built the training cache). The run **fails loudly** if decoders are missing or decoding
fails systematically (> 2% of studies). An undecodable series falls back to another series of the same plane or its slot is explicitly
masked; one undecodable slice at either end of a series (≥ 10 slices) is dropped and reported. An image is never silently
zero-filled. Every step is logged to `/kaggle/working/inference_report.json`.
"""),
        md("## 1. Environment, discovery and offline dependencies"),
        code(DISCOVER),
        code("import torch\nprint('torch', torch.__version__, '| CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\nassert torch.cuda.is_available(), 'enable a GPU accelerator'"),
        code(DEPS),
        md("## 2. Configuration"),
        code(INFER_CONFIG),
        md("## 3. Code (generated from `src/imaging/*.py`; identical to the locally tested modules)"),
        code("import os, sys, math, time, json\nimport numpy as np, pandas as pd, torch, torch.nn as nn\n" + module_body(os.path.join(SRC, "study_index.py"))),
        code(module_body(os.path.join(SRC, "infer_lib.py"))),
        code(module_body(os.path.join(SRC, "data.py"))),
        code(module_body(os.path.join(SRC, "model.py"))),
        code(module_body(os.path.join(SRC, "predict.py"))),
        md("## 4. Run: decode self-test, index, selected-series cache, ensemble prediction, validated submission"),
        code("t_all = time.time()\nsub, report = run(CFG)\nprint('total minutes', round((time.time() - t_all) / 60, 1))\nsub.head()"),
        md("## 5. Final checks"),
        code("""
s = pd.read_csv(CFG["out_path"])
smp = pd.read_csv(CFG["sample_submission"])
tst = pd.read_csv(CFG["test_csv"])
assert list(s.columns) == list(smp.columns)
assert len(s) == len(tst) and set(s.StudyInstanceUID) == set(tst.StudyInstanceUID)
v = s.iloc[:, 1:].values
assert np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
print("submission OK:", s.shape, "| decode failures:", len(report["decode_failures"]), "| header failures:", len(report["header_failures"]))
"""),
    ]
    return cells


TRAIN_CONFIG = r'''
# ---------------- CONFIG ----------------
COMP_DIR = find_dir(["train.csv", "train_series.csv"]) or "/kaggle/input/rsna-knee-abnormality-detection"
LABELS_CSV = (glob.glob("/kaggle/input/**/training_labels_v3.csv", recursive=True) or [None])[0]   # private dataset you upload
WORK = "/kaggle/working"
RUN = dict(run_id="kaggle_final", backbone="convnext_tiny.fb_in22k_ft_in1k", slots="0,1,2", K=8, agg="attn12",
           preproc="v2",                             # "v2" = 224 px squash, "v6" = 0.6 mm/px aspect-preserving 256 px crop
           epochs=8, bs=2, accum=2, lr_bb=1e-4, lr_head=5e-4, gold_weight=0.0, train_all_folds=False, val_fold=0,
           seed=42, workers=4, time_limit_min=0,     # time_limit_min>0: stop cleanly and resume in a new session
           extra_args=["--sag-pos-fix"])             # final recipe (F1-F3); add other train.py flags here
# Final models F1-F3 = these defaults with train_all_folds=True and seeds 42 / 7 / 11 (one run each).
CACHE = (f"/kaggle/tmp/knee_cache_{RUN['preproc']}" if os.path.isdir("/kaggle/tmp") else f"/tmp/knee_cache_{RUN['preproc']}")
SMOKE = False   # True: tiny run (cache 40 studies, 1 epoch) to check the notebook end to end
print(COMP_DIR, LABELS_CSV, CACHE)
assert LABELS_CSV, "attach a private dataset containing training_labels_v3.csv"
'''


def build_training():
    cells = [
        md("""
# RSNA Knee: image-model training (Kaggle)

Trains the same 2.5D multi-slot MIL model as `src/imaging/train.py`.
- **Step 1** rebuilds the study index from DICOM headers: slot choice, knee side (tag first, else coronal/axial geometry), and the 2.5D step.
- **Step 2** caches only the selected series with the exact training preprocessing (`RUN["preproc"]`: v2 224 px or v6 256 px at 0.6 mm/px).
- **Step 3** trains with pseudo-labelled validation (fold 0) and saves sealed gold predictions. Each epoch writes a checkpoint, so the run is resumable.
- **Step 4** packages artefacts into `/kaggle/working/rsna_artifacts` plus zip archives.

Attach the competition data and a **private** dataset holding `training_labels_v3.csv`; never publish it. Internet is only needed if
timm pretrained weights are not attached. Tested locally: construction, forward, loss, checkpoint save/load, packaging.
Untested: the full Kaggle run (no Kaggle credentials on the dev machine).
"""),
        md("## 1. Environment and dependencies"),
        code(DISCOVER),
        code("import torch\nprint('torch', torch.__version__, '| GPUs', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\nassert torch.cuda.is_available()"),
        code(DEPS),
        md("## 2. Configuration"),
        code(TRAIN_CONFIG),
        md("## 3. Code (generated from `src/imaging/*.py` and `src/labeling/metrics.py`)"),
        code("import os, sys, math, time, json, argparse\nimport numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F\n" + module_body(os.path.join(SRC, "study_index.py"))),
        code(module_body(os.path.join(LAB, "metrics.py"))),
        code(module_body(os.path.join(SRC, "infer_lib.py"))),
        code(module_body(os.path.join(SRC, "data.py"))),
        code(module_body(os.path.join(SRC, "model.py"))),
        code(module_body(os.path.join(SRC, "train.py"))),
        md("## 4. Build the train study index from DICOM headers (≈10–20 min on 4 CPUs)"),
        code(r'''
ts = pd.read_csv(f"{COMP_DIR}/train_series.csv")
if SMOKE:
    ts = ts[ts.StudyInstanceUID.isin(ts.StudyInstanceUID.drop_duplicates().head(40))]
def _geo(a):
    sdir, sid = a
    try:
        return sid, series_header_geometry(sdir), None
    except Exception as ex:
        return sid, None, str(ex)
jobs = [(f"{COMP_DIR}/train_series/{r.StudyInstanceUID}/{r.SeriesInstanceUID}", r.SeriesInstanceUID) for r in ts.itertuples()]
geos, bad = {}, []
with process_pool(os.cpu_count()) as ex:
    for sid, g, e in ex.map(_geo, jobs, chunksize=16):
        (bad.append((sid, e)) if g is None else geos.__setitem__(sid, g))
rows = [build_index_row(st, m[m.SeriesInstanceUID.isin(geos)], geos) for st, m in ts.groupby("StudyInstanceUID")]
index = pd.DataFrame(rows)
labels = pd.read_csv(LABELS_CSV)
labels = labels[labels.StudyInstanceUID.isin(index.StudyInstanceUID)]
index = add_folds(index, labels, pd.Series(dtype=str), dup_groups=[])   # duplicate pair handled in the next cell
print("header failures:", len(bad), "| side sources:", index.side_source.value_counts().to_dict())
'''),
        md("Keep the pixel-identical duplicate pair (UIDs ending …23376711 / …31613209) in one fold."),
        code(r'''
pair = [u for u in index.StudyInstanceUID if u[-8:] in ("23376711", "31613209")]   # pixel-identical duplicate pair
if len(pair) == 2:
    f0 = index.loc[index.StudyInstanceUID == pair[0], "fold"].item()
    index.loc[index.StudyInstanceUID == pair[1], "fold"] = f0
os.makedirs(f"{WORK}/rsna_artifacts", exist_ok=True)
index.to_csv(f"{WORK}/rsna_artifacts/study_index_train.csv", index=False)
'''),
        md("## 5. Cache the selected series (exact training preprocessing, version `RUN['preproc']`)"),
        code(r'''
os.makedirs(CACHE, exist_ok=True)
slots = [int(s) for s in RUN["slots"].split(",")]
need = set()
for r in index.itertuples():
    for si in slots:
        u = getattr(r, f"{SLOT_NAMES[si]}_uid")
        if not isinstance(u, str) and si < 3:
            u = getattr(r, f"{SLOT_NAMES[si + 3]}_uid")
        if isinstance(u, str):
            need.add((r.StudyInstanceUID, u))
def _cache(a):
    st, sid = a
    p = f"{CACHE}/{sid}.npy"
    if os.path.exists(p):
        return sid, None
    try:
        sdir = f"{COMP_DIR}/train_series/{st}/{sid}"
        arr, _ = preprocess_series_v6(sdir) if RUN["preproc"] == "v6" else preprocess_series(sdir)
        np.save(p + ".tmp.npy", arr); os.replace(p + ".tmp.npy", p); return sid, None
    except Exception as ex:
        return sid, str(ex)
t0 = time.time(); fails = []
with process_pool(os.cpu_count()) as ex:
    for i, (sid, e) in enumerate(ex.map(_cache, sorted(need), chunksize=8)):
        if e: fails.append((sid, e))
        if i % 1000 == 0: print(i, "/", len(need), round(time.time() - t0), "s", flush=True)
print("cached", len(need) - len(fails), "series;", len(fails), "failures")
assert not fails, f"decode failures must be fixed, not skipped: {fails[:3]}"
'''),
        md("## 6. Train (resumable: re-running this cell continues from `ckpt_last.pt`)"),
        code(r'''
argv = ["--run-id", RUN["run_id"], "--index", f"{WORK}/rsna_artifacts/study_index_train.csv", "--labels", LABELS_CSV,
        "--cache", CACHE, "--preproc", RUN["preproc"], "--out-root", f"{WORK}/runs", "--backbone", RUN["backbone"], "--slots", RUN["slots"],
        "--K", str(RUN["K"]), "--agg", RUN["agg"], "--epochs", str(1 if SMOKE else RUN["epochs"]), "--bs", str(RUN["bs"]),
        "--accum", str(RUN["accum"]), "--lr-bb", str(RUN["lr_bb"]), "--lr-head", str(RUN["lr_head"]),
        "--gold-weight", str(RUN["gold_weight"]), "--val-fold", str(RUN["val_fold"]), "--seed", str(RUN["seed"]),
        "--workers", str(RUN["workers"]), "--time-limit-min", str(RUN["time_limit_min"]), "--resume"]
if RUN["train_all_folds"]: argv.append("--train-all-folds")
argv += list(RUN["extra_args"])
if SMOKE: argv += ["--max-train", "24"]
main(get_args(argv))
'''),
        md("## 7. Package artefacts (model, configs, metrics, predictions, code) for download"),
        code(r'''
import zipfile, platform
A = f"{WORK}/rsna_artifacts"; rd = f"{WORK}/runs/{RUN['run_id']}"
for f in ["config.json", "history.csv", "summary.json", "val_preds_best.csv", "gold_preds_SEALED.npz"]:
    if os.path.exists(f"{rd}/{f}"): shutil.copy(f"{rd}/{f}", A)
ck = torch.load(f"{rd}/ckpt_best.pt", map_location="cpu", weights_only=False)
torch.save(ck, f"{WORK}/knee_{RUN['run_id']}.pt")                 # the file the inference notebook loads
json.dump({"targets": TARGETS, "slots": RUN["slots"], "K": RUN["K"], "preprocessing": RUN["preproc"],
           "versions": {"torch": torch.__version__, "timm": timm.__version__, "python": platform.python_version()}},
          open(f"{A}/model_card.json", "w"), indent=1)
with zipfile.ZipFile(f"{WORK}/rsna_model_artifacts_light.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(A): z.write(f"{A}/{f}", f)
print("download from the Output tab:", f"{WORK}/knee_{RUN['run_id']}.pt", "and rsna_model_artifacts_light.zip")
'''),
    ]
    return cells


def write(cells, name):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    p = os.path.join(HERE, name)
    json.dump(nb, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("wrote", p, len(cells), "cells")


STANDALONE_MAIN = r'''
def _main():
    import argparse
    ap = argparse.ArgumentParser(description="RSNA knee image-model inference (standalone; runs in its own process)")
    ap.add_argument("--model-dir", default=None, help="folder with knee_*.pt (default: auto-discover under /kaggle/input)")
    ap.add_argument("--comp-dir", default=None)
    ap.add_argument("--out", default="/kaggle/working/submission_ours.csv")
    ap.add_argument("--no-tta-shift", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 2))
    ap.add_argument("--cache-dir", default="/tmp/knee_cache")
    c = ap.parse_args()
    comp = c.comp_dir or find_dir(["test.csv", "test_series.csv", "sample_submission.csv"]) or "/kaggle/input/rsna-knee-abnormality-detection"
    md_ = c.model_dir
    if md_ is None:
        cands = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/knee_*.pt", recursive=True)})
        md_ = cands[0] if cands else None
    assert md_ is not None, "no folder with knee_*.pt found"
    cfg = dict(test_csv=f"{comp}/test.csv", test_series_csv=f"{comp}/test_series.csv", test_dicom_root=f"{comp}/test_series",
               sample_submission=f"{comp}/sample_submission.csv", model_dir=md_, cache_dir=c.cache_dir, out_path=c.out,
               workers=c.workers, batch_size=4, device="cuda", tta_shift=not c.no_tta_shift, fast_index=True)
    print(json.dumps(cfg, indent=1), flush=True)
    t0 = time.time()
    sub, report = run(cfg)
    print("standalone inference done in", round((time.time() - t0) / 60, 1), "min ->", c.out, flush=True)


if __name__ == "__main__":
    _main()
'''


def build_standalone():
    """One .py file with the same code as the inference notebook, for running our models in a subprocess
    inside another notebook (e.g. before a public notebook, then rank-blending the two submissions)."""
    isolate = r'''
# Standalone mode: install the bundled wheels into a PRIVATE directory first on sys.path, so running this script inside
# another notebook never changes that notebook's own packages (e.g. its timm or pydicom version).
_KNEE_SITE = "/tmp/knee_site"
os.makedirs(_KNEE_SITE, exist_ok=True)
sys.path.insert(0, _KNEE_SITE)
def pip_offline(pkgs):
    whl_dirs = sorted({os.path.dirname(p) for p in glob.glob("/kaggle/input/**/*.whl", recursive=True)})
    if not whl_dirs:
        print("no wheel directories attached"); return False
    pkgs = [p for p in pkgs if p != "--no-deps"]
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "-q", "--no-deps", "--target", _KNEE_SITE] +           sum([["--find-links", d] for d in whl_dirs], []) + pkgs
    print("installing", pkgs, "into", _KNEE_SITE); ok = subprocess.run(cmd).returncode == 0
    import importlib; importlib.invalidate_caches(); return ok
'''
    parts = ["# Generated by notebooks/build_image_notebooks.py from src/imaging/*.py - do not edit by hand.",
             "import os, sys, math, time, json, glob, subprocess", DISCOVER, isolate, DEPS,
             "import numpy as np, pandas as pd, torch, torch.nn as nn",
             module_body(os.path.join(SRC, "study_index.py")), module_body(os.path.join(SRC, "infer_lib.py")),
             module_body(os.path.join(SRC, "data.py")), module_body(os.path.join(SRC, "model.py")),
             module_body(os.path.join(SRC, "predict.py")), STANDALONE_MAIN]
    p = os.path.join(HERE, "rsna_knee_infer_standalone.py")
    open(p, "w", encoding="utf-8").write((chr(10) * 2).join(parts))
    print("wrote", p)


if __name__ == "__main__":
    # 2026-10-18: the notebooks behind the 0.927 submission (rsna_knee_image_model_{inference,training}.ipynb, frozen copy in
    # D:/rsna_work/fallback_0927/notebooks) are never overwritten; the emergency versions get their own names.
    write(build_inference(), "rsna_knee_image_model_inference_emergency.ipynb")
    write(build_training(), "rsna_knee_image_model_training_emergency.ipynb")
    build_standalone()
