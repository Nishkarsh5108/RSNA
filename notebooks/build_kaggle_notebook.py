"""Builds notebooks/rsna_knee_label_generation.ipynb from the tested local modules,
so the notebook runs the exact same labeler code (no hand-copied drift)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src", "labeling")
labeler_src = open(os.path.join(SRC, "llm_assertion_labeler.py"), encoding="utf-8").read()
metrics_src = open(os.path.join(SRC, "metrics.py"), encoding="utf-8").read()
# keep the module body but drop its CLI entry point
labeler_body = labeler_src.split("\ndef main():")[0]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md("""
# RSNA Knee: report → label generation (forced-choice assertion scoring)

Self-contained Kaggle notebook. It produces **per-finding assertion probabilities**
(A present / B possible / C not mentioned / D absent / E prior-only) for every
training report, using a local open-weight instruction model. Reports are never sent to an external service.

**What was tested locally (RTX 4050, 6 GB, Windows, torch 2.7.1, transformers 5.5.4, bitsandbytes 0.50.1):**
the labeler code in cell 6 (identical to `src/labeling/llm_assertion_labeler.py`), Qwen3-4B-Instruct-2507 in nf4, the
prefix-cache path (equivalent to the uncached path), and a full 4,407-report run.
**Not tested:** the Kaggle-specific path discovery and installation cells, T4/P100 timings, and models other than Qwen3-4B.

Competition-rule assumptions (see `reports/competition_rules_audit.md`): open-weight public models are allowed as
external data. Reports are training-only data. Internet use is only for installing packages and downloading weights in
**this offline-labeling notebook**, never in the submission notebook.
"""),
    md("## 1. Configuration (the only cell you normally edit)"),
    code("""
import os
CONFIG = {
    "MODEL_ID": "Qwen/Qwen3-4B-Instruct-2507",   # or a local path, e.g. /kaggle/input/qwen3-4b-instruct-2507
    "QUANT": "nf4",                # "nf4" (tested) | "none" (fp16/bf16, needs ~9 GB VRAM for 4B)
    "MAX_BATCH_TOKENS": 8000,      # raise to ~24000 on a 16 GB T4 with QUANT="none"
    "MAX_BATCH_SIZE": 4,
    "MAX_REPORT_TOKENS": 6000,     # reports longer than this are truncated AND flagged
    "SMOKE_TEST_N": 5,
    "RUN_FULL": True,              # set False to stop after the smoke test
    "OUT_DIR": "/kaggle/working/labels_v3" if os.path.isdir("/kaggle") else "./labels_v3_out",
}
print(CONFIG)
"""),
    md("## 2. Environment inspection"),
    code("""
import sys, shutil, platform, subprocess
import torch
print("python", sys.version.split()[0], "| torch", torch.__version__, "| platform", platform.platform())
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}  {p.total_memory/1e9:.1f} GB  bf16={torch.cuda.is_bf16_supported()}")
else:
    print("!! No GPU: a 4B model on CPU would take many hours; enable a GPU accelerator.")
total, used, free = shutil.disk_usage(".")
print(f"disk free: {free/1e9:.1f} GB")
try:
    import psutil; print(f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
except Exception:
    pass
"""),
    md("## 3. Dependencies (installs only if missing and internet is on)"),
    code("""
import importlib, socket
def has(mod):
    try:
        importlib.import_module(mod); return True
    except Exception:
        return False
def internet():
    try:
        socket.create_connection(("pypi.org", 443), timeout=3); return True
    except OSError:
        return False
need = [m for m in ["transformers", "accelerate", "bitsandbytes", "sklearn", "scipy"] if not has(m)]
print("missing:", need)
if need:
    if internet():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + [{"sklearn": "scikit-learn"}.get(m, m) for m in need], check=False)
    else:
        print("!! Internet is off and packages are missing: attach a wheels dataset and pip install from it.")
import transformers; print("transformers", transformers.__version__)
"""),
    md("## 4. Input discovery and data-integrity checks"),
    code("""
import glob
import numpy as np, pandas as pd
cands = glob.glob("/kaggle/input/**/train.csv", recursive=True) + glob.glob("../data/Raw_dicom/train.csv") + glob.glob("data/Raw_dicom/train.csv")
cands = [c for c in cands if "Report" in open(c, encoding="utf-8").readline()]
assert cands, "train.csv with a Report column not found -- attach the competition data"
TRAIN_CSV = cands[0]; print("using", TRAIN_CSV)
df = pd.read_csv(TRAIN_CSV)
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
print("studies", len(df), "| unique", df.StudyInstanceUID.nunique())
print("empty reports", int(df.Report.fillna("").str.strip().eq("").sum()))
gold_mask = df[TARGETS].notna().all(axis=1)
print("gold studies", int(gold_mask.sum()), "| unlabeled", int((~gold_mask).sum()))
assert df.StudyInstanceUID.is_unique
"""),
    md("## 5. Shared metrics (identical to `src/labeling/metrics.py`)"),
    code(metrics_src),
    md("## 6. Labeler (identical to `src/labeling/llm_assertion_labeler.py`, minus its CLI)"),
    code(labeler_body),
    md("## 7. Load the model"),
    code("""
os.makedirs(CONFIG["OUT_DIR"], exist_ok=True)
tok, model = load_model(CONFIG["MODEL_ID"], CONFIG["QUANT"], cache_dir=None)
scorer = Scorer(tok, model, CONFIG["MAX_REPORT_TOKENS"], prefix_cache=True)
print("prefix tokens:", len(scorer.prefix_ids), "| VRAM after load: %.2f GB" % (torch.cuda.memory_allocated()/1e9))
"""),
    md("## 8. Smoke test (non-gold reports only). Letter mass must be close to 1."),
    code("""
smoke = df[~gold_mask].head(CONFIG["SMOKE_TEST_N"])
enc = [scorer.prompt_ids(r)[0] for r in smoke.Report]
probs, mass = scorer.score_batch(enc)
print("letter mass min:", float(mass.min()))
assert mass.min() > 0.5, "model is not following the forced answer format -- check the chat template / model"
for i, uid in enumerate(smoke.StudyInstanceUID):
    print(uid[-12:], " ".join(f"{t[:6]}:{LETTERS[probs[i,k].argmax()]}" for k, t in enumerate(TARGETS)))
"""),
    md("## 9. Full run with checkpointing / resume (append-only CSV; re-running skips finished studies)"),
    code("""
import time
OUT_CSV = os.path.join(CONFIG["OUT_DIR"], "llm_assert_raw.csv")
done = set()
if os.path.exists(OUT_CSV):
    prev = pd.read_csv(OUT_CSV, usecols=["StudyInstanceUID", "status"])
    done = set(prev.loc[prev.status == "ok", "StudyInstanceUID"])
todo = df[~df.StudyInstanceUID.isin(done)].reset_index(drop=True)
print("already ok:", len(done), "| todo:", len(todo))
meta = {"model": CONFIG["MODEL_ID"], "quant": CONFIG["QUANT"], "prompt_version": PROMPT_VERSION}
if CONFIG["RUN_FULL"] and len(todo):
    enc = [scorer.prompt_ids(str(r)) for r in todo.Report]
    lengths = np.array([len(e[0]) + len(scorer.prefix_ids) for e in enc])
    batches = make_batches(lengths, CONFIG["MAX_BATCH_TOKENS"], CONFIG["MAX_BATCH_SIZE"])
    t0 = time.time(); n = 0
    for bi, idx in enumerate(batches):
        rows = []
        try:
            probs, mass = scorer.score_batch([enc[i][0] for i in idx])
            for j, i in enumerate(idx):
                r = {"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "ok", "error": "",
                     "n_prompt_tokens": int(lengths[i]), "report_truncated": bool(enc[i][1]), **meta}
                for k, t in enumerate(TARGETS):
                    for li, L in enumerate(LETTERS):
                        r[f"{t}__p{L}"] = round(float(probs[j, k, li]), 5)
                    r[f"{t}__letter_mass"] = round(float(mass[j, k]), 4)
                    r[f"{t}__argmax"] = LETTERS[int(probs[j, k].argmax())]
                if mass[j].min() < 0.5:
                    r["status"] = "low_letter_mass"
                rows.append(r)
        except Exception as ex:  # never silently drop a study
            torch.cuda.empty_cache()
            for i in idx:
                rows.append({"StudyInstanceUID": todo.StudyInstanceUID[i], "status": "failed",
                             "error": f"{type(ex).__name__}: {str(ex)[:200]}", **meta})
        pd.DataFrame(rows).to_csv(OUT_CSV, mode="a", index=False, header=not os.path.exists(OUT_CSV))
        n += len(idx)
        if bi % 25 == 0:
            el = time.time() - t0
            print(f"{n}/{len(todo)}  {n/el:.2f} studies/s  eta {(len(todo)-n)/(n/el)/60:.0f} min")
    print("finished in %.1f min" % ((time.time() - t0) / 60))
"""),
    md("## 10. Output validation (counts, duplicates, ranges, failures)"),
    code("""
out = pd.read_csv(OUT_CSV).drop_duplicates("StudyInstanceUID", keep="last")
print("rows", len(out), "| expected", len(df), "| missing", len(set(df.StudyInstanceUID) - set(out.StudyInstanceUID)))
print(out.status.value_counts().to_dict())
pc = [c for c in out.columns if "__p" in c]
assert out.loc[out.status == "ok", pc].notna().all().all(), "NaN probabilities in ok rows"
assert ((out[pc].fillna(0) >= 0) & (out[pc].fillna(0) <= 1)).all().all()
failed = out[out.status != "ok"][["StudyInstanceUID", "status", "error"]]
failed.to_csv(os.path.join(CONFIG["OUT_DIR"], "failed_or_flagged.csv"), index=False)
print("failed/flagged:", len(failed), "| truncated reports:", int(out.report_truncated.fillna(False).sum()))
"""),
    md("""## 11. Gold validation (information only: never tune prompts or thresholds on these numbers)
Score used: `s0 = pA + 0.5*pB + 0.25*pE` (fixed a priori)."""),
    code("""
g = df[gold_mask].set_index("StudyInstanceUID")[TARGETS].astype(float)
o = out.set_index("StudyInstanceUID").reindex(g.index)
S = pd.DataFrame({t: o[f"{t}__pA"] + 0.5 * o[f"{t}__pB"] + 0.25 * o[f"{t}__pE"] for t in TARGETS})
m = S.notna().all(axis=1)
mp, lo, hi = macro_auc_ci(g[m].values, S[m].values, n_boot=2000)
print(f"gold macro AUC {mp:.3f} [{lo:.3f}, {hi:.3f}] on n={int(m.sum())}")
for t in TARGETS:
    r = binary_report(g.loc[m, t].values, S.loc[m, t].values, n_boot=0)
    print(f"{t:18s} AUC {r['auc']:.3f}  sens {r['recall']:.2f}  spec {r['specificity']:.2f}  (pos {r['n_pos']})")
"""),
    md("## 12. Export"),
    code("""
import zipfile
zp = os.path.join(CONFIG["OUT_DIR"], "labels_v3_export.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in os.listdir(CONFIG["OUT_DIR"]):
        if f.endswith(".csv"):
            z.write(os.path.join(CONFIG["OUT_DIR"], f), f)
print("saved", zp, "-- download it from the Output tab (or commit the notebook to persist /kaggle/working).")
"""),
    md("""
## 13. Runtime / memory notes and troubleshooting
* Measured locally: about 0.55–0.6 studies/s on an RTX 4050 6 GB (nf4, batch ≈ 4). A T4 16 GB with `QUANT="none"`
  and `MAX_BATCH_TOKENS≈24000` should be faster; that estimate is untested.
* `letter mass` near 0 means the model does not follow the forced layout (wrong chat template or a thinking model).
  Use an *instruct* (non-thinking) model, or disable thinking in the template.
* OOM: lower `MAX_BATCH_TOKENS`. The code retries nothing silently: failed rows are written with `status=failed`;
  re-running the cell retries only those.
* Never feed any report-derived value into the image-model submission notebook. Test studies have no report.
"""),
]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out_path = os.path.join(HERE, "rsna_knee_label_generation.ipynb")
json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("wrote", out_path, "cells:", len(cells))
