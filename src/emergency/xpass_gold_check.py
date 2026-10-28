"""Fidelity check of compact_infer_x.py (build_compact_infer_x.py) on gold_58, before final-max v2 uses it.

run:     R6 exactly as r6_gold_preds.py runs it (v27's arguments), through compact_infer_x.py with s6fall as an extra model
         (--x-models E_prior_s42_fall.pt, --x-trainer = s6fall's own compact_train.py 14767c7a, 320 px rows, 288 crop,
         every window, as nartaa's stage runs it); --dump-logits keeps R6's per-fold logits.
compare: (1) R6 predictions.csv vs task1/r6 (the plain worker): must agree to float noise;
         (2) the in-pass s6fall vs s6fall's own pipeline (task1/s6fall, sigmoid of the plain logit): per-finding Spearman,
             max |dp|, and both gold AUCs; the five largest per-study differences and the side-check counts (a study the
             side classifier flips is the one known difference).
         Writes output/emergency_research/task1/r6x/xpass_check.json. Nothing here is fitted or selected on gold.
Usage: python -I src/emergency/xpass_gold_check.py run [--limit N] | compare
"""
import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6"
CODE = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/cloud_code"
ANATOMY = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt"
BUNDLE = f"{PW}/rsna-knee-r6-fe055406"
S6F = f"{PW}/rsna-knee-s6fall-weights/E_prior_s42_fall.pt"
S6F_TRAINER = "D:/rsna_work/public_code/s6_src/compact_train.py"
OUT = Path(f"{R}/output/emergency_research/task1/r6x")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


def run(a):
    m = json.loads(Path(BUNDLE, "manifest.json").read_text())
    assert hashlib.sha256(Path(S6F_TRAINER).read_bytes()).hexdigest().startswith("14767c7a8f0daa98"), "s6fall trainer differs"
    assert hashlib.sha256(Path(S6F).read_bytes()).hexdigest().startswith("0c7297d58f061ab4"), "s6fall weights differ"
    import pandas as pd
    st = pd.read_csv(f"{R}/output/emergency_research/task1/task1_studies.csv", dtype={"StudyInstanceUID": str})
    uids = st.loc[st["set"] == "gold_58", "StudyInstanceUID"].tolist()
    if a.limit:
        uids = uids[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "assignment.json").write_text(json.dumps(uids))
    args = ["--backbone", m["backbone"], "--size", str(m["size"]), "--crop", str(m["crop"]),
            "--eval-stride", str(m.get("eval_stride", 3)), "--views", m.get("views", "v1"),
            "--state-spec", f"{BUNDLE}/{m['state_spec']}",
            "--side-model", f"{BUNDLE}/{m['side_model']}", "--side-threshold", str(m.get("side_threshold", 2.0)),
            "--side-position-mm", str(m.get("side_position_mm", 20.0)), "--side-max-x", str(m.get("side_max_x", 0.0)),
            "--slices", str(m.get("slices", 3)), "--phys-mm", str(m.get("phys_mm", 0.0)),
            "--trainer-md5", m["trainer_md5"],
            "--fluid-stacker", f"{BUNDLE}/{m['fluid_stacker']}",
            "--flip-tta",
            "--state-stacker", f"{BUNDLE}/{m['state_stacker']}",
            "--anchor-weights", ANATOMY,
            "--dump-logits", str(out / "logits.json"),
            "--x-models", S6F, "--x-trainer", S6F_TRAINER, "--x-size", "320", "--x-crop", "288", "--x-eval-stride", "1",
            "--x-chunk", str(a.chunk)]
    argv = ["compact_infer_x.py", "--data", f"{R}/data/Raw_dicom", "--split", "train", "--code", CODE, "--models", BUNDLE,
            "--arm", m["arm"], "--assignment", str(out / "assignment.json"), "--output", str(out),
            "--tmp", "D:/rsna_work/tmp_task1/r6x", "--amp", "fp16", "--workers", str(a.workers), "--chunk", str(a.chunk)] + args
    (out / "argv.json").write_text(json.dumps(argv, indent=1))
    os.environ["PYTHONPATH"] = os.pathsep.join([HERE, SRC, CODE, os.environ.get("PYTHONPATH", "")])   # spawned loader workers
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path[:0] = [HERE, SRC, CODE]
    import torch
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(0.6)
    import compact_infer_x        # imported as a module: Windows spawns the loader workers, which import its classes by name
    sys.argv = argv
    compact_infer_x.main()


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def compare(a):
    import pandas as pd
    from scipy.stats import spearmanr
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    new = pd.read_csv(OUT / "predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    old = pd.read_csv(f"{R}/output/emergency_research/task1/r6/predictions.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    ids = [u for u in new.index if u in old.index]
    d6 = np.abs(new.loc[ids].values - old.loc[ids].values)
    x = pd.read_csv(OUT / "x_E_prior_s42_fall.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LAB]
    nat = {}
    for f in glob.glob(f"{R}/output/emergency_research/task1/s6fall/*.npz"):
        z = np.load(f)
        nat[str(z["uid"])] = 1 / (1 + np.exp(-z["z_plain"][0].astype(np.float64)))
    xi = [u for u in x.index if u in nat]
    xv = x.loc[xi].values
    nv = np.stack([nat[u] for u in xi])
    # locally the worker's final receipt step fails after every output is written: it hashes px_cache.py etc. next to
    # itself, and here compact_infer_x.py sits in src/emergency, not with heliosli's files (on Kaggle they share SRC)
    rec = json.loads((OUT / "receipt.json").read_text()) if (OUT / "receipt.json").exists() else {}
    per = {}
    g = gold.reindex(xi)
    for j, c in enumerate(LAB):
        y = g[c].values.astype(float)
        per[c] = {"spearman": round(float(spearmanr(xv[:, j], nv[:, j]).correlation), 5),
                  "max_abs_dp": round(float(np.abs(xv[:, j] - nv[:, j]).max()), 5),
                  "auc_inpass": round(auc(y, xv[:, j]), 4), "auc_native": round(auc(y, nv[:, j]), 4)}
    res = {"r6_studies_compared": len(ids), "r6_max_abs_dp": float(d6.max()) if len(ids) else None,
           "r6_mean_abs_dp": float(d6.mean()) if len(ids) else None,
           "s6fall_studies_compared": len(xi),
           "s6fall_macro_auc_inpass": round(float(np.nanmean([v["auc_inpass"] for v in per.values()])), 4),
           "s6fall_macro_auc_native": round(float(np.nanmean([v["auc_native"] for v in per.values()])), 4),
           "s6fall_max_abs_dp": round(float(np.abs(xv - nv).max()), 5),
           "s6fall_min_spearman": round(min(v["spearman"] for v in per.values()), 5),
           "side_check": rec.get("side_check"), "x_receipt": json.loads((OUT / "x_receipt.json").read_text()),
           "per_finding": per}
    big = sorted(((float(np.abs(xv[i] - nv[i]).max()), u) for i, u in enumerate(xi)), reverse=True)[:5]
    res["largest_s6fall_study_differences"] = [[round(d, 4), u] for d, u in big]
    (OUT / "xpass_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_finding", "x_receipt")}, indent=1))
    print(json.dumps(per))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    run(a) if a.mode == "run" else compare(a)
