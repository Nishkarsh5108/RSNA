"""Package a model release as a folder to upload as a PRIVATE Kaggle dataset (never touches older releases).

  python src/emergency/package_release.py --name rsna-knee-image-model-v3 \
      --model RUN_ID[:weight[:infer_K]] [--model ...]

  weight   relative ensemble weight (default 1); written to the checkpoint args as "ens_weight"
  infer_K  inference windows per slot: omit = training setting (+ shift TTA if the notebook enables it),
           0 = every slice of the span (one bag); written as "infer_K"

Creates D:/rsna_work/kaggle_upload/<name>/ with knee_<run>.pt, wheels/ (copied from the 0.927 release), the
standalone inference script, README.md and SHA256SUMS.txt. Refuses to write into an existing non-empty folder.
Checkpoints contain model weights and training arguments only (no images, reports or labels).
"""
import argparse
import hashlib
import json
import os
import shutil
import time

import torch

R = "D:/Coding/kaggle/RSNA"
RUNS = f"{R}/output/imaging_v5/runs"
UP_ROOT = "D:/rsna_work/kaggle_upload"
WHEELS = f"{UP_ROOT}/rsna-knee-image-model/wheels"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = f"{UP_ROOT}/{a.name}"
    assert a.name != "rsna-knee-image-model", "that folder is the 0.927 release (fallback C)"
    if os.path.isdir(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty; choose a new --name")
    os.makedirs(out, exist_ok=True)
    rows = []
    for spec in a.model:
        parts = spec.split(":")
        rid = parts[0]
        w = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
        ik = int(parts[2]) if len(parts) > 2 and parts[2] != "" else None
        ck = torch.load(f"{RUNS}/{rid}/ckpt_best.pt", map_location="cpu", weights_only=False)
        args = dict(ck["args"])
        args["ens_weight"] = w
        if ik is not None:
            args["infer_K"] = ik
        torch.save({"model": ck["model"], "args": args, "epoch": ck.get("epoch"), "val_macro_auc": ck.get("val_macro_auc")},
                   f"{out}/knee_{rid}.pt")
        rows.append({"file": f"knee_{rid}.pt", "backbone": args["backbone"], "preproc": args.get("preproc", "v2"),
                     "crop": args.get("crop", 0), "K_train": args["K"], "infer_K": args.get("infer_K", "train setting"),
                     "ens_weight": w, "labels": os.path.basename(args.get("labels", "")), "seed": args.get("seed"),
                     "all_folds": bool(args.get("train_all_folds")), "epoch": ck.get("epoch")})
    shutil.copytree(WHEELS, f"{out}/wheels")
    shutil.copy(f"{R}/notebooks/rsna_knee_infer_standalone.py", f"{out}/rsna_knee_infer_standalone.py")
    lines = [f"# {a.name}", "", f"Packaged {time.strftime('%Y-%m-%d %H:%M')} by src/emergency/package_release.py. "
             "Upload as a **private** Kaggle dataset; attach it as the only dataset containing `knee_*.pt`.", "",
             a.note, "", "| file | backbone | preproc | crop | K train | inference windows | weight | labels | seed | all folds | epoch |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['file']} | {r['backbone']} | {r['preproc']} | {r['crop']} | {r['K_train']} | {r['infer_K']} | {r['ens_weight']} | "
              f"{r['labels']} | {r['seed']} | {r['all_folds']} | {r['epoch']} |" for r in rows]
    lines += ["", "`wheels/`: pydicom 3.0.2, pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.6.0 (CPython 3.10-3.14), "
              "timm 1.0.30.", "", "`rsna_knee_infer_standalone.py`: the same inference code as the notebook, runnable as "
              "`python rsna_knee_infer_standalone.py --out /kaggle/working/submission_ours.csv` (e.g. before blending)."]
    open(f"{out}/README.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(f"{out}/SHA256SUMS.txt", "w") as f:
        for root, _, files in os.walk(out):
            for fn in sorted(files):
                if fn != "SHA256SUMS.txt":
                    p = os.path.join(root, fn)
                    f.write(f"{sha256(p)}  {os.path.relpath(p, out)}\n")
    print(json.dumps(rows, indent=1))
    print("release folder:", out)


if __name__ == "__main__":
    main()
