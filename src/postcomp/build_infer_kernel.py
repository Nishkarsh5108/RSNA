"""Build the inference notebook for the box-pooling family (side -> ROI detector -> classifiers -> submission.csv),
or lay the same files out in a local folder for a smoke test.

Module sources (side.py, detector.py, classifier.py) are taken from the cells of tascj0's three public notebooks
(Team Kangaroo, 1st place) and patched as text: the series folder becomes SERIES_ROOT, and the classifier gets the
same switches as in training (build_bp_kernel.patch). infer.py is src/postcomp/infer_bp.py.

  kernel:  python build_infer_kernel.py --slug knee-bp-infer-v1 --cls knee-bp-a1-f0/fold0_ema.pth
  local :  python build_infer_kernel.py --local-dir <folder>
"""
import argparse
import json
import os

from build_bp_kernel import patch as patch_classifier, sub

PUB = "D:/rsna_work/public_code/tascj"
NB = {"side.py": f"{PUB}/rsna-2026-knee-side-classifier/rsna-2026-knee-side-classifier.ipynb",
      "detector.py": f"{PUB}/rsna-2026-knee-roi-detector/rsna-2026-knee-roi-detector.ipynb",
      "classifier.py": f"{PUB}/rsna-2026-knee-box-pooling-classifier/rsna-2026-knee-box-pooling-classifier.ipynb"}
HERE = os.path.dirname(os.path.abspath(__file__))


def nb_source(nb_path, name):
    """The file a notebook builds with its '%%writefile name' / '%%writefile -a name' cells."""
    out = []
    for c in json.load(open(nb_path, encoding="utf-8"))["cells"]:
        s = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        first, _, rest = s.partition("\n")
        if c["cell_type"] == "code" and first.strip() in (f"%%writefile {name}", f"%%writefile -a {name}"):
            out.append(rest if rest.endswith("\n") else rest + "\n")
    assert out, f"no %%writefile {name} cells in {nb_path}"
    return "".join(out)


def modules():
    src = {}
    for name, path in NB.items():
        s = nb_source(path, name)
        if name == "classifier.py":
            s = patch_classifier(s)
        s = sub(s, 'OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))',
                'OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working"))\n'
                'SERIES_ROOT = Path(os.environ.get("SERIES_ROOT", COMP / "train_series"))')
        s = sub(s, 'COMP / "train_series" /', "SERIES_ROOT /")
        src[name] = s
    src["infer.py"] = open(f"{HERE}/infer_bp.py", encoding="utf-8").read()
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default="")
    ap.add_argument("--user", default="anujchaubey")
    ap.add_argument("--cls", default="", help="comma list 'kernel-slug/file.pth' of classifier weights (outputs of our training kernels)")
    ap.add_argument("--cls-kernels", default="", help="comma list of owner/slug kernels to attach (default: derived from --cls, owner = --user)")
    ap.add_argument("--out-root", default="D:/rsna_work/kaggle_runs")
    ap.add_argument("--local-dir", default="", help="write the four files here instead of building a kernel")
    ap.add_argument("--weights-dataset", default="", help="owner/slug of a private dataset that holds the classifier weights; --cls entries are then 'slug/file.pth'")
    ap.add_argument("--env", default="", help="extra environment for infer.py, 'KEY=VALUE;KEY=VALUE' (e.g. ATT_SHARP=2 or 12 comma-separated values)")
    ap.add_argument("--cpu", action="store_true", help="CPU session (dress rehearsal on the public test studies; uses no GPU quota)")
    a = ap.parse_args()
    src = modules()
    if a.local_dir:
        os.makedirs(a.local_dir, exist_ok=True)
        for name, s in src.items():
            open(f"{a.local_dir}/{name}", "w", encoding="utf-8", newline="\n").write(s)
        print("wrote", a.local_dir, list(src))
        return
    d = f"{a.out_root}/{a.slug}"
    os.makedirs(d, exist_ok=True)

    def cell(text):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}

    env = {"SIDE_WEIGHTS": "rsna-2026-knee-side-classifier/side_ema.pth", "DET_WEIGHTS": "rsna-2026-knee-roi-detector/detector_ema.pth",
           "CLS_WEIGHTS": a.cls}
    if a.cpu:
        env["TEST_FAIL_BATCH"] = "1"
    env.update(kv.split("=", 1) for kv in a.env.split(";") if kv)
    run = ("import os, subprocess, sys\n"
           'for root, dirs, files in os.walk("/kaggle/input"):          # mounted inputs, without the DICOM tree\n'
           '    dirs[:] = [d for d in dirs if d not in ("competitions", "rsna-knee-abnormality-detection")]\n'
           '    if root.count("/") <= 5:\n'
           '        print(root, sorted(files)[:8], flush=True)\n'
           f"env = dict(os.environ, **{env!r})\n"
           'r = subprocess.run([sys.executable, "infer.py"], env=env)\n'
           'print("exit code", r.returncode, flush=True)\n'
           "assert r.returncode == 0\n")
    credit = ("# Inference for a box-pooling knee MRI classifier. Modules (side classifier, ROI detector, box-pooling classifier) are taken\n"
              "# from the public notebooks of tascj0 (1st place, RSNA 2026 knee competition) with small patches; training labels come from the\n"
              "# public label sets of tascj0 (CC BY 4.0) and songqizhou (CC BY-NC-SA 4.0).\n")
    nb = {"cells": [cell(credit)] + [cell(f"%%writefile {n}\n{s}") for n, s in src.items()] + [cell(run)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(f"{d}/{a.slug}.ipynb", "w", encoding="utf-8"), indent=1)
    if a.weights_dataset:
        ours = []
    else:
        ours = a.cls_kernels.split(",") if a.cls_kernels else sorted({f"{a.user}/{c.split('/')[0]}" for c in a.cls.split(",") if c})
    meta = {"id": f"{a.user}/{a.slug}", "title": a.slug.replace("-", " "), "code_file": f"{a.slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": not a.cpu, "enable_internet": False,
            "dataset_sources": [a.weights_dataset] if a.weights_dataset else [], "competition_sources": ["rsna-knee-abnormality-detection"],
            "kernel_sources": ["tascj0/rsna-2026-knee-side-classifier", "tascj0/rsna-2026-knee-roi-detector"] + ours,
            "model_sources": []}
    json.dump(meta, open(f"{d}/kernel-metadata.json", "w"), indent=1)
    print("wrote", d, "| classifiers", a.cls, "| kernels", meta["kernel_sources"])


if __name__ == "__main__":
    main()
