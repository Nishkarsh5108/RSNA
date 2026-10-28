"""Execute the generated Kaggle image notebooks locally (Windows), each in its own fresh interpreter:
  1. training notebook in SMOKE mode (40 studies, header index, cache, 1 tiny epoch, packaging)
  2. inference notebook on the 3 local test studies using the checkpoint produced by step 1
Local substitutions: /kaggle paths -> local dirs; ProcessPoolExecutor -> ThreadPoolExecutor
(Windows 'spawn' cannot pickle functions defined inside a notebook namespace; Kaggle/Linux uses fork).
Options (environment): NB_TEST_PREPROC=v6 (train the smoke model on v6 preprocessing), NB_TEST_TTA=1,
NB_TEST_SMALL=1 (K=2, batch 1: low VRAM, for running beside a training job),
NB_TEST_MODEL_DIR=<folder> (skip the training notebook; run the inference notebook with the knee_*.pt in that folder),
NB_TEST_INFER_NB=<file name in notebooks/> (default rsna_knee_image_model_inference.ipynb).
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

R = "D:/Coding/kaggle/RSNA"
W = "D:/rsna_work/tmp/nb_image_test"


def cells(nb):
    return ["".join(c["source"]) for c in json.load(open(nb, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def run_nb(path, overrides, name):
    ns, log = {"__name__": "__nbtest__"}, []
    for i, src in enumerate(cells(path)):
        for marker, repl in overrides.items():
            if marker in src:
                src = repl(src)
        src = src.replace("from concurrent.futures import ProcessPoolExecutor",
                          "from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor")
        t0 = time.time()
        try:
            exec(compile(src, f"{name}_cell{i}", "exec"), ns)
            ns["ProcessPoolExecutor"] = ThreadPoolExecutor
            st = "ok"
        except Exception as ex:
            traceback.print_exc()
            st = f"ERROR {type(ex).__name__}: {str(ex)[:200]}"
        log.append((i, src.strip().splitlines()[0][:60], st, round(time.time() - t0, 1)))
        print(f"[{name} {i}] {st} ({time.time()-t0:.1f}s)", flush=True)
        if st != "ok":
            break
    return ns, log


def run_train():
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(f"{W}/work")
    extra = f'RUN["preproc"] = "{os.environ["NB_TEST_PREPROC"]}"\n' if os.environ.get("NB_TEST_PREPROC") else ""
    if os.environ.get("NB_TEST_SMALL") == "1":        # ~1.2 GB VRAM: can run next to a training job (mechanics only)
        extra += 'RUN["K"] = 2\nRUN["bs"] = 1\nRUN["accum"] = 1\n'
    train_cfg = lambda s: s.replace('assert LABELS_CSV,', 'pass  #') + (
        f'\nCOMP_DIR = r"{R}/data/Raw_dicom"\nLABELS_CSV = r"{R}/data/labels_v3/training_labels_v3.csv"\n'
        f'WORK = r"{W}/work"\nCACHE = r"{W}/cache"\nSMOKE = True\nRUN["workers"] = 0\nRUN["run_id"] = "nbsmoke"\n' + extra)
    ns, log1 = run_nb(f"{R}/notebooks/rsna_knee_image_model_training.ipynb", {"# ---------------- CONFIG": train_cfg}, "train")
    ck = f"{W}/work/knee_nbsmoke.pt"
    print("training notebook produced checkpoint:", os.path.exists(ck), "| light zip:", os.path.exists(f"{W}/work/rsna_model_artifacts_light.zip"))
    print("SUMMARY train:"); [print(r) for r in log1]
    return all(r[2] == "ok" for r in log1) and os.path.exists(ck)


def run_infer():
    # fresh per-run folders (Windows may keep handles on a previous run's files): models + decode cache
    stamp = time.strftime("%H%M%S")
    MD, CD = f"{W}/models_{stamp}", f"{W}/infer_cache_{stamp}"
    if os.path.exists(f"{W}/work/submission.csv"):
        os.remove(f"{W}/work/submission.csv")             # a failed run must not leave a previous submission behind
    os.makedirs(MD)
    if os.environ.get("NB_TEST_MODEL_DIR"):          # test the real release: every knee_*.pt in that folder
        for f in os.listdir(os.environ["NB_TEST_MODEL_DIR"]):
            if f.startswith("knee_") and f.endswith(".pt"):
                shutil.copy(os.path.join(os.environ["NB_TEST_MODEL_DIR"], f), f"{MD}/{f}")
    else:
        shutil.copy(f"{W}/work/knee_nbsmoke.pt", f"{MD}/knee_nbsmoke.pt")
    print("models under test:", sorted(os.listdir(MD)))
    tta = os.environ.get("NB_TEST_TTA", "0") == "1"
    infer_cfg = lambda s: s.replace('assert MODEL_DIR is not None', '#') + (
        f'\nCOMP_DIR = r"{R}/data/Raw_dicom"\nCFG.update(test_csv=COMP_DIR + "/test.csv", test_series_csv=COMP_DIR + "/test_series.csv", '
        f'test_dicom_root=COMP_DIR + "/test_series", sample_submission=COMP_DIR + "/sample_submission.csv", model_dir=r"{MD}", '
        f'cache_dir=r"{CD}", out_path=r"{W}/work/submission.csv", workers=2, loader_workers=0, tta_shift={tta})\n'
        + (f'CFG["select_models"] = {os.environ["NB_TEST_SELECT"]}\n' if os.environ.get("NB_TEST_SELECT") else ""))
    nb_name = os.environ.get("NB_TEST_INFER_NB", "rsna_knee_image_model_inference.ipynb")
    print("inference notebook under test:", nb_name)
    ns2, log2 = run_nb(f"{R}/notebooks/{nb_name}", {"# ---------------- CONFIG": infer_cfg}, "infer")
    print("SUMMARY infer:"); [print(r) for r in log2]
    return all(r[2] == "ok" for r in log2)


def main():
    """Each notebook runs in its own fresh interpreter (as on Kaggle): the dependency cell requires
    that pydicom has not been imported yet."""
    if len(sys.argv) > 1:
        sys.exit(0 if (run_train() if sys.argv[1] == "train" else run_infer()) else 1)
    steps = ("infer",) if os.environ.get("NB_TEST_MODEL_DIR") else ("train", "infer")
    if steps == ("infer",):
        os.makedirs(f"{W}/work", exist_ok=True)
    rc = [subprocess.run([sys.executable, os.path.abspath(__file__), step]).returncode for step in steps]
    print("ALL OK" if all(r == 0 for r in rc) else f"FAILURES PRESENT (exit codes {dict(zip(steps, rc))})")
    sys.exit(0 if all(r == 0 for r in rc) else 1)


if __name__ == "__main__":
    main()
