"""Task 1 (plan section 12): nartaa's Raptor CoAtNet 384 and 224 predictions on gold_58 + oof_300, run locally with
nartaa's own inference code exactly as the A-strict+ notebook embeds it (cells 3 and 6: raptor_run.py and
raptor224_run.py). Each script's source is executed into its own namespace with main() not called; the only edits are the
weight path (the local copy of nartaa/rsna-knee-publication-swa-weights-20261007; the scripts' own sha256 checks still
run) and the location of their _fastread.py helper. Nothing is trained; no label is read.

Per study: p_pair (plain + anatomical-mirror average, what the Kaggle stage outputs) and p_plain, for 384 and 224.
Usage: python -I src/emergency/task1_raptor_preds.py [--only-set gold_58|oof_300] [--limit N]
Writes output/emergency_research/task1/raptor/<sha16>.npz (resumable) and raptor_index.jsonl.
"""
import argparse
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

R = "D:/Coding/kaggle/RSNA"
NB = f"{R}/notebooks/candidate_A/a_strict_plus/rsna-knee-a-strict-plus.ipynb"
WDIR = "D:/rsna_work/public_weights/rsna-knee-publication-swa-weights-20261007"
DATA = f"{R}/data/Raw_dicom"
WORK = Path(f"{R}/output/emergency_research/task1")


def patch_fastread_for_windows(mod):
    """The fastread module reads files with os.open + os.pread (Linux). Windows has no os.pread (the AttributeError is
    swallowed and every slice comes back empty) and os.open without O_BINARY is text mode (CR/LF and ^Z mangling).
    Same bytes, Windows calls: binary flag, positioned read as lseek + read under one lock (fds may be shared by threads).
    check_reader.py (scratchpad) verified the result byte-identical to the module's reference reader v2 (pydicom paths)."""
    if os.name != "nt":
        return mod
    lock = threading.Lock()

    def pread(fd, n, off):
        with lock:
            os.lseek(fd, off, os.SEEK_SET)
            return os.read(fd, n)
    mod._O_FLAGS = os.O_RDONLY | os.O_BINARY
    mod._pread = pread
    return mod


def load_script(cell, tag):
    nb = json.load(open(NB, encoding="utf-8"))
    src = "".join(nb["cells"][cell]["source"])
    assert src.startswith("%%writefile"), cell
    src = src.split("\n", 1)[1]
    old = "Path('/kaggle/input/rsna-knee-publication-swa-weights-20261007') / _weight_name,"
    assert src.count(old) == 1
    src = src.replace(old, f"Path(r'{WDIR}') / _weight_name,")
    helper_dir = WORK / f"_raptor_{tag}"
    helper_dir.mkdir(parents=True, exist_ok=True)
    ns = {"__name__": f"raptor_{tag}", "__file__": str(helper_dir / f"raptor_{tag}.py")}
    exec(compile(src, f"raptor_{tag}.py", "exec"), ns)
    ns["find_weight_file"] = lambda fname: str(Path(WDIR) / fname)
    return ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--studies", default=f"{WORK}/task1_studies.csv")
    ap.add_argument("--only-set", default=None, choices=[None, "gold_58", "oof_300"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    out_dir = WORK / "raptor"
    out_dir.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(a.studies, dtype={"StudyInstanceUID": str})
    if a.only_set:
        st = st[st["set"] == a.only_set]
    key = lambda u: hashlib.sha256(u.encode()).hexdigest()[:16]
    st = st[[not (out_dir / (key(u) + ".npz")).exists() for u in st.StudyInstanceUID]]
    if a.limit:
        st = st.head(a.limit)
    uids = st.StudyInstanceUID.tolist()
    setof = dict(zip(st.StudyInstanceUID, st["set"]))
    print(f"raptor: {len(uids)} studies to score", flush=True)
    if not uids:
        return
    ns384, ns224 = load_script(3, "384"), load_script(6, "224")
    for k in ("IMG", "SLOTS", "FASTREAD_VARIANT", "K_EVAL", "CROP_MM", "MAXS"):
        assert ns384[k] == ns224[k], f"{k} differs between the 384 and 224 scripts: one volume cannot serve both"
    dev = "cuda:0"
    arms = []
    for tag, ns in (("384", ns384), ("224", ns224)):
        assert len(ns["ARMS"]) == 1, (tag, ns["ARMS"])
        arm = ns["ARMS"][0]
        model, res = ns["load_model"](ns["find_weight_file"](arm["file"]), arm["arch"], arm["res"], dev)
        arms.append((tag, ns, model, res))
        print(f"loaded {tag}: {arm['file']} res {res}", flush=True)
    ser = pd.read_csv(f"{DATA}/train_series.csv")                    # same dtypes as the Kaggle stage's test_series.csv read
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    tsdir = f"{DATA}/train_series"
    reader = patch_fastread_for_windows(ns384["_load_fastread"]()).make_builder(
        ns384["FASTREAD_VARIANT"], ns384["IMG"], ns384["SLOTS"], hdr_threads=ns384["FR_HDR"], px_threads=ns384["FR_PX"])
    q = queue.Queue(maxsize=4)

    def producer():
        for u in uids:
            t0 = time.perf_counter()
            try:
                vol, mask = reader(u, SER, tsdir)
                err = None
            except Exception as e:                       # the Kaggle stage scores such a study 0.5 everywhere
                vol, mask, err = None, None, f"{type(e).__name__}: {e}"[:200]
            q.put((u, vol, mask, err, time.perf_counter() - t0))
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t_start = time.time()
    n = 0
    with open(WORK / "raptor_index.jsonl", "a", encoding="utf-8") as ix:
        while True:
            item = q.get()
            if item is None:
                break
            u, vol, mask, err, read_s = item
            n += 1
            rec = {"uid": u, "set": setof[u], "error": err, "read_s": round(read_s, 2), "valid_slices": None}
            if err is None and not (mask > 0).any():
                err = "reader returned no valid slice"   # never score an empty volume as if it were the study
            if err is None:
                rec["valid_slices"] = int((mask > 0).sum())
                tg = time.perf_counter()
                save = {}
                for tag, ns, model, res in arms:
                    xw = ns["gpu_windows"](vol, mask, ns["K_EVAL"], res, dev)
                    xm = ns["gpu_windows"](vol, mask, ns["K_EVAL"], res, dev, norm=ns["NORM"], mirror=True)
                    p_plain = ns["infer_probs"](model, xw, dev)
                    p_mirror = ns["infer_probs"](model, xm, dev)
                    save[f"p{tag}_plain"] = p_plain.astype(np.float32)
                    save[f"p{tag}_pair"] = (0.5 * (p_plain + p_mirror)).astype(np.float32)   # = infer_pair_probs
                    del xw, xm
                np.savez_compressed(out_dir / (key(u) + ".npz"), uid=u, **save)
                rec["gpu_s"] = round(time.perf_counter() - tg, 2)
            ix.write(json.dumps(rec) + "\n"); ix.flush()
            if n <= 3 or n % 20 == 0:
                print(f"{n}/{len(uids)} {setof[u]} read {read_s:.1f}s gpu {rec.get('gpu_s')} err {err} | "
                      f"{(time.time() - t_start) / n:.1f} s/study", flush=True)
    print(f"done {n} in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
