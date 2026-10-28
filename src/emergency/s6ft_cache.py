"""Plan section 13, Track B: compact training cache for the s6fall fine-tune (s6ft), built locally.

Per non-gold training study, the exact s6fall inference preprocessing (heliosli's compact_infer.CStudies: SKM-TEA knee
anchors, views v4, 320-px rows, 3 adjacent slices 3.5 mm apart, every slice a window centre) -- the same call that scored
s6fall locally within 0.0003 of Kaggle (task1_compact_preds.py, leg s6fall). From the study's windows a fixed random subset
of K (seed: the study's sha1) is kept, or every window with --k 0; each window channel is stored as a JPEG (quality Q) with
the window's meta row. The trainer draws 64 of the K per epoch and random-crops 288 px, as nartaa's trainer does from the
full set (--k 0 reproduces that exactly; about 149 GB at quality 90).

gold_58 is never rendered (their images are not needed for training). Labels are not read here.

Usage: python -I src/emergency/s6ft_cache.py [--limit N] [--workers 4] [--k 96] [--quality 95] [--gpu-workers]
Writes D:/rsna_work/s6ft_cache/studies/<sha16>.npz (resumable) and index.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
SRC = "D:/rsna_work/public_code/s6_src"
CODE = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/cloud_code"
ANATOMY = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt"
DATA = f"{R}/data/Raw_dicom"
OUT = "D:/rsna_work/s6ft_cache"


def key(uid):
    return hashlib.sha256(uid.encode()).hexdigest()[:16]


def training_uids():
    import pandas as pd
    tr = pd.read_csv(f"{DATA}/train.csv", dtype={"StudyInstanceUID": str})
    gold = set(pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID)
    on_disk = set(os.listdir(f"{DATA}/train_series"))
    uids = sorted(u for u in tr.StudyInstanceUID.unique() if u not in gold and u in on_disk)
    assert not (set(uids) & gold) and len(gold) == 58
    return uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=96, help="windows kept per study (0: every window)")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--gpu-workers", action="store_true", help="loader workers run the anchor V-Net on the GPU (default: CPU)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--tmp", default="D:/rsna_work/tmp_task1/s6ft_cache")
    ap.add_argument("--shard", default=None, help="i/n: only studies with index %% n == i (parallel runs)")
    ap.add_argument("--timeout", type=int, default=600, help="seconds the loader may wait for one study before skipping it")
    a = ap.parse_args()
    os.environ["PYTHONPATH"] = os.pathsep.join([SRC, CODE, os.environ.get("PYTHONPATH", "")])
    sys.path[:0] = [SRC, CODE]
    if not a.gpu_workers:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # "-1", not "": an empty value deletes the variable on Windows
    import cv2
    import torch
    import compact_infer as CI
    torch.set_num_threads(1)
    out = Path(a.out) / "studies"
    out.mkdir(parents=True, exist_ok=True)
    uids = training_uids()
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        uids = [u for j, u in enumerate(uids) if j % n == i]
    todo = [u for u in uids if not (out / (key(u) + ".npz")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print(f"s6ft cache: {len(uids)} training studies, {len(todo)} to render (k {a.k}, quality {a.quality}, workers {a.workers}, "
          f"{'GPU' if a.gpu_workers else 'CPU'} anchors)", flush=True)
    if not todo:
        return
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    index = Path(a.out) / ("index.jsonl" if not a.shard else f"index_{a.shard.replace('/', 'of')}.jsonl")
    t0 = time.time()
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), a.quality]
    total, n_done = len(todo), 0
    with open(index, "a", encoding="utf-8") as ix:
        while todo:
            ds = CI.CStudies(DATA, "train", todo, tmp, CODE, 320, 1, "v4", 3, ANATOMY, 3.5, None, 2.0, 0.0, 20.0, extent_meta=True)
            loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                                 prefetch_factor=2 if a.workers else None,
                                                 multiprocessing_context="spawn" if a.workers else None,
                                                 timeout=a.timeout if a.workers else 0)
            k = 0                                                      # studies delivered by this loader (in todo order)
            try:
                for item in loader:
                    k += 1
                    n_done += 1
                    uid = item["uid"]
                    rec = {"uid": uid, "error": item["error"], "seconds": item["seconds"], "anchor": item.get("anchor")}
                    imgs = item["images"]
                    if item["error"] or imgs is None or not len(imgs):
                        rec["error"] = rec["error"] or "no usable window"     # never zero-filled: the trainer skips the study
                        ix.write(json.dumps(rec) + "\n"); ix.flush()
                        print("FAILED", uid[-12:], str(rec["error"])[:160], flush=True)
                        continue
                    imgs = imgs.numpy() if hasattr(imgs, "numpy") else np.asarray(imgs)
                    meta = item["meta"].numpy() if hasattr(item["meta"], "numpy") else np.asarray(item["meta"])
                    assert imgs.dtype == np.uint8 and imgs.ndim == 4 and imgs.shape[1] == 3 and imgs.shape[2] == imgs.shape[3] == 320, imgs.shape
                    W = len(imgs)
                    rng = np.random.default_rng(int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16))
                    pick = np.arange(W) if a.k <= 0 or a.k >= W else np.sort(rng.choice(W, size=a.k, replace=False))
                    blobs, off = [], [0]
                    for w in pick:
                        for ch in range(3):
                            ok, b = cv2.imencode(".jpg", imgs[w, ch], enc)
                            assert ok
                            blobs.append(b.ravel())
                            off.append(off[-1] + len(b))
                    np.savez(out / (key(uid) + ".npz"), uid=uid, jpg=np.concatenate(blobs), off=np.asarray(off, np.int64),
                             meta=meta[pick].astype(np.float32), pick=pick.astype(np.int32), n_windows=np.int32(W))
                    rec.update({"n_windows": int(W), "kept": int(len(pick)), "bytes": int(off[-1]), "meta_cols": int(meta.shape[1])})
                    ix.write(json.dumps(rec) + "\n"); ix.flush()
                    if n_done <= 3 or n_done % 25 == 0:
                        el = time.time() - t0
                        print(f"{n_done}/{total} windows {W} kept {len(pick)} {off[-1] / 2 ** 20:.2f} MB | decode {item['seconds'][0]:.1f}s "
                              f"render {item['seconds'][1]:.1f}s | {el / n_done:.2f} s/study, ETA {(total - n_done) * el / n_done / 60:.0f} min",
                              flush=True)
                todo = []
            except RuntimeError as e:
                if "timed out" not in str(e):
                    raise
                stuck = todo[k]                                    # in-order delivery: the loader was waiting for this study
                ix.write(json.dumps({"uid": stuck, "error": f"loader timeout {a.timeout}s"}) + "\n"); ix.flush()
                print("STUCK", stuck[-12:], "skipped (a rerun retries it); loader restarted", flush=True)
                n_done += 1
                todo = todo[k + 1:]
            del loader
    print(f"done {total} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
