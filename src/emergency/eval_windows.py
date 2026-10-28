"""Inference-window variants of a fold-0 run, scored on the pseudo-val fold (experiments ALLW and HR).

Variants (each saved, so runs can be compared pairwise later by compare_preds.py):
  train   : K = the run's validation K (K_eval if set, else K), evenly spaced
  shift   : train + a second bag shifted by half a window step (the deployed TTA for v2 models)
  all     : every slice of the [lo, hi] span is a window centre (K = 0, padded + window mask)
  kN      : N evenly spaced windows per slot, one bag (efficiency trade-off)
Usage: python src/emergency/eval_windows.py RUN_ID [--variants train,shift,all] [--bs 2]
Writes output/emergency_research/preds/<RUN_ID>__<variant>.npz and prints macro AUC + timing.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

R = "D:/Coding/kaggle/RSNA"
sys.path.insert(0, f"{R}/src/imaging")
sys.path.insert(0, f"{R}/src/labeling")
from data import KneeStudyDataset, pad_collate  # noqa: E402
from metrics import safe_auc  # noqa: E402
from model import KneeMIL  # noqa: E402
from study_index import TARGETS  # noqa: E402

OUT = f"{R}/output/emergency_research/preds"


@torch.no_grad()
def run_bag(model, va, a, K, lo, hi, bs, dev="cuda"):
    slots = [int(s) for s in str(a["slots"]).split(",")]
    ds = KneeStudyDataset(va, a["cache"], slots, K, train=False, lo=lo, hi=hi, orient=not a.get("no_orient", False),
                          sag_pos_fix=a.get("sag_pos_fix", False), crop=a.get("crop", 0), k_max=a.get("k_max", 64))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs if K > 0 else 1, num_workers=4,
                                     collate_fn=pad_collate if K == 0 else None)
    out = np.zeros((len(ds), len(TARGETS)), np.float32)
    n_win = 0
    for b in dl:
        x = b["x"].to(dev).float().div_(255)
        wm = b["wmask"].to(dev) if "wmask" in b else None
        n_win += int(wm.sum()) if wm is not None else int(b["mask"].sum()) * x.shape[2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x, b["mask"].to(dev), b["pos"].to(dev), wm)
        out[b["i"].numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
    return out, n_win / len(ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--variants", default="train,shift,all")
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ckpt", default="ckpt_best.pt")
    ap.add_argument("--crop", type=int, default=-1, help="override the inference crop (hr: 352 = the full stored field)")
    ap.add_argument("--tag", default="", help="suffix for the saved prediction-set names")
    c = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rd = f"{R}/output/imaging_v5/runs/{c.run_id}"
    ck = torch.load(f"{rd}/{c.ckpt}", map_location="cpu", weights_only=False)
    a = ck["args"]
    if c.crop >= 0:
        a["crop"] = c.crop
    assert not a.get("train_all_folds"), "fold-0 evaluation needs a holdout run"
    m = KneeMIL(a["backbone"], pretrained=False, n_slots=6, agg=a["agg"], use_pos=not a.get("no_pos", False))
    m.load_state_dict(ck["model"]); m.eval().cuda()
    idx = pd.read_csv(a["index"])
    va = idx[idx.fold == a["val_fold"]].reset_index(drop=True)
    lab = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv").set_index("StudyInstanceUID")
    Y = (lab.loc[va.StudyInstanceUID, TARGETS].values >= 0.5).astype(float)
    lo, hi = a["lo"], a["hi"]
    k_tr = a.get("K_eval", -1)
    k_tr = a["K"] if k_tr is None or k_tr < 0 else k_tr
    half = (hi - lo) / (2 * (k_tr - 1))
    bags = {"train": [(k_tr, lo, hi)], "shift": [(k_tr, lo, hi), (k_tr, lo + half, min(1.0, hi + half))], "all": [(0, lo, hi)]}
    for v in c.variants.split(","):                      # "kN" = N evenly spaced windows per slot, one bag
        if v.startswith("k") and v[1:].isdigit():
            bags[v] = [(int(v[1:]), lo, hi)]
    done = {}
    for v in c.variants.split(","):
        t0, ps, nw = time.time(), [], 0
        for bag in bags[v]:
            if bag not in done:
                done[bag] = run_bag(m, va, a, *bag, bs=c.bs)
            ps.append(done[bag][0]); nw += done[bag][1]
        p = np.mean(ps, 0)
        per = [safe_auc(Y[:, j], p[:, j]) for j in range(len(TARGETS))]
        np.savez(f"{OUT}/{c.run_id}__{v}{c.tag}.npz", uids=va.StudyInstanceUID.values, p=p, windows_per_study=nw,
                 sec=time.time() - t0)
        print(f"{c.run_id}{c.tag} {v:6s} macro {np.nanmean(per):.4f} | windows/study {nw:.1f} | {time.time()-t0:.0f}s | "
              + " ".join(f"{t[:6]}={x:.3f}" for t, x in zip(TARGETS, per)), flush=True)


if __name__ == "__main__":
    main()
