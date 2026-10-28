"""Inference-time variants for a trained checkpoint, scored on the PSEUDO-val fold only (gold stays sealed).

Variants: number of 2.5D windows per series at test time (K), and averaging bags of windows
taken at shifted slice positions. The model pools over instances, so K can differ from training.
Usage: python eval_tta.py RUN_ID [--bs 2]
Writes reports/eda/imaging/tta_<RUN_ID>.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "labeling"))
from data import KneeStudyDataset  # noqa: E402
from metrics import paired_macro_auc_diff, safe_auc  # noqa: E402
from model import KneeMIL  # noqa: E402
from study_index import TARGETS  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


@torch.no_grad()
def run_variant(model, va, a, K, lo, hi, bs, dev="cuda"):
    slots = [int(s) for s in str(a["slots"]).split(",")]
    ds = KneeStudyDataset(va, a["cache"], slots, K, train=False, lo=lo, hi=hi, orient=not a.get("no_orient", False),
                              sag_pos_fix=a.get("sag_pos_fix", False))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4)
    out = np.zeros((len(ds), len(TARGETS)), np.float32)
    for b in dl:
        x = b["x"].to(dev).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x, b["mask"].to(dev), b["pos"].to(dev))
        out[b["i"].numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--bs", type=int, default=2)
    c = ap.parse_args()
    rd = f"{R}/output/imaging_v5/runs/{c.run_id}"
    ck = torch.load(f"{rd}/ckpt_best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = KneeMIL(a["backbone"], pretrained=False, n_slots=6, agg=a["agg"], use_pos=not a.get("no_pos", False))
    m.load_state_dict(ck["model"]); m.eval().cuda()
    idx = pd.read_csv(a["index"])
    va = idx[idx.fold == a["val_fold"]].reset_index(drop=True)
    lab = pd.read_csv(a["labels"]).set_index("StudyInstanceUID")
    Y = (lab.loc[va.StudyInstanceUID, TARGETS].values >= 0.5).astype(float)
    lo, hi, K0 = a["lo"], a["hi"], a["K"]
    half = (hi - lo) / (2 * (K0 - 1))                       # half the spacing between K0 window centres
    variants = {f"K{K0} (train setting)": [(K0, lo, hi)], "K12": [(12, lo, hi)], "K16": [(16, lo, hi)],
                f"K{K0} + K{K0} shifted half a step": [(K0, lo, hi), (K0, lo + half, min(1.0, hi + half))],
                f"K{K0} + K16": [(K0, lo, hi), (16, lo, hi)]}
    cache, rows, P = {}, [], {}
    for name, bags in variants.items():
        t0, ps = time.time(), []
        for bag in bags:
            if bag not in cache:
                cache[bag] = run_variant(m, va, a, *bag, bs=c.bs)
            ps.append(cache[bag])
        p = np.mean(ps, 0)
        P[name] = p
        per = [safe_auc(Y[:, j], p[:, j]) for j in range(len(TARGETS))]
        rows.append({"variant": name, "macro_auc": float(np.nanmean(per)), **dict(zip(TARGETS, per)),
                     "sec": round(time.time() - t0, 1)})
        print(f"{name:32s} {np.nanmean(per):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    base = next(iter(P))
    for r in rows:                                          # paired bootstrap vs the training setting
        if r["variant"] != base:
            d, dlo, dhi, ple0 = paired_macro_auc_diff(Y, P[r["variant"]], P[base], n_boot=1000, seed=9)
            r.update({"delta_vs_train_setting": d, "delta_lo": dlo, "delta_hi": dhi, "p_delta_le_0": ple0})
            print(f"{r['variant']:32s} delta {d:+.4f} [{dlo:+.4f}, {dhi:+.4f}] p(<=0) {ple0:.3f}", flush=True)
    np.savez(f"{R}/reports/eda/imaging/tta_preds_{c.run_id}.npz", uids=va.StudyInstanceUID.values,
             **{f"v{i}": p for i, p in enumerate(P.values())}, names=np.array(list(P)))
    out = pd.DataFrame(rows)
    os.makedirs(f"{R}/reports/eda/imaging", exist_ok=True)
    out.to_csv(f"{R}/reports/eda/imaging/tta_{c.run_id}.csv", index=False)
    print(json.dumps({"stored_best_val_macro_auc": ck.get("val_macro_auc")}))


if __name__ == "__main__":
    main()
