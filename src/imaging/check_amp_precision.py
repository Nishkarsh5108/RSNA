"""Check that inference precision does not change predictions materially: fp32 vs bf16 (training
precision) vs fp16 (used on pre-Ampere GPUs such as Kaggle's T4), on pseudo-val studies.
Usage: python check_amp_precision.py RUN_ID [N_STUDIES]
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import KneeStudyDataset  # noqa: E402
from model import KneeMIL  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


def main():
    run_id = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    ck = torch.load(f"{R}/output/imaging_v5/runs/{run_id}/ckpt_best.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    m = KneeMIL(a["backbone"], pretrained=False, n_slots=6, agg=a["agg"], use_pos=not a.get("no_pos", False))
    m.load_state_dict(ck["model"]); m.eval().cuda()
    idx = pd.read_csv(a["index"])
    va = idx[idx.fold == a["val_fold"]].head(n).reset_index(drop=True)
    slots = [int(s) for s in str(a["slots"]).split(",")]
    ds = KneeStudyDataset(va, a["cache"], slots, a["K"], train=False, lo=a["lo"], hi=a["hi"], orient=not a.get("no_orient", False),
                              sag_pos_fix=a.get("sag_pos_fix", False))
    dl = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    out = {k: np.zeros((len(ds), 12), np.float32) for k in ("fp32", "bf16", "fp16")}
    with torch.no_grad():
        for b in dl:
            x = b["x"].cuda().float().div_(255)
            mk, ps, i = b["mask"].cuda(), b["pos"].cuda(), b["i"].numpy()
            out["fp32"][i] = torch.sigmoid(m(x, mk, ps).float()).cpu().numpy()
            for k, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
                with torch.autocast("cuda", dtype=dt):
                    out[k][i] = torch.sigmoid(m(x, mk, ps).float()).cpu().numpy()
    for k in ("bf16", "fp16"):
        d = np.abs(out[k] - out["fp32"])
        rho = min(spearmanr(out[k][:, j], out["fp32"][:, j])[0] for j in range(12))
        print(f"{k} vs fp32: max |dp| {d.max():.4f}, mean |dp| {d.mean():.5f}, min per-class Spearman {rho:.5f}, "
              f"finite {bool(np.isfinite(out[k]).all())}")
    d = np.abs(out["fp16"] - out["bf16"])
    print(f"fp16 vs bf16: max |dp| {d.max():.4f}, mean |dp| {d.mean():.5f}  (n={len(ds)} studies)")


if __name__ == "__main__":
    main()
