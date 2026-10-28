"""Verify the v6 cache after build_cache_v6.py exits and only then write the _READY marker
that lets the experiment queue start runs on it.

Checks: the build's _failures.csv exists and is empty; every expected series file exists
(same job list as the build); a random sample of 300 arrays has shape (T, 256, 256), dtype
uint8, T >= 3, and is not constant. Any failure -> no marker, non-zero exit, reasons printed.
Usage: python finalize_cache_v6.py [--wait-pid PID]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_index import SLOT_NAMES  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
OUT = "D:/rsna_work/cache_v6_256"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, default=0)
    ap.add_argument("--index", default=f"{R}/data/imaging_v5/study_index_train_v2.csv")
    ap.add_argument("--slots", default="0,1,2")
    a = ap.parse_args()
    if a.wait_pid:
        import psutil
        while psutil.pid_exists(a.wait_pid):
            time.sleep(30)
    problems = []
    tag = "" if a.slots == "0,1,2" else f"_slots_{a.slots.replace(',', '')}"     # same naming as build_cache_v6.py
    fpath = f"{OUT}/_failures{tag}.csv"
    if not os.path.exists(fpath):
        problems.append("no _failures.csv: the build did not finish")
    else:
        f = pd.read_csv(fpath)
        if len(f):
            problems.append(f"{len(f)} series failed: {f.head(3).to_dict('records')}")
    idx = pd.read_csv(a.index)
    expected = set()
    for r in idx.itertuples():
        for si in [int(s) for s in a.slots.split(",")]:
            for name in [SLOT_NAMES[si]] + ([SLOT_NAMES[si + 3]] if si < 3 else []):
                u = getattr(r, f"{name}_uid")
                if isinstance(u, str):
                    expected.add(u)
                    break
    missing = [u for u in expected if not os.path.exists(f"{OUT}/{u}.npy")]
    if missing:
        problems.append(f"{len(missing)} expected files missing, e.g. {missing[:3]}")
    rng = np.random.default_rng(0)
    sample = rng.choice(sorted(expected - set(missing)), size=min(300, len(expected) - len(missing)), replace=False)
    bad = []
    for u in sample:
        x = np.load(f"{OUT}/{u}.npy", mmap_mode="r")
        if x.dtype != np.uint8 or x.ndim != 3 or x.shape[1:] != (256, 256) or x.shape[0] < 3 or x.max() == x.min():
            bad.append((u, x.dtype.name, x.shape))
    if bad:
        problems.append(f"{len(bad)} sampled arrays invalid: {bad[:3]}")
    tmp_left = [f for f in os.listdir(OUT) if f.endswith(".tmp.npy")]
    rep = {"expected": len(expected), "missing": len(missing), "sampled": len(sample), "bad_sampled": len(bad),
           "leftover_tmp_files": len(tmp_left), "problems": problems, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(json.dumps(rep, indent=1))
    if problems:
        sys.exit(1)
    json.dump(rep, open(f"{OUT}/_READY{tag}", "w"), indent=1)
    print("READY marker written")


if __name__ == "__main__":
    main()
