"""Build the v6 cache (0.6 mm/px aspect-preserving, 256x256 crop) for the series a model needs.

Default: the 3 FS-first slots (with the plane's non-FS series where the FS slot is empty)
of every train + test study in the given index. Resumable (skips existing files); records
failures to a CSV and exits non-zero if any series failed (never silently skipped).
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_lib import preprocess_series_v6  # noqa: E402
from study_index import SLOT_NAMES  # noqa: E402

R = "D:/Coding/kaggle/RSNA"


def job(a):
    sdir, out = a
    if os.path.exists(out):
        return out, None, []
    try:
        arr, geo = preprocess_series_v6(sdir)
        tmp = out + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out)
        return out, None, geo.get("dropped_slices", [])
    except Exception as ex:
        return out, f"{type(ex).__name__}: {ex}", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=f"{R}/data/imaging_v5/study_index_train_v2.csv")
    ap.add_argument("--dicom", default=f"{R}/data/Raw_dicom/train_series")
    ap.add_argument("--out", default="D:/rsna_work/cache_v6_256")
    ap.add_argument("--slots", default="0,1,2")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = pd.read_csv(a.index)
    slots = [int(s) for s in a.slots.split(",")]
    jobs = set()
    for r in idx.itertuples():
        for si in slots:
            for name in [SLOT_NAMES[si]] + ([SLOT_NAMES[si + 3]] if si < 3 else []):
                u = getattr(r, f"{name}_uid")
                if isinstance(u, str):
                    jobs.add((f"{a.dicom}/{r.StudyInstanceUID}/{u}", f"{a.out}/{u}.npy"))
                    break
    jobs = sorted(jobs)
    print(f"{len(jobs)} series to cache -> {a.out}", flush=True)
    t0, fails, dropped = time.time(), [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (out, err, drop) in enumerate(ex.map(job, jobs, chunksize=4)):
            if err:
                fails.append((out, err))
            if drop:
                dropped.append((out, "|".join(drop)))
            if i % 500 == 0:
                el = time.time() - t0
                print(f"{i}/{len(jobs)} {el/60:.1f} min, eta {(len(jobs)-i)*el/max(i,1)/60:.0f} min, fails {len(fails)}", flush=True)
    fname = "_failures.csv" if a.slots == "0,1,2" else f"_failures_slots_{a.slots.replace(',', '')}.csv"
    pd.DataFrame(fails, columns=["file", "error"]).to_csv(f"{a.out}/{fname}", index=False)
    if dropped:                                   # one undecodable end slice dropped (infer_lib.decode_series)
        pd.DataFrame(dropped, columns=["file", "dropped_slices"]).to_csv(
            f"{a.out}/_dropped_slices{fname.replace('_failures', '').replace('.csv', '')}_{time.strftime('%H%M%S')}.csv", index=False)
    print(f"done in {(time.time()-t0)/60:.1f} min; failures {len(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
