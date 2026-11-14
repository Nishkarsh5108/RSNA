"""Engine vs the verified Kaggle pipeline (src/postcomp/infer_bp.py) on the local public test studies, same weights.
Run: python tests/test_fidelity.py  (writes nothing outside OUT)."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
COMP = Path(os.environ.get("KNEE_TEST_DIR", "D:/Coding/kaggle/RSNA/data/Raw_dicom"))
OUT = Path(os.environ.get("OUT", "D:/rsna_work/demo_fidelity")); OUT.mkdir(parents=True, exist_ok=True)
import engine as E
series = pd.read_csv(COMP / "test_series.csv")
eng = E.Engine()
print("loaded", [k for k, _ in eng.cls], f"{eng.load_seconds:.1f}s", flush=True)
rows, errs, times = [], [], []
for study, g in series.groupby("StudyInstanceUID"):
    t = time.time()
    res = eng.run(COMP / "test_series", g, lambda ev: None)
    times.append(time.time() - t)
    s = res["summary"]
    print(study[-8:], "knee", s["knee"], f"{s['seconds']:.1f}s peak {s['peakVramGB']} GB, attr err {s['maxAttributionError']:.2e}, failures {s['failures']}", flush=True)
    for uid, r in res["arrays"].items():
        for k, v in r["models"].items():
            rows.append((k, uid, *v["probs"]))
ours = pd.DataFrame(rows, columns=["model", "SeriesInstanceUID", *E.LABELS])
# reference: infer_bp.py with the same weights
env = dict(os.environ, SERIES_ROOT=str(COMP / "test_series"), SERIES_CSV=str(COMP / "test_series.csv"), SUBMISSION_CSV=str(COMP / "sample_submission.csv"),
           SIDE_WEIGHTS=str(E.SIDE_WEIGHTS), DET_WEIGHTS=str(E.DET_WEIGHTS), CLS_WEIGHTS=",".join(str(E.CLS_DIR / f"{k}.pth") for k, _ in eng.cls),
           OUT_DIR=str(OUT), WORKERS="0", COMP_DIR=str(COMP))
del eng
import torch; torch.cuda.empty_cache()
r = subprocess.run([sys.executable, str(E.MODULES_DIR / "infer.py")], env=env, cwd=str(E.MODULES_DIR), capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-1500:])
ref = pd.read_csv(OUT / "series_preds.csv")
ref["model"] = ref["model"].map(dict(enumerate([k for k in sorted(p.stem for p in E.CLS_DIR.glob('*.pth'))])))
m = ours.merge(ref, on=["model", "SeriesInstanceUID"], suffixes=("", "_ref"))
d = np.abs(m[E.LABELS].to_numpy() - m[[f"{l}_ref" for l in E.LABELS]].to_numpy())
print(f"pairs {len(m)} of {len(ours)}; max |diff| {d.max():.2e}, mean {d.mean():.2e}")
json.dump(dict(pairs=len(m), max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), seconds=times), open(OUT / "fidelity.json", "w"), indent=1)
