"""Execute notebooks/rsna_knee_label_generation.ipynb locally as a smoke test.

Runs every code cell in order in one namespace, with small overrides:
the full-run cell processes an 8-study subset, run twice to exercise resume.
"""
import json
import os
import shutil
import sys
import time
import traceback

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsna_knee_label_generation.ipynb")
OUT = "D:/rsna_work/tmp/nb_smoke"
shutil.rmtree(OUT, ignore_errors=True)
os.chdir("D:/Coding/kaggle/RSNA")  # so the notebook's relative discovery finds data/Raw_dicom/train.csv
cells = [c for c in json.load(open(NB, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
ns = {}
log = []
for i, c in enumerate(cells):
    src = "".join(c["source"])
    first = src.strip().splitlines()[0][:70]
    if i == 0:
        src += f"\nCONFIG['OUT_DIR'] = r'{OUT}'\nCONFIG['SMOKE_TEST_N'] = 3\n"
    if "OUT_CSV = os.path.join" in src:  # full-run cell: restrict to a subset, run twice (resume)
        pre = ("_full_df = df\n_sub = list(df[~gold_mask].StudyInstanceUID[:6]) + list(df[gold_mask].StudyInstanceUID[:2])\n"
               "df = _full_df[_full_df.StudyInstanceUID.isin(_sub)].reset_index(drop=True)\n"
               "gold_mask = df[TARGETS].notna().all(axis=1)\n")
        src = pre + src + "\n" + src  # second pass must find everything done
    t0 = time.time()
    try:
        exec(compile(src, f"cell{i}", "exec"), ns)
        status = "ok"
    except Exception as ex:
        status = f"ERROR {type(ex).__name__}: {ex}"
        traceback.print_exc()
    log.append((i, first, status, round(time.time() - t0, 1)))
    print(f"[cell {i}] {status} ({time.time()-t0:.1f}s) :: {first}", flush=True)
print("\nSUMMARY")
for row in log:
    print(row)
sys.exit(0 if all(r[2] == "ok" for r in log) else 1)
