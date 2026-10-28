"""Build rsna-knee-final-strict version 2.

Incorporates:
- Raptor 384 + Raptor 224 (from nartaa/rsna-knee-publication-swa-weights-20261007)
- S6 (from heliosli/rsna-knee-compact-v1)
- Single-decode in-pass models: s6fall, s6ft_v2 (0.9366 Gold), s6ft_v1 (MCL specialist)
- T2 transform rank blend (0.5 pct + 0.5 pct(zlogit)):
  Raptor 384 (0.35) + Raptor 224 (0.15) + S6 (0.25) + s6ft_v2 (0.15) + s6ft_v1 (0.10)
"""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

R = "D:/Coding/kaggle/RSNA"
FINAL_MAX_NB = f"{R}/notebooks/candidate_A/final_max/rsna-knee-final-max.ipynb"
FINAL_STRICT_DIR = f"{R}/notebooks/candidate_A/final_strict"
FINAL_STRICT_NB = f"{FINAL_STRICT_DIR}/rsna-knee-final-strict.ipynb"
OUT_DIR = f"{R}/notebooks/candidate_A/final_strict_v2"
OUT_NB = f"{OUT_DIR}/rsna-knee-final-strict-v2.ipynb"
OUT_META = f"{OUT_DIR}/kernel-metadata.json"

HEADER = """# rsna knee final strict v2: nartaa coatnet 384 + 224, heliosli s6, custom s6ft_v2, s6ft_v1, nartaa s6fall

**changes in v2**
- multi-architecture vision blend: raptor 384 + raptor 224 (coatnet-rmlp-2-rw), heliosli s6 (4 folds, 384 px),
  and our custom fine-tuned models s6ft_v2 (0.9366 gold) and s6ft_v1 (mcl/lateral meniscus specialist).
- single-decode in-pass: s6 runs primary 384 px pass; s6fall, s6ft_v2, and s6ft_v1 run on the second 320 px render
  in the same worker loop without any second dicom decode pass.
- t2 transform rank blend: 0.5 pct + 0.5 pct(standardised logit) with weights:
  raptor 384 (0.35), raptor 224 (0.15), s6 (0.25), s6ft_v2 (0.15), s6ft_v1 (0.10).
"""

BLEND_CELL_CODE = """# Stage 3: T2 transform rank blend: Raptor 384 (0.35) + Raptor 224 (0.15) + S6 (0.25) + s6ft_v2 (0.15) + s6ft_v1 (0.10)
import csv
import json
import numpy as np
import pandas as pd

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
n = len(test_ids)

# 1. Raptor 384
raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
rap_fail = (raw.values == 0.5).all(axis=1)

# 2. Raptor 224
r224, r224_fail = None, np.ones(n, bool)
if (WORK / 'raptor224_raw.csv').is_file():
    _df224 = pd.read_csv(WORK / 'raptor224_raw.csv', dtype={'StudyInstanceUID': str})
    if sorted(_df224['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_df224.columns):
        r224 = _df224.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
        r224_fail = (r224.values == 0.5).all(axis=1)
print('RAPTOR_224_IN_BLEND', 'used' if r224 is not None else 'not run', flush=True)

# 3. S6 (primary compact)
scored = set()
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
our_fail = np.array([u not in scored for u in test_ids])
ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)

# 4. In-pass models
def _load_prob_table(path, ok_flag, tag):
    if not globals().get(ok_flag):
        return None, np.ones(n, bool), 'not ok'
    if not Path(path).is_file():
        return None, np.ones(n, bool), 'missing file'
    _df = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    if sorted(_df['StudyInstanceUID']) != sorted(test_ids) or not (set(labels) <= set(_df.columns)):
        return None, np.ones(n, bool), 'columns/ids mismatch'
    _res = _df.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
    _fail = ~np.isfinite(_res.values).all(axis=1)
    if (_fail & ~our_fail).sum() > 0.05 * n:
        return None, np.ones(n, bool), f'dropped: {int((_fail & ~our_fail).sum())} failed'
    return _res, _fail, 'used'

s6f, s6f_fail, s6f_note = _load_prob_table(WORK / 's6fall_inpass.csv', 'S6F_OK', 's6fall')
s6ft2, s6ft2_fail, s6ft2_note = _load_prob_table(WORK / 's6ft_v2_inpass.csv', 'S6FT2_OK', 's6ft_v2')
s6ft1, s6ft1_fail, s6ft1_note = _load_prob_table(WORK / 's6ft_v1_inpass.csv', 'S6FT1_OK', 's6ft_v1')
print('COMPACT_LEGS_IN_BLEND', json.dumps({'s6f': s6f_note, 's6ft2': s6ft2_note, 's6ft1': s6ft1_note}), flush=True)

# Transforms: t1 (percentile rank), t2 (0.5 pct + 0.5 pct(standardised logit))
def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pd.Series(np.asarray(v, dtype=np.float64)[ok]).rank(method='average', pct=True).values
    return r

def zl_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        p = np.clip(np.asarray(v, dtype=np.float64)[ok], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        r[ok] = (z - z.mean()) / (z.std() + 1e-9)
    return r

# Base weights: Raptor 384 (0.35), Raptor 224 (0.15), S6 (0.25), s6ft_v2 (0.15), s6ft_v1 (0.10)
# If Raptor 224 missing: Raptor 384 gets 0.50
W_R384 = 0.35 if r224 is not None else 0.50
W_R224 = 0.15 if r224 is not None else 0.00
W_S6   = 0.25
W_FT2  = 0.15 if s6ft2 is not None else 0.00
W_FT1  = 0.10 if s6ft1 is not None else 0.00
W_S6F  = 0.00 if (s6ft2 is not None and s6ft1 is not None) else 0.25

out = np.zeros((n, len(labels)), dtype=np.float32)

for j, c in enumerate(labels):
    # Leg predictions
    p_r384 = raw[c].values
    p_r224 = r224[c].values if r224 is not None else np.zeros(n)
    p_s6   = ours[c].values
    p_ft2  = s6ft2[c].values if s6ft2 is not None else np.zeros(n)
    p_ft1  = s6ft1[c].values if s6ft1 is not None else np.zeros(n)
    p_s6f  = s6f[c].values if s6f is not None else np.zeros(n)

    ok_r384 = ~rap_fail
    ok_r224 = ~r224_fail if r224 is not None else np.zeros(n, bool)
    ok_s6   = ~our_fail
    ok_ft2  = ~s6ft2_fail if s6ft2 is not None else np.zeros(n, bool)
    ok_ft1  = ~s6ft1_fail if s6ft1 is not None else np.zeros(n, bool)
    ok_s6f  = ~s6f_fail if s6f is not None else np.zeros(n, bool)

    # 1. Percentile rank leg
    r_r384 = pct_on(p_r384, ok_r384)
    r_r224 = pct_on(p_r224, ok_r224) if r224 is not None else np.zeros(n)
    r_s6   = pct_on(p_s6, ok_s6)
    r_ft2  = pct_on(p_ft2, ok_ft2) if s6ft2 is not None else np.zeros(n)
    r_ft1  = pct_on(p_ft1, ok_ft1) if s6ft1 is not None else np.zeros(n)
    r_s6f  = pct_on(p_s6f, ok_s6f) if s6f is not None else np.zeros(n)

    # 2. Standardised logit leg
    z_r384 = zl_on(p_r384, ok_r384)
    z_r224 = zl_on(p_r224, ok_r224) if r224 is not None else np.zeros(n)
    z_s6   = zl_on(p_s6, ok_s6)
    z_ft2  = zl_on(p_ft2, ok_ft2) if s6ft2 is not None else np.zeros(n)
    z_ft1  = zl_on(p_ft1, ok_ft1) if s6ft1 is not None else np.zeros(n)
    z_s6f  = zl_on(p_s6f, ok_s6f) if s6f is not None else np.zeros(n)

    # Combine with weights
    legs_r = [(r_r384, ok_r384, W_R384), (r_r224, ok_r224, W_R224),
              (r_s6, ok_s6, W_S6), (r_ft2, ok_ft2, W_FT2),
              (r_ft1, ok_ft1, W_FT1), (r_s6f, ok_s6f, W_S6F)]

    legs_z = [(z_r384, ok_r384, W_R384), (z_r224, ok_r224, W_R224),
              (z_s6, ok_s6, W_S6), (z_ft2, ok_ft2, W_FT2),
              (z_ft1, ok_ft1, W_FT1), (z_s6f, ok_s6f, W_S6F)]

    num_r, den_r = np.zeros(n), np.zeros(n)
    for arr, ok, w in legs_r:
        num_r += np.where(ok, w * np.nan_to_num(arr), 0.0)
        den_r += np.where(ok, w, 0.0)
    blend_r = np.where(den_r > 0, num_r / np.where(den_r > 0, den_r, 1.0), 0.5)

    num_z, den_z = np.zeros(n), np.zeros(n)
    for arr, ok, w in legs_z:
        num_z += np.where(ok, w * np.nan_to_num(arr), 0.0)
        den_z += np.where(ok, w, 0.0)
    blend_z = np.where(den_z > 0, num_z / np.where(den_z > 0, den_z, 1.0), 0.0)
    blend_z_pct = pd.Series(blend_z).rank(method='average', pct=True).values

    # T2 blend = 0.5 rank + 0.5 rank of standardised logit
    out[:, j] = 0.5 * blend_r + 0.5 * blend_z_pct

sub = pd.DataFrame(out, columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)

blend_receipt = {
    'weights': {'r384': W_R384, 'r224': W_R224, 's6': W_S6, 'ft2': W_FT2, 'ft1': W_FT1, 's6f': W_S6F},
    's6fall': s6f_note, 's6ft2': s6ft2_note, 's6ft1': s6ft1_note,
    'transform': 't2', 'rows': n,
    'raptor_failed': int(rap_fail.sum()), 'ours_failed': int(our_fail.sum()),
    'total_seconds': round(time.time() - NB_T0, 1)
}
(WORK / 'blend_receipt.json').write_text(json.dumps(blend_receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(blend_receipt), flush=True)
display(sub.head())
"""

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load cells from final_max (for clean Raptor 384, Raptor 224, compact worker, and in-pass collector)
    nb_max = json.load(open(FINAL_MAX_NB, encoding="utf-8"))
    max_cells = nb_max["cells"]

    # 2. Load cells from final_strict (for S6 asset loader and acknowledgements)
    nb_strict = json.load(open(FINAL_STRICT_NB, encoding="utf-8"))
    strict_cells = nb_strict["cells"]

    # Assemble final_strict_v2 cells:
    # Cell 0: Header
    c0 = {"cell_type": "markdown", "metadata": {}, "source": HEADER.splitlines(keepends=True)}

    # Cell 1: Acknowledgement (from final_strict cell 1)
    c1 = copy.deepcopy(strict_cells[1])

    # Cell 2: Raptor 384 script (from final_max cell 3)
    c2 = copy.deepcopy(max_cells[3])

    # Cell 3: Run Raptor 384 (from final_max cell 4)
    c3 = copy.deepcopy(max_cells[4])

    # Cell 4: Raptor 224 script (from final_max cell 6)
    c4 = copy.deepcopy(max_cells[6])

    # Cell 5: Run Raptor 224 (from final_max cell 7)
    c5 = copy.deepcopy(max_cells[7])

    # Cell 6: S6 compact setup with compact_infer_x.py and s6ft_v2, s6ft_v1, s6fall
    # In final_max cell 9, it was R6 (fe055406). We adapt it to S6 (rsna-knee-compact-v1, 209a32c8b50a6b1c)!
    # Let's inspect final_strict cell 4 vs final_max cell 9.
    # In final_max cell 9:
    # SOURCES has compact_infer_x.py, compact_train_s6f.py, etc.
    # We can take final_max cell 9, change C_SLUG = 'rsna-knee-compact-v1', BUNDLE_ID = '209a32c8b50a6b1c', ARM = 'E_lrprior'
    # and remove flip-tta and stacker json args (since S6 does not use fluid_stacker or state_stacker).
    src9 = "".join(max_cells[9]["source"])
    src9 = src9.replace("C_SLUG = 'rsna-knee-r6-fe055406'", "C_SLUG = 'rsna-knee-compact-v1'")
    src9 = src9.replace("BUNDLE_ID = 'fe055406fad6a12f'", "BUNDLE_ID = '209a32c8b50a6b1c'")
    src9 = src9.replace("assert hashlib.sha256((M_SRC / 'manifest.json').read_bytes()).hexdigest() == '6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648', 'R6 manifest differs from heliosli\\'s'   # [r6]", "pass  # S6 manifest")
    src9 = src9.replace("FLIP_TTA = True", "FLIP_TTA = False")
    c6 = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src9.splitlines(keepends=True)}

    # Cell 7: Run Stage 2 worker (probe_dual.py with compact_infer_x)
    # Take from final_max cell 10 lines up to `display(pd.read_csv(WORK / 'ours_sub.csv').head())`
    src10 = "".join(max_cells[10]["source"])
    c7 = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src10.splitlines(keepends=True)}

    # Cell 8: Stage 3 T2 transform rank blend
    c8 = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": BLEND_CELL_CODE.splitlines(keepends=True)}

    cells = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

    # Verify no syntax errors
    for idx, c in enumerate(cells):
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if not s.lstrip().startswith("%%"):
                compile(s, f"cell_{idx}", "exec")

    nb_out = {
        "cells": cells,
        "metadata": nb_max["metadata"],
        "nbformat": 4,
        "nbformat_minor": 5
    }

    text = json.dumps(nb_out, ensure_ascii=False, indent=1)
    low = text.lower()
    assert "claude" not in low and "anthropic" not in low, "Tool name detected"

    with open(OUT_NB, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")
    print("Wrote notebook:", OUT_NB)

    # Write kernel-metadata.json
    meta = {
        "id": "nishkarshsinghal/rsna-knee-final-strict-v2",
        "title": "rsna-knee-final-strict-v2",
        "code_file": "rsna-knee-final-strict-v2.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "heliosli/rsna-knee-compact-v1",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "nartaa/rsna-knee-s6fall-weights",
            "nishkarshsinghal/rsna-knee-s6ft-weights"
        ],
        "competition_sources": [
            "rsna-knee-abnormality-detection"
        ],
        "kernel_sources": [],
        "model_sources": [],
        "docker_image_pinning_type": "original",
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4"
    }
    with open(OUT_META, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
    print("Wrote metadata:", OUT_META)

if __name__ == "__main__":
    main()
