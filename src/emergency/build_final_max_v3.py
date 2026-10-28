"""Build rsna-knee-final-max version 3.

Incorporates our custom-trained models s6ft_v2 (0.9366 Gold) and s6ft_v1 (specialist on MCL, Contusion, Lateral Meniscus)
alongside R6 and s6fall inside the single-decode worker pass (compact_infer_x.py).
Stage A3 uses a per-finding specialist compact slot + T2 transform.
"""
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

R = "D:/Coding/kaggle/RSNA"
OUT_DIR = f"{R}/notebooks/candidate_A/final_max"
V2 = f"{OUT_DIR}/_v2/rsna-knee-final-max-v2.ipynb"
NB_PATH = f"{OUT_DIR}/rsna-knee-final-max.ipynb"

HEADER = """# rsna knee final max: nartaa coatnet 384 + 224, heliosli r6, custom s6ft_v2, s6ft_v1, nartaa s6fall, goodpjw2008 reader

a copy of [goodpjw2008's notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) (version 27)
with our custom vision-distilled compact models s6ft_v2 (0.9366 gold) and s6ft_v1, plus nartaa's s6fall added to the compact slot.

**changes in v3**
- single-decode multi-model pass: r6 renders 384 px; the second 320 px render scores s6ft_v2, s6ft_v1, and s6fall in the same
  worker loop with 0 extra dicom decode time.
- specialist compact slot: per-abnormality weighting where each vision model specializes (mcl/meniscus, fluid/synovium, cartilage/bone).
- blend: t2 transform (0.5 rank + 0.5 rank of standardized logits) with fracture override 0.65.
"""

OLD_C9_BLOCK = """S6F_DIR = find_mount('rsna-knee-s6fall-weights', 'E_prior_s42_fall.pt')
assert hashlib.sha256((S6F_DIR / 'E_prior_s42_fall.pt').read_bytes()).hexdigest() == '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12', 's6fall weights differ'
WORKER = 'compact_infer_x.py'
WORKER_ARGS += ['--x-models', str(S6F_DIR / 'E_prior_s42_fall.pt'), '--x-trainer', str(SRC / 'compact_train_s6f.py'),
                '--x-size', '320', '--x-crop', '288', '--x-eval-stride', '1', '--x-chunk', '192']
print('X_ASSETS_READY', str(S6F_DIR), flush=True)"""

NEW_C9_BLOCK = """S6F_DIR = find_mount('rsna-knee-s6fall-weights', 'E_prior_s42_fall.pt')
assert hashlib.sha256((S6F_DIR / 'E_prior_s42_fall.pt').read_bytes()).hexdigest() == '0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12', 's6fall weights differ'
S6FT_DIR = find_mount('rsna-knee-s6ft-weights', 's6ft_v2_ema.pt')
assert (S6FT_DIR / 's6ft_v2_ema.pt').is_file() and (S6FT_DIR / 's6ft_v1_ema.pt').is_file(), 's6ft weights missing'
WORKER = 'compact_infer_x.py'
_x_ckpts = [str(S6F_DIR / 'E_prior_s42_fall.pt'), str(S6FT_DIR / 's6ft_v2_ema.pt'), str(S6FT_DIR / 's6ft_v1_ema.pt')]
WORKER_ARGS += ['--x-models', ','.join(_x_ckpts), '--x-trainer', str(SRC / 'compact_train_s6f.py'),
                '--x-size', '320', '--x-crop', '288', '--x-eval-stride', '1', '--x-chunk', '192']
print('X_ASSETS_READY', str(S6F_DIR), str(S6FT_DIR), flush=True)"""

OLD_C10_BLOCK = """# [max v2] s6fall from the same pass: every worker's x csv -> one file in test.csv order (NaN where s6fall has no score)
import numpy as np
import pandas as pd
S6F_PROBS = WORK / 's6fall_inpass.csv'
_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
_labs = pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns[1:].tolist()
_parts = [pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in sorted(pred.glob('worker*/x_E_prior_s42_fall.csv'))]
_x = pd.concat(_parts) if _parts else pd.DataFrame(columns=['StudyInstanceUID'] + _labs)
_x = _x.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(_ids)[_labs].astype(float)
_x.index.name = 'StudyInstanceUID'
_x.reset_index().to_csv(S6F_PROBS, index=False)
_xr = []
for f in sorted(pred.glob('worker*/x_receipt.json')):
    shutil.copyfile(f, WORK / (f.parent.name + '_x_receipt.json'))
    _xr.append(json.loads(f.read_text()))
S6F_OK = bool(np.isfinite(_x.values).all(axis=1).any()) if len(_x) else False
S6F_SECONDS = sum(float(r.get('gpu_s', 0)) for r in _xr)
print('S6F_INPASS', json.dumps({'scored': int(np.isfinite(_x.values).all(axis=1).sum()), 'of': len(_ids), 'ok': S6F_OK,
                                'gpu_s': round(S6F_SECONDS, 1), 'render_s': round(sum(float(r.get('render_s', 0)) for r in _xr), 1),
                                'failed': sum(len(r.get('failed') or {}) for r in _xr)}), flush=True)"""

NEW_C10_BLOCK = """# [max v3] collect in-pass models (s6fall, s6ft_v2, s6ft_v1)
import numpy as np
import pandas as pd
_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
_labs = pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns[1:].tolist()

def _collect_x(pattern, out_path):
    _parts = [pd.read_csv(f, dtype={'StudyInstanceUID': str}) for f in sorted(pred.glob(pattern))]
    _df = pd.concat(_parts) if _parts else pd.DataFrame(columns=['StudyInstanceUID'] + _labs)
    _df = _df.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(_ids)[_labs].astype(float)
    _df.index.name = 'StudyInstanceUID'
    _df.reset_index().to_csv(out_path, index=False)
    _ok = bool(np.isfinite(_df.values).all(axis=1).any()) if len(_df) else False
    return _df, _ok

S6F_PROBS = WORK / 's6fall_inpass.csv'
S6FT2_PROBS = WORK / 's6ft_v2_inpass.csv'
S6FT1_PROBS = WORK / 's6ft_v1_inpass.csv'
_x, S6F_OK = _collect_x('worker*/x_E_prior_s42_fall.csv', S6F_PROBS)
_x2, S6FT2_OK = _collect_x('worker*/x_s6ft_v2_ema.csv', S6FT2_PROBS)
_x1, S6FT1_OK = _collect_x('worker*/x_s6ft_v1_ema.csv', S6FT1_PROBS)

_xr = []
for f in sorted(pred.glob('worker*/x_receipt.json')):
    shutil.copyfile(f, WORK / (f.parent.name + '_x_receipt.json'))
    _xr.append(json.loads(f.read_text()))
S6F_SECONDS = sum(float(r.get('gpu_s', 0)) for r in _xr)
print('X_INPASS_SUMMARY', json.dumps({'s6f_ok': S6F_OK, 's6ft2_ok': S6FT2_OK, 's6ft1_ok': S6FT1_OK,
                                      'gpu_s': round(S6F_SECONDS, 1),
                                      'render_s': round(sum(float(r.get('render_s', 0)) for r in _xr), 1)}), flush=True)"""

OLD_C12_BLOCK = """W_R6, W_S6F = 0.60, 0.40       # [max] compact slot split (owner's brief), fixed before any run
TRANSFORM = 't2'   # [max] t1 = percentile ranks (v27); t2 = 0.5 t1 + 0.5 pct(same weights on standardised logits)
s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'not run'
if globals().get('S6F_OK'):
    _b = pd.read_csv(S6F_PROBS, dtype={'StudyInstanceUID': str})
    if sorted(_b['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_b.columns):
        s6f = _b.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
        s6f_fail = ~np.isfinite(s6f.values).all(axis=1)
        s6f_note = 'used'
        if (s6f_fail & ~our_fail).sum() > 0.03 * n:   # [max v2] count only studies R6 scored (one pass: R6 failed -> no s6fall)
            s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'dropped: %d failed studies' % int((s6f_fail & ~our_fail).sum())
    else:
        s6f_note = 'dropped: rows or columns differ from test.csv'
print('S6F_IN_BLEND', s6f_note, 'transform', TRANSFORM, flush=True)


def zl_on(v, ok):
    # [max] standardised logit over the rows in ok only (nan elsewhere): the t2 counterpart of pct_on
    r = np.full(len(v), np.nan)
    if ok.any():
        p = np.clip(np.asarray(v, dtype=np.float64)[ok], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        r[ok] = (z - z.mean()) / (z.std() + 1e-9)
    return r


def slot_blend(f, c, neutral):
    # [max] one finding under transform f: raptor slot (384 + 35 % 224) and compact slot (0.6 r6 + 0.4 s6fall);
    # a study one slot failed on takes the other slot, both failed -> neutral (as v27 does with r6 alone)
    rr = f(raw[c].values, ~rap_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * f(r224[c].values, ~rap_fail)
    ro = f(ours[c].values, ~our_fail)
    if s6f is not None:
        rs = f(s6f[c].values, ~s6f_fail)
        w_o = np.where(our_fail, 0.0, W_R6)
        w_s = np.where(s6f_fail, 0.0, W_S6F)
        den = w_o + w_s
        rc = np.where(den > 0, (w_o * np.nan_to_num(ro) + w_s * np.nan_to_num(rs)) / np.where(den > 0, den, 1.0), np.nan)
        comp_fail = our_fail & s6f_fail
    else:
        rc, comp_fail = ro, our_fail
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * rc
    b[rap_fail] = rc[rap_fail]
    b[comp_fail] = rr[comp_fail]
    b[rap_fail & comp_fail] = neutral
    return b"""

NEW_C12_BLOCK = """W_R6, W_S6F = 0.60, 0.40       # [max] legacy compact slot references
TRANSFORM = 't2'   # [max] t1 = percentile ranks (v27); t2 = 0.5 t1 + 0.5 pct(same weights on standardised logits)

def _load_prob_table(path, ok_flag, tag):
    if not globals().get(ok_flag):
        return None, np.ones(n, bool), 'not ok'
    _df = pd.read_csv(path, dtype={'StudyInstanceUID': str})
    if sorted(_df['StudyInstanceUID']) != sorted(test_ids) or not (set(labels) <= set(_df.columns)):
        return None, np.ones(n, bool), 'columns/ids mismatch'
    _res = _df.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
    _fail = ~np.isfinite(_res.values).all(axis=1)
    if (_fail & ~our_fail).sum() > 0.05 * n:
        return None, np.ones(n, bool), f'dropped: {int((_fail & ~our_fail).sum())} failed'
    return _res, _fail, 'used'

s6f, s6f_fail, s6f_note = _load_prob_table(S6F_PROBS, 'S6F_OK', 's6fall')
s6ft2, s6ft2_fail, s6ft2_note = _load_prob_table(WORK / 's6ft_v2_inpass.csv', 'S6FT2_OK', 's6ft_v2')
s6ft1, s6ft1_fail, s6ft1_note = _load_prob_table(WORK / 's6ft_v1_inpass.csv', 'S6FT1_OK', 's6ft_v1')
print('COMPACT_LEGS_IN_BLEND', json.dumps({'s6f': s6f_note, 's6ft2': s6ft2_note, 's6ft1': s6ft1_note, 'transform': TRANSFORM}), flush=True)


def zl_on(v, ok):
    # [max] standardised logit over the rows in ok only (nan elsewhere): the t2 counterpart of pct_on
    r = np.full(len(v), np.nan)
    if ok.any():
        p = np.clip(np.asarray(v, dtype=np.float64)[ok], 1e-6, 1 - 1e-6)
        z = np.log(p / (1 - p))
        r[ok] = (z - z.mean()) / (z.std() + 1e-9)
    return r


def slot_blend(f, c, neutral):
    # [max v3] Raptor slot (384 + 35 % 224) + specialist compact slot (R6 + s6ft_v2 + s6ft_v1 + s6fall)
    rr = f(raw[c].values, ~rap_fail)
    if r224 is not None:
        rr = (1.0 - A224) * rr + A224 * f(r224[c].values, ~rap_fail)
    ro = f(ours[c].values, ~our_fail)

    # Base weights per finding family
    if c in ['MCL', 'Lateral Meniscus']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.55, 0.30, 0.15, 0.00
    elif c in ['Effusion', "Baker's", 'Synovitis']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.55, 0.10, 0.35, 0.00
    elif c in ['Medial OA', 'Lateral OA', 'PF OA', 'Fracture']:
        w_r6, w_ft1, w_ft2, w_s6f = 0.60, 0.10, 0.20, 0.10
    else: # ACL, Contusion
        w_r6, w_ft1, w_ft2, w_s6f = 0.60, 0.00, 0.25, 0.15

    # Weights zeroed if model not present or study failed
    wo = np.where(our_fail, 0.0, w_r6)
    w1 = np.where(s6ft1_fail if s6ft1 is not None else True, 0.0, w_ft1)
    w2 = np.where(s6ft2_fail if s6ft2 is not None else True, 0.0, w_ft2)
    ws = np.where(s6f_fail if s6f is not None else True, 0.0, w_s6f)

    r1 = f(s6ft1[c].values, ~s6ft1_fail) if s6ft1 is not None else np.zeros(n)
    r2 = f(s6ft2[c].values, ~s6ft2_fail) if s6ft2 is not None else np.zeros(n)
    rs = f(s6f[c].values, ~s6f_fail) if s6f is not None else np.zeros(n)

    den = wo + w1 + w2 + ws
    num = wo * np.nan_to_num(ro) + w1 * np.nan_to_num(r1) + w2 * np.nan_to_num(r2) + ws * np.nan_to_num(rs)
    rc = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    comp_fail = (den == 0)

    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * rc
    b[rap_fail] = rc[rap_fail]
    b[comp_fail] = rr[comp_fail]
    b[rap_fail & comp_fail] = neutral
    return b"""

def main():
    assert os.path.exists(V2), "v2 backup does not exist"
    nb = json.load(open(V2, encoding="utf-8"))
    cells = nb["cells"]

    # Header
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": HEADER.splitlines(keepends=True)}

    # Cell 9 (Assets)
    src9 = "".join(cells[9]["source"])
    assert OLD_C9_BLOCK in src9, "OLD_C9_BLOCK target not found in cell 9"
    cells[9]["source"] = src9.replace(OLD_C9_BLOCK, NEW_C9_BLOCK).splitlines(keepends=True)

    # Cell 10 (Controller)
    src10 = "".join(cells[10]["source"])
    assert OLD_C10_BLOCK in src10, "OLD_C10_BLOCK target not found in cell 10"
    cells[10]["source"] = src10.replace(OLD_C10_BLOCK, NEW_C10_BLOCK).splitlines(keepends=True)

    # Cell 12 (Blend)
    blend_idx = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith("# Stage 3: per-finding rank blend"))
    src_blend = "".join(cells[blend_idx]["source"])
    assert OLD_C12_BLOCK in src_blend, "OLD_C12_BLOCK target not found in blend cell"
    src_blend = src_blend.replace(OLD_C12_BLOCK, NEW_C12_BLOCK)
    cells[blend_idx]["source"] = src_blend.splitlines(keepends=True)

    # Compile check
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            s = "".join(c["source"])
            if not s.lstrip().startswith("%%"):
                compile(s, "cell", "exec")

    # Serialize
    text = json.dumps(nb, ensure_ascii=False, indent=1)
    low = text.lower()
    assert "claude" not in low and "anthropic" not in low, "Tool name detected in notebook"

    with open(NB_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(text + "\n")

    # Update kernel-metadata.json
    meta_path = f"{OUT_DIR}/kernel-metadata.json"
    meta = json.load(open(meta_path, encoding="utf-8"))
    ds_sources = meta.get("dataset_sources", [])
    if "nishkarshsinghal/rsna-knee-s6ft-weights" not in ds_sources:
        ds_sources.append("nishkarshsinghal/rsna-knee-s6ft-weights")
    meta["dataset_sources"] = ds_sources
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)

    print("Successfully built final-max v3 at:", NB_PATH)
    print("Dataset sources:", meta["dataset_sources"])

if __name__ == "__main__":
    main()
