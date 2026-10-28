"""Build rsna-knee-final-max version 2 (plan section 14 A; mission 2026-10-21).

final-max v1 (pushed 2026-10-20, submission 57444580, 0.958) with ONE change: s6fall is scored inside R6's own worker, on
a second 320-px render of the same decoded study, instead of in its own a2b stage that decodes every DICOM again.
That stage most likely never finished on the hidden test (plan 14), so v1 there was v27 alone.
  - cell 9 (R6 assets): compact_infer_x.py (build_compact_infer_x.py: heliosli's v4c worker + --x-models) and s6fall's own
    trainer compact_train.py (sha 14767c7a, heliosli, as nartaa's stage runs it) are written next to the embedded sources,
    each sha-checked; the worker becomes compact_infer_x.py with --x-models <s6fall> --x-size 320 --x-crop 288
    --x-eval-stride 1 --x-chunk 192 (nartaa's stage geometry).
  - cell 10 (R6 controller): after it, the workers' x_E_prior_s42_fall.csv files are merged into the csv the blend reads
    (a NaN row for a study s6fall did not score), S6F_OK is set.
  - cells 11-12 (the a2b stage): removed.
  - the blend (A3): one line. s6fall is dropped only if it failed on > 3 % of the studies R6 scored (in one pass, a study R6
    could not score has no s6fall either; v1's count would drop s6fall everywhere once R6 missed 3 %).
  - Raptor stages, reader and everything else: byte-identical to v1.
Usage: python -I src/emergency/build_final_max_v2.py   (after build_compact_infer_x.py)
Writes notebooks/candidate_A/final_max/{rsna-knee-final-max.ipynb, kernel-metadata.json}; v1's notebook is kept in _v1/.
"""
import base64
import copy
import hashlib
import json
import os
import shutil
import zlib

R = "D:/Coding/kaggle/RSNA"
OUT_DIR = f"{R}/notebooks/candidate_A/final_max"
V1 = f"{OUT_DIR}/_v1/rsna-knee-final-max-v1.ipynb"
V1_SHA = "a060ce0200f1c483951211f842bd15144fe9dacb46b19237e4d90b09589a3a9e"       # pushed as version 1, 2026-10-20 21:02 IST
XW = f"{R}/src/emergency/compact_infer_x.py"
S6T = "D:/rsna_work/public_code/s6_src/compact_train.py"
S6T_SHA = "14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80"      # s6fall run_config heliosli_source_sha256
S6F_SHA = "0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12"      # E_prior_s42_fall.pt (nartaa's stage B_MODEL)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def replace_once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, found {n}")
    return s.replace(old, new)


HEADER = """# rsna knee final max: nartaa coatnet 384 + 224, heliosli r6, nartaa s6fall, goodpjw2008 reader

a copy of [goodpjw2008's notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) (version 27)
with nartaa's all-data s6 convnext (s6fall) added to the compact slot.

**changes**
- s6fall is scored inside r6's worker: each test study is decoded once, its knee anchors are placed once, and the rows are
  rendered twice (384 px for r6, 320 px for s6fall). s6fall then runs on every 320 px window with a 288 centre crop and
  a plain fp16 pass, using its own trainer code from
  [nartaa/rsna-knee-s6fall-weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights) (cc by-nc 4.0),
  exactly as [nartaa/rsna-knee-s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo) does. a second
  decode pass is not needed. the worker is heliosli's compact worker with this one addition (`compact_infer_x.py`).
- blend: compact slot = 0.60 r6 + 0.40 s6fall (r6 alone where s6fall is missing); raptor weight per finding as in v27, plus
  fracture 0.65; half per-finding percentile ranks, half the rank of the same weighted blend on standardised logits.
- everything else is unchanged.

**external data:** oai (nartaa's coatnets), skm-tea (`anatomy.pt`, the knee anchors of r6 and s6fall), and fastmri+ (r6's
auxiliary training task). labels come from the competition's own reports.

all credit for the models and code goes to their authors (below). the source notebooks are released under apache 2.0.
"""

ASSETS_OLD = "print('ASSETS_READY', str(CODE_SRC), str(M_SRC), str(DATA), 'flip_tta', '--flip-tta' in WORKER_ARGS, 'seconds %.1f' % (time.time() - T0), flush=True)"
ASSETS_NEW = '''# [max v2] s6fall inside this worker: compact_infer_x.py = the v4c worker above + extra models scored on a second render of
# the same decoded study (320 px rows, 288 centre crop, every window, s6fall's own trainer and plain fp16 pass)
X_FILES = {'compact_infer_x.py': ("__XW_B64__", "__XW_SHA__"), 'compact_train_s6f.py': ("__S6T_B64__", "__S6T_SHA__")}
for _n, (_b, _h) in X_FILES.items():
    (SRC / _n).write_bytes(_zlib.decompress(_b64.b64decode(_b)))
    assert hashlib.sha256((SRC / _n).read_bytes()).hexdigest() == _h, _n
S6F_DIR = find_mount('rsna-knee-s6fall-weights', 'E_prior_s42_fall.pt')
assert hashlib.sha256((S6F_DIR / 'E_prior_s42_fall.pt').read_bytes()).hexdigest() == '__S6F_SHA__', 's6fall weights differ'
WORKER = 'compact_infer_x.py'
WORKER_ARGS += ['--x-models', str(S6F_DIR / 'E_prior_s42_fall.pt'), '--x-trainer', str(SRC / 'compact_train_s6f.py'),
                '--x-size', '320', '--x-crop', '288', '--x-eval-stride', '1', '--x-chunk', '192']
print('X_ASSETS_READY', str(S6F_DIR), flush=True)
''' + ASSETS_OLD

CTRL_OLD = "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')\n"
CTRL_NEW = CTRL_OLD + '''
# [max v2] s6fall from the same pass: every worker's x csv -> one file in test.csv order (NaN where s6fall has no score)
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
                                'failed': sum(len(r.get('failed') or {}) for r in _xr)}), flush=True)
'''


A3_DROP_OLD = """        if s6f_fail.sum() > 0.03 * n:
            s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'dropped: %d failed studies' % int(s6f_fail.sum())"""
A3_DROP_NEW = """        if (s6f_fail & ~our_fail).sum() > 0.03 * n:   # [max v2] count only studies R6 scored (one pass: R6 failed -> no s6fall)
            s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'dropped: %d failed studies' % int((s6f_fail & ~our_fail).sum())"""


def main():
    os.makedirs(os.path.dirname(V1), exist_ok=True)
    cur = f"{OUT_DIR}/rsna-knee-final-max.ipynb"
    if not os.path.exists(V1):
        assert sha(open(cur, "rb").read()) == V1_SHA, "the folder's notebook is not version 1; restore _v1 first"
        shutil.copyfile(cur, V1)
    raw = open(V1, "rb").read()
    assert sha(raw) == V1_SHA, "_v1 notebook differs from version 1"
    xw, s6t = open(XW, "rb").read(), open(S6T, "rb").read()
    assert sha(s6t) == S6T_SHA, "s6fall trainer differs"
    nb = json.loads(raw)
    nb2 = copy.deepcopy(nb)
    cells = nb2["cells"]
    src = ["".join(c["source"]) for c in cells]
    assert src[9].startswith("ARM = 'E_lrprior'") and src[10].startswith("deadline = T0 + DEADLINE_HOURS * 3600")
    assert src[12].startswith("# stage a2b: nartaa's s6fall") and src[14].startswith("# Stage 3: per-finding rank blend")
    b9 = replace_once(src[9], "WORKER = 'compact_infer.py'", "WORKER = 'compact_infer.py'", "worker line")
    new9 = (ASSETS_NEW.replace("__XW_B64__", base64.b64encode(zlib.compress(xw, 9)).decode()).replace("__XW_SHA__", sha(xw))
            .replace("__S6T_B64__", base64.b64encode(zlib.compress(s6t, 9)).decode()).replace("__S6T_SHA__", S6T_SHA)
            .replace("__S6F_SHA__", S6F_SHA))
    b9 = replace_once(b9, ASSETS_OLD, new9, "assets print")
    b10 = replace_once(src[10], CTRL_OLD, CTRL_NEW, "controller copy")
    b14 = replace_once(src[14], A3_DROP_OLD, A3_DROP_NEW, "s6fall drop rule")
    cells[14]["source"] = b14.splitlines(keepends=True)
    cells[9]["source"] = b9.splitlines(keepends=True)
    cells[10]["source"] = b10.splitlines(keepends=True)
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": HEADER.splitlines(keepends=True)}
    del cells[11:13]                                                   # the a2b stage (markdown + code)
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            s = "".join(c["source"])
            if not s.lstrip().startswith("%%"):
                compile(s, "cell", "exec")
    text = json.dumps(nb2, ensure_ascii=False, indent=1)
    low = text.lower()
    assert "claude" not in low and "anthropic" not in low, "tool name in the notebook"
    assert "s6f_solo" not in "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    out_nb = f"{OUT_DIR}/rsna-knee-final-max.ipynb"
    open(out_nb, "w", encoding="utf-8", newline="\n").write(text + "\n")
    meta = json.load(open(f"{OUT_DIR}/kernel-metadata.json", encoding="utf-8"))
    assert meta["id"] == "nishkarshsinghal/rsna-knee-final-max" and meta["is_private"] is True
    assert "nartaa/rsna-knee-s6fall-weights" in meta["dataset_sources"]
    print("v1", V1_SHA[:8], "| worker_x", sha(xw)[:16], "| s6 trainer", S6T_SHA[:8])
    print("wrote", out_nb, sha(open(out_nb, "rb").read())[:16], "| cells", len(cells))


if __name__ == "__main__":
    main()
