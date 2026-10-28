"""Build candidate final-strict (fastMRI-free): A-strict + nartaa's all-data S6 ConvNeXt ("s6fall") as a third leg.

- Stages 1-2 are heliosli's 0.954 notebook unchanged (notebooks/candidate_A/a_strict): nartaa's 0.949 Raptor CoAtNet with
  mirror TTA, then heliosli's S6 (4 folds).
- New stage 2b runs nartaa's own s6fall script (public notebook nartaa/rsna-knee-s6fall-solo, 0.950 solo; weights
  nartaa/rsna-knee-s6fall-weights, CC BY-NC 4.0). It gets the same 'v2' patch the public four-model notebook
  (laimerbotrolenk/rsna-four-model-ensemble) applies: component A, their 0.949 kernel, is not recomputed; it is read from
  stage 1's raptor_raw.csv, and the hang guard comes from S6_GUARD_S.
- Blend (fixed before any run; no labels): per-finding percentile ranks, Raptor 0.5 / S6 0.25 / s6fall 0.25. This is the
  four-model notebook's family split (Raptor half, compact half).
  - s6fall stage skipped, failed, or more than 3 % of studies failed: exactly A-strict's 0.7 / 0.3.
  - A study that some leg failed on: the weights of the legs it has, renormalised.
- Training data of every leg: competition data, OAI (Raptor) and SKM-TEA anchors (S6, s6fall). No fastMRI.

Usage: python -I src/emergency/build_final_strict.py
Writes notebooks/candidate_A/final_strict/{rsna-knee-final-strict.ipynb, kernel-metadata.json, _build/s6f_solo_v2.py}.
"""
import base64
import copy
import hashlib
import json
import os
import re
import zlib

R = "D:/Coding/kaggle/RSNA"
SRC_NB = f"{R}/notebooks/candidate_A/a_strict/rsna-knee-a-strict.ipynb"
OUT_DIR = f"{R}/notebooks/candidate_A/final_strict"
NARTAA = f"{OUT_DIR}/_build/nartaa_rsna-knee-s6fall-solo.py"
NARTAA_SHA = "6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe"   # after CRLF -> LF (the CLI pull writes CRLF)
A_STRICT_SHA = "1dc9a5df"            # prefix of the submitted A-strict notebook (0.954, submission 57384567)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def replace_once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected 1 match, found {n}")
    return s.replace(old, new)


def patch_v2(src):
    """The four-model notebook's v2 patch of nartaa's script (four edits; checked against its embedded copy)."""
    src, n = re.subn(r'^A_B64 = ".*"$', 'A_B64 = ""   # [v2] component A is reused from the notebook stage 1 (raptor_raw.csv)',
                     src, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("A_B64 line not found")
    src = replace_once(src, "A_TIMEOUT_S = 9000\n", "A_TIMEOUT_S = 60       # [v2] A is not run here\n", "A_TIMEOUT_S")
    src = replace_once(src, "B_TIMEOUT_S = 14400\n",
                       "B_TIMEOUT_S = 3600        # [v2] floor only; B gets whatever the guard has left\n", "B_TIMEOUT_S")
    src = replace_once(src, "GUARD_MAX_S = 480 * 60           # the hidden run's hang guard (8 h); + backstop 10 min stays under Kaggle's 9 h\n",
                       'GUARD_MAX_S = int(os.environ.get("S6_GUARD_S", str(480 * 60)))   # [v2] seconds this script may use\n',
                       "GUARD_MAX_S")
    src = replace_once(src, "\ndef run_a(I):\n", "\ndef _run_a_original_unused(I):\n", "run_a rename")
    new_run_a = (
        '\n\n\ndef run_a(I):\n'
        '    """[v2] component A = the notebook\'s own stage-1 output (nartaa\'s 0.949 kernel, same checkpoint), not recomputed here."""\n'
        '    p = WORK / "raptor_raw.csv"\n'
        '    df = pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")\n'
        '    A = df.loc[I.test_ids, LAB].to_numpy(np.float64)\n'
        '    A = np.where(np.isfinite(A), A, 0.5)\n'
        '    return A, {"reused": str(p), "rows": int(len(df))}')
    # insert right after the original run_a, before the component-B section (where the four-model notebook puts it)
    i = src.index("\ndef _run_a_original_unused(I):\n")
    j = src.index("\n\n\n# ---- component B", i)
    return src[:j] + new_run_a + src[j:]


STAGE_2B_MD = """## Stage 2b: nartaa's all-data S6 ConvNeXt (s6fall)

nartaa's own inference script (public notebook `nartaa/rsna-knee-s6fall-solo`, 0.950 solo), run unchanged apart from the
small patch the public four-model notebook applies: the 0.949 CoAtNet is not recomputed, it is read from stage 1's
`raptor_raw.csv`, and the hang guard takes the time left in this notebook. The script checks every code file and the
checkpoint against nartaa's scored hashes. If it is skipped or fails, the blend below is exactly the 0.954 recipe."""

STAGE_2B = '''# Stage 2b: nartaa's s6fall (all-data S6 ConvNeXt). Runs only with >= 100 min left before NB_T0 + 8.3 h.
import base64, zlib, shutil
from pathlib import Path
S6F_OK, S6F_SECONDS = False, None
S6F_PROBS = WORK / 'k1' / 'B' / 'B_probs.csv'
S6F_SRC_B64 = "__S6F_SRC_B64__"
S6F_SRC_SHA256 = "__S6F_SRC_SHA256__"
for _d in (WORK / 'k1', Path('/tmp/k1_s6f')):
    if _d.exists():
        shutil.rmtree(_d)
_left = NB_T0 + 8.3 * 3600 - time.time() - 10 * 60
if _left < 100 * 60:
    print('S6F_SKIPPED: %.0f min left' % (_left / 60), flush=True)
else:
    _t = time.time()
    try:
        _src = zlib.decompress(base64.b64decode(S6F_SRC_B64))
        import hashlib as _hl
        assert _hl.sha256(_src).hexdigest() == S6F_SRC_SHA256, 's6fall script hash differs'
        Path('/tmp/s6f_solo.py').write_bytes(_src)
        _env = dict(os.environ, S6_GUARD_S=str(int(_left)), K1_KAGGLE_ROOT='/kaggle', K1_TMP='/tmp/k1_s6f')
        _rc = subprocess.run([sys.executable, '-u', '/tmp/s6f_solo.py'], env=_env, timeout=_left + 120).returncode
        if (_rc != 0 or not S6F_PROBS.is_file() or not (WORK / 'k1' / 'k1_receipt.json').is_file()
                or (WORK / 'k1' / 'k1_error.json').exists()):
            raise RuntimeError('s6fall stage did not complete cleanly (exit code %s)' % _rc)
        S6F_OK = True
    except Exception as _e:
        print('S6F_STAGE_FAILED', repr(_e)[:400], flush=True)
    finally:
        if (WORK / 'submission.csv').exists():          # the script's own single-model file is never the submission
            (WORK / 'submission.csv').replace(WORK / '_s6f_solo_discarded.csv')
        S6F_SECONDS = time.time() - _t
        print('S6F_STAGE_DONE', S6F_OK, round(S6F_SECONDS, 1), flush=True)
'''

BLEND_HEAD_OLD = """# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.7, ours 0.3.  Not fitted on any labels:"""
BLEND_HEAD_NEW = """# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, S6 0.25, s6fall 0.25 (the public
# four-model notebook's family split); without s6fall (skipped, failed, > 3 % failed studies): Raptor 0.7, ours 0.3 = A-strict.
# Not fitted on any labels:"""

BLEND_LOOP_OLD = """out = np.zeros((n, len(labels)))
rho_rr, rho_ro = {}, {}
for j, c in enumerate(labels):
    rr, ro = pct(raw[c].values), pct(ours[c].values)
    b = W_RAPTOR * rr + W_OURS * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)"""

BLEND_LOOP_NEW = """W3 = {'raptor': 0.5, 'ours': 0.25, 's6f': 0.25}
s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'not run'
if S6F_OK:
    _b = pd.read_csv(S6F_PROBS, dtype={'StudyInstanceUID': str})
    if sorted(_b['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_b.columns):
        s6f = _b.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
        s6f_fail = ~np.isfinite(s6f.values).all(axis=1)
        s6f_note = 'used'
        if s6f_fail.sum() > 0.03 * n:
            s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'dropped: %d failed studies' % int(s6f_fail.sum())
    else:
        s6f_note = 'dropped: rows or columns differ from test.csv'
print('S6F_IN_BLEND', s6f_note, flush=True)
out = np.zeros((n, len(labels)))
rho_rr, rho_ro, rho_rs = {}, {}, {}
for j, c in enumerate(labels):
    rr, ro = pct(raw[c].values), pct(ours[c].values)
    if s6f is not None:
        v = s6f[c].values.copy()
        v[s6f_fail] = np.nanmean(v[~s6f_fail])
        rs = pct(v)
    else:
        rs = np.full(n, 0.5)
    legs = ((rr, ~rap_fail, W3['raptor'], W_RAPTOR), (ro, ~our_fail, W3['ours'], W_OURS), (rs, ~s6f_fail, W3['s6f'], 0.0))
    num, den = np.zeros(n), np.zeros(n)
    for r_, ok_, w3_, w2_ in legs:
        w = np.where(s6f_fail, w2_, w3_) * ok_       # three-leg weights where s6fall scored the study, else A-strict's
        num += w * r_
        den += w
    b = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.5)
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)
        if s6f is not None:
            rho_rs[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rs), method='spearman')), 4)"""

RECEIPT_OLD = """blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS}, 'rows': n, 'raptor_failed': int(rap_fail.sum()),"""
RECEIPT_NEW = """blend = {'weights': W3 if s6f is not None else {'raptor': W_RAPTOR, 'ours': W_OURS}, 's6fall': s6f_note,
         's6fall_failed': int(s6f_fail.sum()) if s6f is not None else None, 's6fall_seconds': S6F_SECONDS,
         'spearman_raptor_vs_s6fall': rho_rs, 'rows': n, 'raptor_failed': int(rap_fail.sum()),"""


def main():
    raw = open(NARTAA, "rb").read().replace(b"\r\n", b"\n")
    if sha(raw) != NARTAA_SHA:
        raise SystemExit("nartaa script hash differs from the downloaded public version")
    v2 = patch_v2(raw.decode("utf-8")).encode("utf-8")
    open(f"{OUT_DIR}/_build/s6f_solo_v2.py", "wb").write(v2)
    nb = json.load(open(SRC_NB, encoding="utf-8"))
    nb2 = copy.deepcopy(nb)
    cells = nb2["cells"]
    blend_i = next(i for i, c in enumerate(cells) if "".join(c["source"]).startswith(BLEND_HEAD_OLD))
    b = "".join(cells[blend_i]["source"])
    b = replace_once(b, BLEND_HEAD_OLD, BLEND_HEAD_NEW, "blend header")
    b = replace_once(b, BLEND_LOOP_OLD, BLEND_LOOP_NEW, "blend loop")
    b = replace_once(b, RECEIPT_OLD, RECEIPT_NEW, "blend receipt")
    cells[blend_i]["source"] = b.splitlines(keepends=True)
    cells[blend_i]["outputs"], cells[blend_i]["execution_count"] = [], None
    code = STAGE_2B.replace("__S6F_SRC_B64__", base64.b64encode(zlib.compress(v2, 9)).decode()).replace(
        "__S6F_SRC_SHA256__", sha(v2))
    new_cells = [{"cell_type": "markdown", "metadata": {}, "source": STAGE_2B_MD.splitlines(keepends=True)},
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": code.splitlines(keepends=True)}]
    cells[blend_i:blend_i] = new_cells
    t0 = "".join(cells[0]["source"])
    t0 = replace_once(t0, "# RSNA Knee: rank blend of the public Raptor model and our compact model (S6)",
                      "# RSNA Knee: rank blend of the public Raptor model, heliosli's S6 and nartaa's s6fall", "title")
    cells[0]["source"] = t0.splitlines(keepends=True)
    credits = "".join(cells[1]["source"])
    credits += ("\n\nnartaa's all-data S6 ConvNeXt (`nartaa/rsna-knee-s6fall-weights`, CC BY-NC 4.0) and its inference script "
                "(`nartaa/rsna-knee-s6fall-solo`), built on heliosli's S6 pipeline and the SKM-TEA anatomy checkpoint. The "
                "stage-2b patch follows `laimerbotrolenk/rsna-four-model-ensemble`.")
    cells[1]["source"] = credits.splitlines(keepends=True)
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    text = json.dumps(nb2, ensure_ascii=False, indent=1)
    low = text.lower()
    assert "claude" not in low and "anthropic" not in low, "tool name in the notebook"
    out_nb = f"{OUT_DIR}/rsna-knee-final-strict.ipynb"
    open(out_nb, "w", encoding="utf-8", newline="\n").write(text + "\n")
    meta = json.load(open(f"{R}/notebooks/candidate_A/a_strict/kernel-metadata.json", encoding="utf-8"))
    meta.update({"id": "nishkarshsinghal/rsna-knee-final-strict", "title": "rsna-knee-final-strict",
                 "code_file": "rsna-knee-final-strict.ipynb"})
    meta["dataset_sources"] = meta["dataset_sources"] + ["nartaa/rsna-knee-s6fall-weights"]
    open(f"{OUT_DIR}/kernel-metadata.json", "w", encoding="utf-8", newline="\n").write(json.dumps(meta, indent=2) + "\n")
    print("source A-strict sha", sha(open(SRC_NB, "rb").read())[:8], "(expected", A_STRICT_SHA + ")")
    print("s6f_solo_v2.py sha", sha(v2)[:16], len(v2), "bytes")
    print("wrote", out_nb, sha(open(out_nb, "rb").read())[:16], "| cells", len(cells), "| blend cell", blend_i + 2)


if __name__ == "__main__":
    main()
