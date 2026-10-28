"""Build rsna-knee-final-max (plan section 13, Track A; owner's brief 2026-10-20 ~18:05 IST, final pick 1).

goodpjw2008's v27 notebook (0.958; nartaa CoAtNet 384 + 224, heliosli R6 with flip TTA, goodpjw2008 reader at 1/10) with:
  - new stage a2b after R6: nartaa's s6fall inference script, exactly final-strict's stage 2b (the four-model notebook's
    v2 patch: the 0.949 CoAtNet is read from stage 1's raptor_raw.csv), started only with >= 45 min left before
    NB_T0 + 8.35 h (the brief); its hang guard gets the time left minus 10 min.
  - A3 blend: compact slot = 0.60 R6 + 0.40 s6fall (R6 alone where s6fall is missing; s6fall dropped if > 3 % of studies
    failed), Raptor weight per finding = v27's W_RAPTOR_C + Fracture 0.65; transform t1 (v27's percentile ranks) or t2
    (0.5 t1 + 0.5 pct of the same weights on per-leg standardised logits), chosen on gold_58 before the push
    (final_max_gold_eval.py; owner's rule).
  - everything else byte-identical to v27 (Parts A1, A1b, A2, B). The unused R7 dataset is not attached.
External data: OAI (CoAtNets), SKM-TEA (anatomy.pt anchors of R6 and s6fall), fastMRI+ (R6's auxiliary training task).

Usage: python -I src/emergency/build_final_max.py --transform t1|t2
Writes notebooks/candidate_A/final_max/{rsna-knee-final-max.ipynb, kernel-metadata.json}.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
import zlib

R = "D:/Coding/kaggle/RSNA"
V27 = ("C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/pub_nb/"
       "goodpjw2008_rsna-knee-coatnet-r6-2-5d-mil-lb-0-958/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958.ipynb")
V27_SHA = "a906df9328b6a53d7e0fe4e18962c655867c8617c54e6745a8af647033b2399c"      # version 27, pulled 2026-10-19
S6F_V2 = f"{R}/notebooks/candidate_A/final_strict/_build/s6f_solo_v2.py"           # final-strict's stage 2b script
OUT_DIR = f"{R}/notebooks/candidate_A/final_max"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


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
- new stage a2b, after r6: nartaa's s6fall inference script (public notebook
  [nartaa/rsna-knee-s6fall-solo](https://www.kaggle.com/code/nartaa/rsna-knee-s6fall-solo), weights
  [nartaa/rsna-knee-s6fall-weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights), cc by-nc 4.0) with the patch of
  [laimerbotrolenk/rsna-four-model-ensemble](https://www.kaggle.com/code/laimerbotrolenk/rsna-four-model-ensemble): the 0.949 coatnet
  is read from stage 1, not recomputed. it starts only if at least 45 minutes remain before 8.35 h.
- blend: compact slot = 0.60 r6 + 0.40 s6fall (r6 alone if s6fall did not run); raptor weight per finding as in v27, plus
  fracture 0.65; __TRANSFORM_NOTE__
- everything else is unchanged.

**external data:** oai (nartaa's coatnets), skm-tea (`anatomy.pt`, the knee anchors of r6 and s6fall), and fastmri+ (r6's
auxiliary training task). labels come from the competition's own reports.

all credit for the models and code goes to their authors (below). the source notebooks are released under apache 2.0.
"""

TRANSFORM_NOTE = {
    "t1": "per-finding percentile ranks, as in v27.",
    "t2": "half per-finding percentile ranks, half the rank of the same weighted blend on standardised logits.",
}

S6F_MD = """---
# part a2b: nartaa's all-data s6 convnext (s6fall)

nartaa's own inference script, unchanged apart from the four-model notebook's patch (the 0.949 coatnet comes from stage 1's
`raptor_raw.csv`) and a hang guard set to the time left. the script checks every code file and the checkpoint against nartaa's
scored hashes. if it is skipped, fails, or misses more than 3 % of studies, the blend below uses r6 alone in the compact slot."""

S6F_CODE = '''# stage a2b: nartaa's s6fall (all-data s6 convnext). starts only with >= 45 min left before NB_T0 + 8.35 h.
import base64, zlib, shutil
from pathlib import Path
S6F_OK, S6F_SECONDS = False, None
S6F_PROBS = WORK / 'k1' / 'B' / 'B_probs.csv'
S6F_SRC_B64 = "__S6F_SRC_B64__"
S6F_SRC_SHA256 = "__S6F_SRC_SHA256__"
for _d in (WORK / 'k1', Path('/tmp/k1_s6f')):
    if _d.exists():
        shutil.rmtree(_d)
_left = NB_T0 + 8.35 * 3600 - time.time()
if _left < 45 * 60:
    print('S6F_SKIPPED: %.0f min left' % (_left / 60), flush=True)
else:
    _t = time.time()
    _guard = _left - 10 * 60                     # 10 min margin for the blend and the reader check
    try:
        _src = zlib.decompress(base64.b64decode(S6F_SRC_B64))
        import hashlib as _hl
        assert _hl.sha256(_src).hexdigest() == S6F_SRC_SHA256, 's6fall script hash differs'
        Path('/tmp/s6f_solo.py').write_bytes(_src)
        _env = dict(os.environ, S6_GUARD_S=str(int(_guard)), K1_KAGGLE_ROOT='/kaggle', K1_TMP='/tmp/k1_s6f')
        _rc = subprocess.run([sys.executable, '-u', '/tmp/s6f_solo.py'], env=_env, timeout=_guard + 120).returncode
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

A3_HEAD_OLD = "# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5.  Not fitted on any labels:"
A3_HEAD_NEW = ("# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5.  Not fitted on any labels:\n"
               "# [max] compact slot = 0.60 r6 + 0.40 s6fall (r6 alone where s6fall is missing), fracture raptor weight 0.65, "
               "transform __TRANSFORM__ (owner's brief 2026-10-20)")

WC_OLD = """W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}   # [wc]"""
WC_NEW = """W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35, 'Fracture': 0.65}   # [max] + fracture 0.65 (owner's brief)   # [wc]"""

LOOP_OLD = """out = np.zeros((n, len(labels)))
rho_rr, rho_ro = {}, {}
for j, c in enumerate(labels):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows
    if r224 is not None:   # [s5] mix the 224 model into the Raptor slot, over Raptor's non-failed rows
        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)"""

LOOP_NEW = """W_R6, W_S6F = 0.60, 0.40       # [max] compact slot split (owner's brief), fixed before any run
TRANSFORM = '__TRANSFORM__'   # [max] t1 = percentile ranks (v27); t2 = 0.5 t1 + 0.5 pct(same weights on standardised logits)
s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'not run'
if globals().get('S6F_OK'):
    _b = pd.read_csv(S6F_PROBS, dtype={'StudyInstanceUID': str})
    if sorted(_b['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_b.columns):
        s6f = _b.set_index('StudyInstanceUID').loc[test_ids, labels].astype(float)
        s6f_fail = ~np.isfinite(s6f.values).all(axis=1)
        s6f_note = 'used'
        if s6f_fail.sum() > 0.03 * n:
            s6f, s6f_fail, s6f_note = None, np.ones(n, bool), 'dropped: %d failed studies' % int(s6f_fail.sum())
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
    return b


out = np.zeros((n, len(labels)))
rho_rr, rho_ro, rho_rs = {}, {}, {}
for j, c in enumerate(labels):
    b = slot_blend(pct_on, c, 0.5)
    if TRANSFORM == 't2':
        b = 0.5 * b + 0.5 * pct(slot_blend(zl_on, c, 0.0))
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)
        if s6f is not None:
            rho_rs[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(s6f[c].values), method='spearman')), 4)"""

RECEIPT_OLD = "(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))"
RECEIPT_NEW = """blend.update({'final_max': {'compact': {'r6': W_R6, 's6fall': W_S6F}, 'transform': TRANSFORM, 's6fall': s6f_note,
                            's6fall_failed': int(s6f_fail.sum()) if s6f is not None else None,
                            's6fall_seconds': globals().get('S6F_SECONDS'), 'spearman_raptor_vs_s6fall': rho_rs}})   # [max]
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", required=True, choices=["t1", "t2"])
    a = ap.parse_args()
    raw = open(V27, "rb").read()
    assert sha(raw) == V27_SHA, "v27 notebook differs from the pulled version 27"
    s6f_src = open(S6F_V2, "rb").read()
    nb = json.loads(raw)
    nb2 = copy.deepcopy(nb)
    cells = nb2["cells"]
    src = ["".join(c["source"]) for c in cells]
    i_ctrl = next(i for i, s in enumerate(src) if s.startswith("deadline = T0 + DEADLINE_HOURS * 3600"))
    i_a3 = next(i for i, s in enumerate(src) if s.startswith(A3_HEAD_OLD))
    assert i_a3 == i_ctrl + 2, (i_ctrl, i_a3)                      # controller, A3 markdown, A3 code
    b = src[i_a3]
    b = replace_once(b, A3_HEAD_OLD, A3_HEAD_NEW.replace("__TRANSFORM__", a.transform), "a3 header")
    b = replace_once(b, WC_OLD, WC_NEW, "W_RAPTOR_C")
    b = replace_once(b, LOOP_OLD, LOOP_NEW.replace("__TRANSFORM__", a.transform), "blend loop")
    b = replace_once(b, RECEIPT_OLD, RECEIPT_NEW, "receipt")
    cells[i_a3]["source"] = b.splitlines(keepends=True)
    code = S6F_CODE.replace("__S6F_SRC_B64__", base64.b64encode(zlib.compress(s6f_src, 9)).decode()).replace(
        "__S6F_SRC_SHA256__", sha(s6f_src))
    cells[i_ctrl + 1:i_ctrl + 1] = [
        {"cell_type": "markdown", "metadata": {}, "source": S6F_MD.splitlines(keepends=True)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)}]
    # header: v27's diagram cell and its changelog are replaced; its references, OAI acknowledgement and thanks are kept verbatim
    notes = src[1]
    k = notes.index("## References")
    header = HEADER.replace("__TRANSFORM_NOTE__", TRANSFORM_NOTE[a.transform])
    credits = notes[k:] + ("\n- **[nartaa (Danial Zakaria)](https://www.kaggle.com/nartaa)** also for the all-data s6 convnext "
                           "(s6fall) and its inference script, and **[laimerbotrolenk](https://www.kaggle.com/laimerbotrolenk)** for the "
                           "stage patch that reuses the 0.949 coatnet.\n")
    cells[0] = {"cell_type": "markdown", "metadata": {}, "source": header.splitlines(keepends=True)}
    cells[1] = {"cell_type": "markdown", "metadata": {}, "source": credits.splitlines(keepends=True)}
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    for c in cells:                                                    # every code cell must still parse
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if not s.lstrip().startswith("%%"):
                compile(s, "cell", "exec")
    text = json.dumps(nb2, ensure_ascii=False, indent=1)
    low = text.lower()
    assert "claude" not in low and "anthropic" not in low, "tool name in the notebook"
    assert "r7cnxfsss" not in "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_nb = f"{OUT_DIR}/rsna-knee-final-max.ipynb"
    open(out_nb, "w", encoding="utf-8", newline="\n").write(text + "\n")
    meta = {"id": "nishkarshsinghal/rsna-knee-final-max", "title": "rsna-knee-final-max", "code_file": "rsna-knee-final-max.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "rostigor/rsna-knee-r6-fe055406",
                                "nartaa/rsna-knee-publication-swa-weights-20261007",
                                "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", "nartaa/rsna-knee-s6fall-weights"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": [],
            "docker_image_pinning_type": "original", "docker_image": DOCKER, "machine_shape": "NvidiaTeslaT4"}
    open(f"{OUT_DIR}/kernel-metadata.json", "w", encoding="utf-8", newline="\n").write(json.dumps(meta, indent=2) + "\n")
    print("v27", V27_SHA[:8], "| s6f_solo_v2", sha(s6f_src)[:16], "| transform", a.transform)
    print("wrote", out_nb, sha(open(out_nb, "rb").read())[:16], "| cells", len(cells), "| s6fall cells at", i_ctrl + 1,
          "| blend cell", i_a3 + 2)


if __name__ == "__main__":
    main()
