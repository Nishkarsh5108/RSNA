"""Hidden-test inference: DICOM -> study index -> selected-series cache -> model ensemble -> submission.

Failure policy (never zero-fill):
  * missing pixel decoders                      -> abort (assert_decoders)
  * a series that cannot be decoded             -> try the next candidate series of the same slot;
                                                   if none is left the slot is explicitly MASKED
                                                   (the model was trained with masks) and logged
  * a study with no decodable series at all     -> abort
  * more than MAX_FAILED_STUDY_FRACTION of studies with any failure -> abort (systematic problem)

Each checkpoint records the preprocessing it was trained with (args["preproc"]: "v2" = 224 px squash,
"v6" = 0.6 mm/px aspect-preserving 256 px crop; older checkpoints without the key are v2). Series are
decoded once per preprocessing version into cache_dir/<version>/, so v2 and v6 models can be ensembled.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from data import KneeStudyDataset, pad_collate
from infer_lib import DecodeError, build_index_row, preprocess_series, preprocess_series_hr, preprocess_series_multi, preprocess_series_v6, process_pool, series_header_geometry, series_header_geometry_fast  # noqa: E501
from model import KneeMIL
from study_index import SLOT_NAMES, TARGETS

MAX_FAILED_STUDY_FRACTION = 0.02


def assert_decoders():
    from pydicom.pixels import get_decoder
    from pydicom.uid import JPEG2000, JPEG2000Lossless, JPEGLossless, JPEGLosslessSV1
    missing = [u.name for u in (JPEGLosslessSV1, JPEGLossless, JPEG2000Lossless, JPEG2000) if not get_decoder(u).is_available]
    if missing:
        raise RuntimeError(f"No pixel decoder for {missing}: install pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg (or gdcm)")


def _geo_job(args):
    sdir, sid, fast = args if len(args) == 3 else (*args, False)
    try:
        return sid, (series_header_geometry_fast(sdir) if fast else series_header_geometry(sdir)), None
    except Exception as ex:
        return sid, None, f"{type(ex).__name__}: {ex}"


def build_test_index(series_csv, dicom_root, workers=4, log=print, fast=False):
    """fast=True: one header per series (series_header_geometry_fast) instead of every header."""
    ser = pd.read_csv(series_csv)
    jobs = [(os.path.join(dicom_root, r.StudyInstanceUID, r.SeriesInstanceUID), r.SeriesInstanceUID, fast) for r in ser.itertuples()]
    geos, failures = {}, []
    with process_pool(workers) as ex:
        for sid, g, err in ex.map(_geo_job, jobs, chunksize=8):
            if g is None:
                failures.append({"SeriesInstanceUID": sid, "stage": "header", "error": err})
            else:
                geos[sid] = g
    rows = []
    for st, meta in ser.groupby("StudyInstanceUID"):
        ok = meta[meta.SeriesInstanceUID.isin(geos)]
        if ok.empty:
            raise RuntimeError(f"study {st}: no readable series headers")
        rows.append(build_index_row(st, ok, geos))
    log(f"test index: {len(rows)} studies, {len(geos)} series headers ok, {len(failures)} header failures")
    return pd.DataFrame(rows), failures, geos


def _decode_job(args):
    """-> (series uid, error or None, dropped end-slice file names). out = path (one version `preproc`) or
    {version: path} (several versions from ONE decode, infer_lib.preprocess_series_multi)."""
    sdir, sid, out, preproc = args
    outs = out if isinstance(out, dict) else {preproc: out}
    if all(os.path.exists(p) for p in outs.values()):
        return sid, None, []
    try:
        if len(outs) > 1:
            arrs, geo = preprocess_series_multi(sdir, list(outs))
        elif preproc == "v6":
            a, geo = preprocess_series_v6(sdir); arrs = {preproc: a}
        elif preproc == "hr":
            a, geo = preprocess_series_hr(sdir); arrs = {preproc: a}
        else:
            a, geo = preprocess_series(sdir, 224); arrs = {preproc: a}
        for v, p in outs.items():
            tmp = p + ".tmp.npy"               # atomic: an interrupted write never looks like a cached series
            np.save(tmp, arrs[v])
            os.replace(tmp, p)
        return sid, None, geo.get("dropped_slices", [])
    except Exception as ex:
        return sid, f"{type(ex).__name__}: {ex}", []


def preprocess_selected(index_df, geos, dicom_root, cache_dir, slots, workers=4, preproc="v2", log=print):
    """Decode only the series needed for `slots`. Candidate order per slot: the slot's own
    candidates, then (FS slots whose non-FS twin is not itself a model slot) the plane's non-FS
    candidates. On a decode failure the next candidate is tried; if none is left the slot is
    explicitly masked. UID and 2.5D-step columns are rewritten to what was actually decoded."""
    multi = isinstance(preproc, (list, tuple))       # several versions decoded once: cache_dir/<version>/<uid>.npy
    for d in ([os.path.join(cache_dir, v) for v in preproc] if multi else [cache_dir]):
        os.makedirs(d, exist_ok=True)
    idx = index_df.copy().reset_index(drop=True)
    plan = {}
    for ri, r in idx.iterrows():
        for si in slots:
            own = [c for c in str(r[f"{SLOT_NAMES[si]}_cands"] or "").split("|") if c and c != "nan"]
            fb = []
            if si < 3 and (si + 3) not in slots:
                fb = [c for c in str(r[f"{SLOT_NAMES[si + 3]}_cands"] or "").split("|") if c and c != "nan"]
            plan[(ri, si)] = [(si, c) for c in own] + [(si + 3, c) for c in fb]
    chosen = {k: None for k in plan}
    failures, failed_studies = [], set()
    with process_pool(workers) as ex:
        while True:
            jobs = {k: v[0] for k, v in plan.items() if chosen[k] is None and v}
            if not jobs:
                break
            args = [(os.path.join(dicom_root, idx.at[k[0], "StudyInstanceUID"], c), c,
                     {v: os.path.join(cache_dir, v, f"{c}.npy") for v in preproc} if multi else os.path.join(cache_dir, f"{c}.npy"),
                     None if multi else preproc) for k, (_, c) in jobs.items()]
            for k, (sid, err, dropped) in zip(list(jobs.keys()), ex.map(_decode_job, args, chunksize=4)):
                if dropped:                     # non-fatal: one undecodable end slice dropped (infer_lib.decode_series)
                    failures.append({"StudyInstanceUID": idx.at[k[0], "StudyInstanceUID"], "slot": SLOT_NAMES[k[1]],
                                     "SeriesInstanceUID": sid, "stage": "dropped_end_slice", "error": "|".join(dropped)})
                if err is None:
                    chosen[k] = jobs[k]
                else:
                    failures.append({"StudyInstanceUID": idx.at[k[0], "StudyInstanceUID"], "slot": SLOT_NAMES[k[1]],
                                     "SeriesInstanceUID": sid, "stage": "decode", "error": err})
                    failed_studies.add(idx.at[k[0], "StudyInstanceUID"])
                    plan[k] = plan[k][1:]
    for (ri, si), ch in chosen.items():
        for col in {si, si + 3} if si < 3 and (si + 3) not in slots else {si}:
            idx.at[ri, f"{SLOT_NAMES[col]}_uid"] = None
        if ch is None:
            continue
        col, sid = ch
        idx.at[ri, f"{SLOT_NAMES[col]}_uid"] = sid
        idx.at[ri, f"{SLOT_NAMES[col]}_step"] = int(max(1, round(3.5 / geos[sid]["spacing_mm"])))
    n = len(idx)
    for ri, k_ok in pd.Series({ri: any(chosen[(ri, si)] is not None for si in slots) for ri in range(n)}).items():
        if not k_ok:
            raise RuntimeError(f"study {idx.at[ri, 'StudyInstanceUID']}: no decodable series for any model slot")
    if len(failed_studies) / max(n, 1) > MAX_FAILED_STUDY_FRACTION:
        raise RuntimeError(f"{len(failed_studies)}/{n} studies had decode failures (> {MAX_FAILED_STUDY_FRACTION:.0%}); "
                           f"first: {failures[:3]}")
    n_masked = sum(1 for v in chosen.values() if v is None)
    n_fail = sum(f["stage"] == "decode" for f in failures)
    log(f"[{'+'.join(preproc) if multi else preproc}] decoded slots for {n} studies; {n_fail} series failures in {len(failed_studies)} studies; "
        f"{n_masked} slot(s) masked; {len(failures) - n_fail} series with a dropped end slice")
    return idx, failures


def load_models(model_dir, device="cuda", select=None):
    """model_dir contains one or more <name>.pt checkpoints saved by train.py (key 'model' + 'args').
    select: None = every .pt in the folder; or {file name: weight} = only those files, with these ensemble
    weights (overriding the weight packaged in each checkpoint). Unknown names raise an error."""
    models = []
    files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pt"))
    if select:
        missing = [f for f in select if f not in files]
        if missing:
            raise RuntimeError(f"selected checkpoints not found in {model_dir}: {missing}")
        files = [f for f in files if f in select]
    for f in files:
        ck = torch.load(os.path.join(model_dir, f), map_location="cpu", weights_only=False)
        a = dict(ck["args"])
        if select:
            a["ens_weight"] = float(select[f])
        m = KneeMIL(a["backbone"], pretrained=False, n_slots=6, agg=a["agg"], use_pos=not a.get("no_pos", False))
        m.load_state_dict(ck["model"])
        m.eval().to(device)
        models.append((f, m, a))
    if not models:
        raise RuntimeError(f"no .pt checkpoints in {model_dir}")
    return models


def amp_dtype(device="cuda"):
    """bf16 (the training precision) on Ampere or newer; fp16 on older GPUs such as Kaggle's T4, where
    bf16 is only emulated (slow). Measured on 120 val studies (check_amp_precision.py): max |p - p_fp32|
    = 0.0014 for fp16 and 0.013 for bf16; per-class Spearman vs fp32 >= 0.9998 for both."""
    if str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def window_bags(a, tta_shift=False):
    """Window-centre settings (K, lo, hi) per model: the training setting, plus (tta_shift) a second
    bag shifted by half the spacing between window centres; the bags' probabilities are averaged.
    A checkpoint whose args set "infer_K" = 0 is evaluated on every slice of the span (one bag)."""
    if a.get("infer_K", None) == 0:
        return [(0, a["lo"], a["hi"])]
    k = a.get("infer_K") or a["K"]              # HR checkpoints: infer_K = 12 (trained with K = 6 random windows)
    bags = [(k, a["lo"], a["hi"])]
    if tta_shift:
        half = (a["hi"] - a["lo"]) / (2 * (k - 1))
        bags.append((k, a["lo"] + half, min(1.0, a["hi"] + half)))
    return bags


@torch.no_grad()
def predict_studies(index_df, cache_dir, models, batch_size=4, workers=2, device="cuda", tta_shift=False):
    """-> list with one (N, 12) probability array per model (rows aligned with index_df)."""
    preds = []
    for name, m, a in models:
        slots = [int(s) for s in str(a["slots"]).split(",")]
        pb = []
        for K, lo, hi in window_bags(a, tta_shift):
            ds = KneeStudyDataset(index_df, cache_dir, slots, K, train=False, lo=lo, hi=hi,
                                  orient=not a.get("no_orient", False),
                                  sag_pos_fix=a.get("sag_pos_fix", False), crop=a.get("crop", 0), k_max=a.get("k_max", 64))
            bs = batch_size if (K > 0 and not a.get("crop", 0)) else 1     # all-window / 320 px bags: one study per batch
            dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=workers,
                                             collate_fn=pad_collate if K == 0 else None)
            p = np.zeros((len(ds), len(TARGETS)), np.float32)
            for b in dl:
                x = b["x"].to(device).float().div_(255)
                wm = b["wmask"].to(device) if "wmask" in b else None
                with torch.autocast("cuda", dtype=amp_dtype(device)):
                    lg = m(x, b["mask"].to(device), b["pos"].to(device), wm)
                p[b["i"].numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
            pb.append(p)
        preds.append(np.mean(pb, 0))
    return preds


def write_submission(index_df, probs, sample_sub_csv, test_csv, out_path):
    sample = pd.read_csv(sample_sub_csv)
    test = pd.read_csv(test_csv)
    sub = pd.DataFrame(probs, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", index_df.StudyInstanceUID.values)
    sub = sub.set_index("StudyInstanceUID").reindex(test.StudyInstanceUID).reset_index()
    sub = sub[list(sample.columns)]
    vals = sub[TARGETS].values
    assert list(sub.columns) == list(sample.columns), "column mismatch with sample_submission"
    assert len(sub) == len(test) and sub.StudyInstanceUID.is_unique, "row count / uniqueness mismatch"
    assert set(sub.StudyInstanceUID) == set(test.StudyInstanceUID), "study id mismatch"
    assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all(), "non-finite or out-of-range predictions"
    sub.to_csv(out_path, index=False)
    return sub


def run(cfg, log=print):
    t0 = time.time()
    assert_decoders()
    idx, f1, geos = build_test_index(cfg["test_series_csv"], cfg["test_dicom_root"], cfg["workers"], log,
                                     fast=cfg.get("fast_index", False))
    models = load_models(cfg["model_dir"], cfg["device"], cfg.get("select_models"))
    groups = {}
    for mm in models:
        groups.setdefault(mm[2].get("preproc", "v2"), []).append(mm)
    f2, ready = [], {}
    slot_sets = {pp: sorted({int(s) for _, _, a in ms for s in str(a["slots"]).split(",")}) for pp, ms in groups.items()}
    if len(groups) > 1 and len({tuple(v) for v in slot_sets.values()}) == 1:
        # every preprocessing version uses the same slots: decode each series ONCE (bit-identical arrays)
        pps = sorted(groups)
        idx_pp, fails = preprocess_selected(idx, geos, cfg["test_dicom_root"], cfg["cache_dir"], slot_sets[pps[0]],
                                            cfg["workers"], preproc=pps, log=log)
        f2 += [{**f, "preproc": "+".join(pps)} for f in fails]
        ready = {pp: idx_pp for pp in pps}
    else:
        for pp, ms in sorted(groups.items()):
            idx_pp, fails = preprocess_selected(idx, geos, cfg["test_dicom_root"], os.path.join(cfg["cache_dir"], pp), slot_sets[pp],
                                                cfg["workers"], preproc=pp, log=log)
            f2 += [{**f, "preproc": pp} for f in fails]
            ready[pp] = idx_pp                # same row order as idx
    t1 = time.time()
    preds, wts = [], []
    for pp, ms in sorted(groups.items()):
        preds += predict_studies(ready[pp], os.path.join(cfg["cache_dir"], pp), ms, cfg["batch_size"],
                                 cfg.get("loader_workers", 2), cfg["device"], cfg.get("tta_shift", False))
        wts += [float(a.get("ens_weight", 1.0)) for _, _, a in ms]
    wts = np.asarray(wts) / np.sum(wts)       # equal weight per model unless a checkpoint sets args["ens_weight"]
    probs = np.tensordot(wts, np.asarray(preds), axes=1)
    sub = write_submission(idx, probs, cfg["sample_submission"], cfg["test_csv"], cfg["out_path"])
    t2 = time.time()
    report = {"n_studies": len(sub), "n_models": len(models), "models": [n for n, _, _ in models],
              "ensemble_weights": [round(float(w), 4) for w in wts],
              "preproc_versions": {pp: len(ms) for pp, ms in groups.items()}, "tta_shift": bool(cfg.get("tta_shift", False)),
              "header_failures": f1, "decode_failures": f2,
              "fast_index": bool(cfg.get("fast_index", False)),
              "seconds_preprocess": round(t1 - t0, 1), "seconds_predict": round(t2 - t1, 1),
              "side_source_counts": idx.side_source.value_counts().to_dict(),
              "side_conflicts": int(idx.side_conflict.sum()), "side_ambiguous": int(idx.side_ambiguous.sum())}
    json.dump(report, open(os.path.join(os.path.dirname(cfg["out_path"]) or ".", "inference_report.json"), "w"), indent=1, default=str)
    log(json.dumps({k: v for k, v in report.items() if "failures" not in k}))
    return sub, report
