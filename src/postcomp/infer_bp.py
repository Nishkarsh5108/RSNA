"""Test-time inference for the box-pooling classifier family: knee side -> ROI boxes -> classifier(s) -> submission.csv.

Three passes over the test series, each with the public modules' own datasets (side.py, detector.py, classifier.py):
  1. side classifier on coronal / axial series      -> left / right per study (right when a study has none)
  2. ROI detector (flip TTA) on every series         -> 12 boxes per series
  3. box-pooling classifier(s) on every series       -> series probabilities; study = mean over its series,
                                                        then the mean over the models
A series that cannot be read or processed is skipped and reported, never zero-filled; a study with no usable
series gets the column means. Writes submission.csv, series_preds.csv (per model and series) and infer_log.json.

Environment:
  SERIES_ROOT / SERIES_CSV / SUBMISSION_CSV   override the test paths (local smoke test)
  SIDE_WEIGHTS, DET_WEIGHTS                   path, or "dir-hint/file" searched under /kaggle/input
  CLS_WEIGHTS                                 comma list of the same
  WORKERS (4), CLS_BATCH (8), OUT_DIR
  TEST_FAIL_BATCH=1                           rehearsal only: every multi-series classifier batch fails once
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import classifier as C
import detector as D
import side as S

T0 = time.time()
LABELS = C.LABELS


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def find(spec):
    """An existing path, or 'hint/file': the file named `file` under /kaggle/input whose path contains `hint`
    (the competition's DICOM tree is never walked)."""
    if os.path.exists(spec):
        return spec
    hint, name = spec.rsplit("/", 1) if "/" in spec else ("", spec)
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("competitions", "rsna-knee-abnormality-detection")]
        if name in files and hint in root:
            hits.append(os.path.join(root, name))
    assert len(hits) == 1, f"{spec}: expected exactly one match, found {hits}"
    print("weights", spec, "->", hits[0], flush=True)
    return hits[0]


class Safe(torch.utils.data.Dataset):
    """Items of `ds` with their index; an item that raises becomes an error record instead of killing the loader."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        try:
            item = self.ds[i]
            item["_index"] = i
            return item
        except Exception as ex:
            return {"_error": f"{type(ex).__name__}: {ex}", "_index": i}


def safe_collate(collate):
    def f(items):
        good = [s for s in items if "_error" not in s]
        bad = [(s["_index"], s["_error"]) for s in items if "_error" in s]
        return (collate(good) if good else None), [s["_index"] for s in good], bad
    return f


def loader_of(ds, collate, batch, workers):
    return torch.utils.data.DataLoader(Safe(ds), batch_size=batch, shuffle=False, num_workers=workers,
                                       collate_fn=safe_collate(collate), prefetch_factor=4 if workers else None)


def load(model, path, dev):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(dev).eval()


@torch.no_grad()
def main():
    comp = C.COMP
    root = Path(os.environ.get("SERIES_ROOT", comp / "test_series"))
    for m in (S, D, C):
        m.SERIES_ROOT = root
    series = pd.read_csv(os.environ.get("SERIES_CSV", comp / "test_series.csv")).reset_index(drop=True)
    sub = pd.read_csv(os.environ.get("SUBMISSION_CSV", comp / "sample_submission.csv"))
    out_dir = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
    workers, cls_batch = int(os.environ.get("WORKERS", 4)), int(os.environ.get("CLS_BATCH", 8))
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")   # CPU only for dress rehearsals
    amp = dict(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda")
    log(f"device {dev}")
    studies = list(sub.StudyInstanceUID)
    log(f"{len(studies)} studies, {len(series)} series ({series.StudyInstanceUID.nunique()} studies in the series table), root {root}")
    failures = []

    # 1. knee side
    side_model = load(S.SideClassifier(pretrained=False), find(os.environ["SIDE_WEIGHTS"]), dev).to(memory_format=torch.channels_last)
    ca = series[series.Anatomical_Plane.isin(["Coronal", "Axial"])].reset_index(drop=True)
    p_right = {}
    for x, _, bad in loader_of(S.SideDataset(ca, {}), S.collate, 8, workers):
        failures += [("side", ca.SeriesInstanceUID[i], e) for i, e in bad]
        if x is None:
            continue
        try:
            with torch.autocast(**amp):
                p = torch.sigmoid(side_model(S.side_volumes(x, dev), x["slice_series"].to(dev), x["series_depth"].to(dev)).float())
            for study, q in zip(x["study"], p.cpu().numpy()):
                p_right.setdefault(study, []).append(float(q))
        except Exception as ex:
            failures.append(("side-batch", ";".join(x["study"]), f"{type(ex).__name__}: {ex}"))
    all_studies = sorted(set(studies) | set(series.StudyInstanceUID))
    side = {s: ("right" if np.mean(p_right[s]) >= 0.5 else "left") if s in p_right else "right" for s in all_studies}
    n_left = sum(v == "left" for v in side.values())
    log(f"side: {len(p_right)} studies by the classifier, {len(all_studies) - len(p_right)} by default (right); left {n_left}, right {len(side) - n_left}")
    del side_model

    # 2. ROI boxes
    det = load(D.Detector(pretrained=False), find(os.environ["DET_WEIGHTS"]), dev)
    for part in (det.encoder, det.projections):
        part.to(memory_format=torch.channels_last)
    boxes = {}
    for groups, _, bad in loader_of(D.DetectorDataset(series, side), D.collate_volumes, 4, workers):
        failures += [("detector", series.SeriesInstanceUID[i], e) for i, e in bad]
        for g in groups or []:                       # one series per group: a bad series does not take others with it
            try:
                for _, uid, _, box, *_ in D.detect(det, [[g]], dev):
                    boxes[uid] = box
            except Exception as ex:
                failures.append(("detector-forward", g["series"][0], f"{type(ex).__name__}: {ex}"))
    # A series without boxes gets the median box of its plane (whole volume if the plane has none).
    plane_of = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
    med = {pl: np.median(np.stack([b for u, b in boxes.items() if plane_of[u] == pl]), 0)
           for pl in series.Anatomical_Plane.unique() if any(plane_of[u] == pl for u in boxes)}
    whole = np.tile(np.float32([0, 0, 0, 1, 1, 1]), (len(LABELS), 1))
    n_fallback = 0
    for uid in series.SeriesInstanceUID:
        if uid not in boxes:
            boxes[uid] = med.get(plane_of[uid], whole)
            n_fallback += 1
    roi = pd.DataFrame([(uid, LABELS[k], *b[k]) for uid, b in boxes.items() for k in range(len(LABELS))],
                       columns=["SeriesInstanceUID", "label", "x0", "y0", "z0", "x1", "y1", "z1"])
    log(f"detector: boxes for {len(boxes) - n_fallback} series, {n_fallback} by fallback")
    del det
    torch.cuda.empty_cache()

    # 3. classifiers
    specs = [s for s in os.environ["CLS_WEIGHTS"].split(",") if s]
    models = [(spec, load(C.BoxPoolClassifier(pretrained=False), find(spec), dev).to(memory_format=torch.channels_last)) for spec in specs]
    log(f"classifiers: {len(models)}: {specs}")
    rows = []                                        # (model index, series uid, 12 probabilities)
    ds = C.SeriesDataset(series, {}, side, roi)

    def forward(x, idx):
        """Rows for one collated batch; all models or nothing, so a batch is never half recorded."""
        if os.environ.get("TEST_FAIL_BATCH") == "1" and len(idx) > 1:      # rehearsal only: exercise the one-by-one path
            raise RuntimeError("TEST_FAIL_BATCH")
        image = C.frame_volumes(x, dev)
        ss, depth, bx = (x[k].to(dev, non_blocking=True) for k in ("slice_series", "series_depth", "boxes"))
        theta = C.crop_theta(bx)
        inp = C.sample_slices(image, ss, theta, C.CROP_PX)
        new = []
        with torch.autocast(**amp):
            for mi, (_, m) in enumerate(models):
                p = torch.sigmoid(m(inp, ss, depth, theta, C.pooling_boxes(bx)).float()).cpu().numpy()
                new += [(mi, series.SeriesInstanceUID[i], *q) for i, q in zip(idx, p)]
        return new

    n_retry = 0
    for x, idx, bad in loader_of(ds, C.collate, cls_batch, workers):
        failures += [("classifier", series.SeriesInstanceUID[i], e) for i, e in bad]
        if x is None:
            continue
        try:
            rows += forward(x, idx)
        except Exception as ex:                      # e.g. out of memory on one very long series: redo the batch one series at a time
            failures.append(("classifier-batch", ";".join(series.SeriesInstanceUID[i] for i in idx), f"{type(ex).__name__}: {ex}"))
            del x
            torch.cuda.empty_cache()
            for i in idx:
                try:
                    rows += forward(C.collate([ds[i]]), [i])
                    n_retry += 1
                except Exception as ex1:
                    failures.append(("classifier-series", series.SeriesInstanceUID[i], f"{type(ex1).__name__}: {ex1}"))
                    torch.cuda.empty_cache()
    log(f"classifier pass done; {n_retry} series recovered one at a time")
    sp = pd.DataFrame(rows, columns=["model", "SeriesInstanceUID", *LABELS])
    sp = sp.merge(series[["SeriesInstanceUID", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]], on="SeriesInstanceUID")
    sp.to_csv(out_dir / "series_preds.csv", index=False)
    # study = mean over its series per model, then the mean over the models
    study = sp.groupby(["model", "StudyInstanceUID"])[LABELS].mean().groupby("StudyInstanceUID").mean()
    pred = study.reindex(studies)
    n_missing = int(pred[LABELS[0]].isna().sum())
    pred = pred.fillna(study.mean() if len(study) else 0.5).fillna(0.5)
    out = sub[["StudyInstanceUID"]].copy()
    out[LABELS] = pred[LABELS].to_numpy()
    assert list(out.columns) == list(sub.columns) and len(out) == len(sub) and np.isfinite(out[LABELS].to_numpy()).all()
    out.to_csv(out_dir / "submission.csv", index=False)
    json.dump({"studies": len(studies), "series": len(series), "series_predicted": int(sp.SeriesInstanceUID.nunique()),
               "studies_without_prediction": n_missing, "detector_fallback": n_fallback, "failures": failures[:200],
               "n_failures": len(failures), "seconds": time.time() - T0, "models": specs},
              open(out_dir / "infer_log.json", "w"), indent=1)
    log(f"done: {len(out)} studies, {n_missing} without a prediction, {len(failures)} failures; first: {failures[:3]}")
    print(out.head(3).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
