"""DICOM inventory scan for the RSNA knee competition data.

Two passes, both resumable (results are appended per chunk to a CSV and already
scanned series are skipped on restart):

  * series pass  -- every series: file count, bytes, and the header of ONE file
                    (sorted by filename) for series-level tags.
  * slice pass   -- a random sample of series: headers of ALL files, so slice
                    geometry (spacing regularity, ordering, duplicates) can be
                    audited without decoding pixels.

Pixel data is never decoded here.

Usage:
  python dicom_inventory.py --root D:/.../train_series --out reports/eda/dicom \
      --workers 8 --slice-sample 1500
"""
import argparse
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom

SERIES_TAGS = [
    "PatientID", "PatientSex", "Manufacturer", "ManufacturerModelName",
    "MagneticFieldStrength", "SeriesDescription", "ScanningSequence",
    "SequenceVariant", "ScanOptions", "MRAcquisitionType", "SliceThickness",
    "SpacingBetweenSlices", "RepetitionTime", "EchoTime", "InversionTime",
    "EchoTrainLength", "FlipAngle", "PixelBandwidth", "ReceiveCoilName",
    "PatientPosition", "Laterality", "BodyPartExamined", "Rows", "Columns",
    "PixelSpacing", "BitsStored", "PhotometricInterpretation", "RescaleSlope",
    "RescaleIntercept", "ImageOrientationPatient", "SeriesNumber",
    "ContrastBolusAgent", "ImageType", "NumberOfFrames", "SoftwareVersions",
    "StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID",
]


def _val(ds, tag):
    v = getattr(ds, tag, None)
    if v is None:
        return None
    if isinstance(v, pydicom.multival.MultiValue) or isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v)


def _plane_from_iop(iop):
    """Dominant acquisition plane from ImageOrientationPatient (row x col normal)."""
    try:
        r = np.array(iop[:3], dtype=float)
        c = np.array(iop[3:6], dtype=float)
        n = np.cross(r, c)
        ax = int(np.argmax(np.abs(n)))
        return {0: "Sagittal", 1: "Coronal", 2: "Axial"}[ax], float(np.max(np.abs(n)))
    except Exception:
        return None, None


def scan_series(task):
    study, series, path, full = task
    out = {"StudyInstanceUID_dir": study, "SeriesInstanceUID_dir": series}
    try:
        entries = [e for e in os.scandir(path) if e.is_file()]
    except FileNotFoundError:
        out["status"] = "missing_dir"
        return out, None
    files = sorted(e.name for e in entries)
    out["n_files"] = len(files)
    out["n_dcm"] = sum(f.lower().endswith(".dcm") for f in files)
    out["bytes"] = int(sum(e.stat().st_size for e in entries))
    if not files:
        out["status"] = "empty_dir"
        return out, None
    try:
        ds = pydicom.dcmread(os.path.join(path, files[0]), stop_before_pixels=True)
    except Exception as ex:  # unreadable header
        out["status"] = f"unreadable:{type(ex).__name__}"
        return out, None
    out["status"] = "ok"
    out["TransferSyntax"] = str(getattr(ds.file_meta, "TransferSyntaxUID", ""))
    for t in SERIES_TAGS:
        out[t] = _val(ds, t)
    plane, conf = _plane_from_iop(getattr(ds, "ImageOrientationPatient", None) or [])
    out["iop_plane"] = plane
    out["iop_plane_conf"] = conf

    slice_rows = None
    if full:
        slice_rows = []
        n_bad = 0
        for f in files:
            try:
                d = pydicom.dcmread(os.path.join(path, f), stop_before_pixels=True)
            except Exception:
                n_bad += 1
                continue
            ipp = getattr(d, "ImagePositionPatient", None)
            iop = getattr(d, "ImageOrientationPatient", None)
            slice_rows.append({
                "SeriesInstanceUID_dir": series,
                "file": f,
                "InstanceNumber": getattr(d, "InstanceNumber", None),
                "ipp": "|".join(str(x) for x in ipp) if ipp is not None else None,
                "iop": "|".join(str(x) for x in iop) if iop is not None else None,
                "SliceLocation": getattr(d, "SliceLocation", None),
                "SliceThickness": getattr(d, "SliceThickness", None),
                "Rows": getattr(d, "Rows", None),
                "Columns": getattr(d, "Columns", None),
                "PixelSpacing": _val(d, "PixelSpacing"),
                "TransferSyntax": str(getattr(d.file_meta, "TransferSyntaxUID", "")),
            })
        out["n_unreadable_in_series"] = n_bad
    return out, slice_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--slice-sample", type=int, default=1500,
                    help="number of series whose every slice header is read")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N series")
    ap.add_argument("--seed", type=int, default=20261017)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    series_csv = os.path.join(args.out, "series_headers.csv")
    slices_csv = os.path.join(args.out, "slice_headers_sample.csv")

    tasks = []
    for study in sorted(os.listdir(args.root)):
        sp = os.path.join(args.root, study)
        if not os.path.isdir(sp):
            continue
        for series in sorted(os.listdir(sp)):
            tasks.append([study, series, os.path.join(sp, series), False])
    if args.limit:
        tasks = tasks[: args.limit]
    rng = random.Random(args.seed)
    full_idx = set(rng.sample(range(len(tasks)), min(args.slice_sample, len(tasks))))
    for i in full_idx:
        tasks[i][3] = True

    done = set()
    if os.path.exists(series_csv):
        prev = pd.read_csv(series_csv, usecols=["SeriesInstanceUID_dir"], dtype=str)
        done = set(prev.SeriesInstanceUID_dir)
    todo = [tuple(t) for t in tasks if t[1] not in done]
    print(f"series total={len(tasks)} done={len(done)} todo={len(todo)} "
          f"full-slice sample={len(full_idx)}", flush=True)

    t0 = time.time()
    buf, sbuf, n = [], [], 0

    def flush():
        nonlocal buf, sbuf
        if buf:
            pd.DataFrame(buf).to_csv(series_csv, mode="a", index=False,
                                     header=not os.path.exists(series_csv))
            buf = []
        if sbuf:
            pd.DataFrame(sbuf).to_csv(slices_csv, mode="a", index=False,
                                      header=not os.path.exists(slices_csv))
            sbuf = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(scan_series, t) for t in todo]
        for f in as_completed(futs):
            row, srows = f.result()
            buf.append(row)
            if srows:
                sbuf.extend(srows)
            n += 1
            if n % 500 == 0:
                flush()
                el = time.time() - t0
                print(f"{n}/{len(todo)} series  {n/el:.1f} series/s  "
                      f"eta {(len(todo)-n)/(n/el)/60:.1f} min", flush=True)
    flush()
    print(f"finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
