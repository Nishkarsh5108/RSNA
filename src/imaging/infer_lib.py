"""DICOM -> model-input preprocessing for inference (and for re-building caches).

Reproduces the v2 training cache exactly (verified bit-exact on the local test studies by
src/imaging/check_infer_exactness.py):
  order slices by ImagePositionPatient . slice-normal (fallback InstanceNumber, then name)
  -> RescaleSlope/Intercept, MONOCHROME1 inversion, middle frame of multi-frame files
  -> per-series 1st/99th percentile window -> uint8 -> PIL bilinear resize to 224x224.
Differences from v2, on purpose: v2 silently skipped ANY slice that failed to decode. Here only one
undecodable slice at either end of the slice order is dropped, and it is reported; any other failure
raises DecodeError (see decode_series). An unreadable series is reported, never replaced by zeros.

The study index for unseen studies (slots, knee side, 2.5D step) is rebuilt from the DICOM
headers with the same rules as src/imaging/study_index.py.
"""
import os

import numpy as np
import pandas as pd
import pydicom
from PIL import Image

from study_index import SLOT_NAMES, TARGET_STEP_MM, choose_slots


class DecodeError(RuntimeError):
    pass


def process_pool(workers):
    """Process pool that also works for functions defined inside a notebook: 'fork' where available
    (Linux/Kaggle; Python 3.14 changed the POSIX default to 'forkserver', which cannot pickle them)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    if "fork" in mp.get_all_start_methods():
        return ProcessPoolExecutor(workers, mp_context=mp.get_context("fork"))
    return ProcessPoolExecutor(workers)


def order_series_files(series_dir):
    fnames = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not fnames:
        raise DecodeError(f"no .dcm files in {series_dir}")
    entries = []
    for f in fnames:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f))
        except Exception as ex:
            raise DecodeError(f"unreadable header {series_dir}/{f}: {type(ex).__name__}: {ex}")
        entries.append([None, f, ds])
    have_geo = True
    for e in entries:
        try:
            iop = np.array(e[2].ImageOrientationPatient, dtype=np.float64)
            ipp = np.array(e[2].ImagePositionPatient, dtype=np.float64)
            e[0] = float(np.dot(np.cross(iop[:3], iop[3:]), ipp))
        except Exception:
            have_geo = False
            break
    if have_geo and len({round(e[0], 3) for e in entries}) > 1:
        entries.sort(key=lambda e: e[0])
        return [(e[1], e[2]) for e in entries]
    try:
        nums = [int(getattr(e[2], "InstanceNumber", -1)) for e in entries]
        if len(set(nums)) == len(entries) and all(n >= 0 for n in nums):
            order = sorted(range(len(entries)), key=lambda i: nums[i])
            return [(entries[i][1], entries[i][2]) for i in order]
    except Exception:
        pass
    entries.sort(key=lambda e: e[1])
    return [(e[1], e[2]) for e in entries]


def decode_slice(ds):
    arr = ds.pixel_array.astype(np.float32)        # raises if no decoder is available
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    arr = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


def decode_series(ordered, series_dir):
    """Decode every slice of an ordered series -> (list of float arrays, list of dropped file names).

    Exactly ONE undecodable slice at either END of the slice order, in a series of >= 10 slices, is
    dropped and reported (seen twice in the 24k training series: a truncated first/last file; the
    legacy v2 cache dropped the same two slices). Any other decode failure raises DecodeError, so the
    caller falls back to another series or masks the slot; nothing is ever zero-filled."""
    slices, errs = [], []
    for i, (f, ds) in enumerate(ordered):
        try:
            slices.append(decode_slice(ds))
        except Exception as ex:
            errs.append((i, f, ds, ex))
    if not errs:
        return slices, []
    i, f, ds, ex = errs[0]
    if len(errs) == 1 and i in (0, len(ordered) - 1) and len(ordered) >= 10:
        return slices, [f]
    raise DecodeError(f"pixel decode failed {series_dir}/{f} (transfer syntax {getattr(ds.file_meta, 'TransferSyntaxUID', '?')}): "
                      f"{type(ex).__name__}: {ex} [{len(errs)} of {len(ordered)} slices failed]")


def _decoded_u8(series_dir):
    """Shared first stage of every preprocessing version: order, decode, per-series 1st/99th percentile window
    -> uint8 [T, H, W] (native matrix). -> (stack, ordered, dropped end-slice names)."""
    ordered = order_series_files(series_dir)
    slices, dropped = decode_series(ordered, series_dir)
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise DecodeError(f"inconsistent slice shapes {shapes} in {series_dir}")
    stack = np.stack(slices, 0)
    lo, hi = np.percentile(stack, [1.0, 99.0])
    hi = max(hi, lo + 1.0)
    stack = (np.clip((stack - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
    return stack, ordered, dropped


def _pixel_spacing(ds0, stack):
    try:
        return [float(v) for v in ds0.PixelSpacing]
    except Exception:
        return [160.0 / max(stack.shape[1:])] * 2                  # assume a 160 mm FOV if spacing is missing


def _resize_v2(stack, series_dir, img_size=224):
    out = np.stack([np.array(Image.fromarray(s).resize((img_size, img_size), Image.BILINEAR), dtype=np.uint8)
                    for s in stack])
    if not np.isfinite(out).all() or out.max() == out.min():
        raise DecodeError(f"degenerate (constant) series {series_dir}")
    return out


def _resize_v6(stack, ds0, series_dir, out_size=256, target_mm=0.6):
    ps_r, ps_c = _pixel_spacing(ds0, stack)
    H, W = stack.shape[1:]
    nh, nw = max(1, int(round(H * ps_r / target_mm))), max(1, int(round(W * ps_c / target_mm)))
    out = np.zeros((len(stack), out_size, out_size), np.uint8)
    h, w = min(out_size, nh), min(out_size, nw)                 # copied extent
    sy, sx = max(0, (nh - out_size) // 2), max(0, (nw - out_size) // 2)   # centred crop in the resized image
    dy, dx = max(0, (out_size - nh) // 2), max(0, (out_size - nw) // 2)   # centred placement (zero pad)
    for i, s in enumerate(stack):
        r = np.array(Image.fromarray(s).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
        out[i, dy:dy + h, dx:dx + w] = r[sy:sy + h, sx:sx + w]
    if out.max() == out.min():
        raise DecodeError(f"degenerate (constant) series {series_dir}")
    return out


HR_FIELD_MM, HR_GRID, HR_KEEP = 140.0, 384, 352


def _resize_hr(stack, ds0, series_dir, field_mm=HR_FIELD_MM, grid=HR_GRID, keep=HR_KEEP):
    ps_r, ps_c = _pixel_spacing(ds0, stack)
    mm = field_mm / grid
    H, W = stack.shape[1:]
    nh, nw = max(1, int(round(H * ps_r / mm))), max(1, int(round(W * ps_c / mm)))
    off = (grid - keep) // 2
    out = np.zeros((len(stack), keep, keep), np.uint8)
    # place the resized slice centred on the grid, then keep grid[off:off+keep] (zero pad where the FOV is smaller)
    gy, gx = (grid - nh) // 2, (grid - nw) // 2                 # top-left of the resized slice in grid coords
    y0, x0 = max(off, gy), max(off, gx)
    y1, x1 = min(off + keep, gy + nh), min(off + keep, gx + nw)
    for i, s in enumerate(stack):
        r = np.array(Image.fromarray(s).resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
        out[i, y0 - off:y1 - off, x0 - off:x1 - off] = r[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
    if out.max() == out.min():
        raise DecodeError(f"degenerate (constant) series {series_dir}")
    return out


def preprocess_series(series_dir, img_size=224):
    """v2: -> (uint8 [T, img_size, img_size], geometry dict). Raises DecodeError on failure (see decode_series)."""
    stack, ordered, dropped = _decoded_u8(series_dir)
    return _resize_v2(stack, series_dir, img_size), {**series_geometry(ordered), "dropped_slices": dropped}


def preprocess_series_v6(series_dir, out_size=256, target_mm=0.6):
    """v6: same decode/order/percentile as v2, then ASPECT-PRESERVING physical resampling to
    `target_mm` per pixel (PIL bilinear) and a centred out_size x out_size crop/zero-pad
    (256 x 0.6 mm = 153.6 mm field of view). -> (uint8 [T, out, out], geometry dict)."""
    stack, ordered, dropped = _decoded_u8(series_dir)
    return _resize_v6(stack, ordered[0][1], series_dir, out_size, target_mm), {**series_geometry(ordered), "dropped_slices": dropped}


def preprocess_series_hr(series_dir, field_mm=HR_FIELD_MM, grid=HR_GRID, keep=HR_KEEP):
    """hr (2026-10-18): the public high-resolution geometry (140 mm field on a 384 px grid = 0.365 mm/px,
    as in the public CoAtNet corpus) with the v2 decode/order/1-99 percentile. Only the centre `keep` px
    (352 = 128.3 mm) are stored: training crops 320 px with jitter, inference the centre 320 px.
    -> (uint8 [T, keep, keep], geometry dict). Raises DecodeError on failure (see decode_series)."""
    stack, ordered, dropped = _decoded_u8(series_dir)
    return _resize_hr(stack, ordered[0][1], series_dir, field_mm, grid, keep), {**series_geometry(ordered), "dropped_slices": dropped}


def preprocess_series_multi(series_dir, versions):
    """Decode ONCE and produce several preprocessing versions ('v2', 'v6', 'hr') from the same uint8 stack;
    each array is identical to the single-version function's output. -> ({version: array}, geometry dict)."""
    stack, ordered, dropped = _decoded_u8(series_dir)
    ds0, out = ordered[0][1], {}
    for v in versions:
        if v == "v2":
            out[v] = _resize_v2(stack, series_dir)
        elif v == "v6":
            out[v] = _resize_v6(stack, ds0, series_dir)
        elif v == "hr":
            out[v] = _resize_hr(stack, ds0, series_dir)
        else:
            raise ValueError(f"unknown preprocessing version {v}")
    return out, {**series_geometry(ordered), "dropped_slices": dropped}


def series_geometry(ordered):
    ds0 = ordered[0][1]
    geo = {"n_slices": len(ordered), "tag_side": "", "center_x_mm": np.nan, "spacing_mm": np.nan}
    lat = str(getattr(ds0, "Laterality", "") or "").strip().upper()[:1]
    geo["tag_side"] = lat if lat in ("L", "R") else ""
    try:
        iop = np.array(ds0.ImageOrientationPatient, dtype=float)
        ps = [float(v) for v in ds0.PixelSpacing]
        ipp = np.array(ds0.ImagePositionPatient, dtype=float)
        c = ipp + iop[:3] * ps[1] * int(ds0.Columns) / 2 + iop[3:] * ps[0] * int(ds0.Rows) / 2
        geo["center_x_mm"] = float(c[0])
    except Exception:
        pass
    # Same source priority as the training index (study_index.headers_table):
    # SpacingBetweenSlices -> SliceThickness -> measured position spacing -> 3.5 mm.
    for tag in ("SpacingBetweenSlices", "SliceThickness"):
        try:
            v = float(getattr(ds0, tag))
            if np.isfinite(v) and v > 0:
                geo["spacing_mm"] = v
                break
        except Exception:
            continue
    if not np.isfinite(geo["spacing_mm"]):
        try:
            iop = np.array(ds0.ImageOrientationPatient, dtype=float)
            n = np.cross(iop[:3], iop[3:])
            pos = [float(np.dot(n, np.array(d.ImagePositionPatient, dtype=float))) for _, d in ordered]
            geo["spacing_mm"] = float(np.median(np.abs(np.diff(pos)))) if len(pos) > 1 else 3.5
        except Exception:
            geo["spacing_mm"] = 3.5
    return geo


def series_header_geometry(series_dir):
    """Header-only pass (no pixel decoding): slice count, tag side, centre-x, spacing."""
    fnames = [f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not fnames:
        raise DecodeError(f"no .dcm files in {series_dir}")
    entries = []
    for f in fnames:
        try:
            ds = pydicom.dcmread(os.path.join(series_dir, f), stop_before_pixels=True)
        except Exception as ex:
            raise DecodeError(f"unreadable header {series_dir}/{f}: {type(ex).__name__}: {ex}")
        try:
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            key = float(np.dot(np.cross(iop[:3], iop[3:]), np.array(ds.ImagePositionPatient, dtype=float)))
        except Exception:
            key = 0.0
        entries.append((key, f, ds))
    entries.sort(key=lambda e: e[0])
    return series_geometry([(f, ds) for _, f, ds in entries])


def series_header_geometry_fast(series_dir):
    """Fast index pass (efficiency candidate): ONE header per series. series_geometry only reads the first slice's
    header for side tag, in-plane centre and spacing tags, and n_slices = number of .dcm files. Differences from the
    full pass: the centre comes from an arbitrary slice (identical for coronal/axial, which the side vote uses) and
    the measured-position spacing fallback (no spacing tags) is replaced by 3.5 mm."""
    fnames = sorted(f for f in os.listdir(series_dir) if f.lower().endswith(".dcm"))
    if not fnames:
        raise DecodeError(f"no .dcm files in {series_dir}")
    try:
        ds = pydicom.dcmread(os.path.join(series_dir, fnames[len(fnames) // 2]), stop_before_pixels=True)
    except Exception as ex:
        raise DecodeError(f"unreadable header {series_dir}: {type(ex).__name__}: {ex}")
    geo = series_geometry([(fnames[0], ds)])
    geo["n_slices"] = len(fnames)
    return geo


def study_side_from_geo(geos):
    """geos: list of geometry dicts (with 'plane') for one study's series; same policy as
    study_index.study_side: tag majority if any, else coronal/axial geometric majority."""
    tags = [g["tag_side"] for g in geos if g["tag_side"] in ("L", "R")]
    ca = [g for g in geos if g.get("plane") in ("Coronal", "Axial") and np.isfinite(g["center_x_mm"])]
    ca = ca if ca else [g for g in geos if np.isfinite(g["center_x_mm"])]
    conf = [g for g in ca if abs(g["center_x_mm"]) >= 20]
    src = conf if conf else ca
    geo_votes = ["L" if g["center_x_mm"] > 0 else "R" for g in src]
    geo = max(set(geo_votes), key=geo_votes.count) if geo_votes else "L"
    if tags:
        tag = max(set(tags), key=tags.count)
        conflict = bool(conf) and (max(set(["L" if g["center_x_mm"] > 0 else "R" for g in conf]),
                                       key=["L" if g["center_x_mm"] > 0 else "R" for g in conf].count) != tag)
        return tag, "tag", conflict, not conf
    return geo, "geometry", False, not conf


def build_index_row(study_uid, series_meta, geos):
    """series_meta: DataFrame (SeriesInstanceUID, Anatomical_Plane, Fluid_Sensitive) for one study;
    geos: dict series_uid -> geometry dict (from preprocess_series)."""
    ser = series_meta.copy()
    ser["n_slices"] = ser.SeriesInstanceUID.map(lambda s: geos[s]["n_slices"] if s in geos else 0)
    slots, cands = choose_slots(ser)
    plane_of = dict(zip(ser.SeriesInstanceUID, ser.Anatomical_Plane))
    side, src, conflict, amb = study_side_from_geo([{**geos[s], "plane": plane_of[s]} for s in ser.SeriesInstanceUID if s in geos])
    r = {"StudyInstanceUID": study_uid, "side": side, "side_source": src, "side_conflict": conflict, "side_ambiguous": amb}
    for si, name in enumerate(SLOT_NAMES):
        sid = slots[si]
        r[f"{name}_uid"], r[f"{name}_n_cands"], r[f"{name}_cands"] = sid, len(cands[si]), "|".join(cands[si])
        if sid is not None:
            sp = geos[sid]["spacing_mm"]
            r[f"{name}_n"], r[f"{name}_spacing"] = geos[sid]["n_slices"], sp
            r[f"{name}_step"] = int(max(1, round(TARGET_STEP_MM / sp)))
        else:
            r[f"{name}_n"], r[f"{name}_spacing"], r[f"{name}_step"] = 0, np.nan, 0
    return r
