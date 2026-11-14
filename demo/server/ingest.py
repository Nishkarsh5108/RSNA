"""DICOM ingest for the local workstation.

Files (bytes) -> one study: its series written to the local cache in the layout the model modules read
(<root>/<study>/<series>/*.dcm), a display volume per series (uint8, windowed at the same 0.5 / 99.5 percentiles the
models use) and the series geometry in patient coordinates (LPS, mm) so the client can place every series in one scene.

Nothing here leaves the machine: the cache is a local folder, deleted when the study is purged or the server stops.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian

DISPLAY_MAX_PX = 512          # in-plane size cap of the display volume sent to the browser
MIN_SLICES = 6                # fewer slices: a localizer or a scout, not a diagnostic series
WINDOW = (0.5, 99.5)          # the models' window (side.py / classifier.py / detector.py)

# Fat suppression from the header. Checked on 600 training series against the dataset's own Fat_Suppression column:
# 99% agreement (SeriesDescription when present, else ScanOptions "FS" or an inversion time > 50 ms, i.e. STIR).
FS_POS = re.compile(r"(?<![a-z])(fs|fat ?sat|fatsat|spair|spir|stir|tirm|we|water|dixon|fsat|smart fat)(?![a-z])"
                    r"|_fs|fs_|spair|spir|stir|tirm", re.I)
FS_NEG = re.compile(r"non[ -]?fat|no[ -]?fs|non[ -]?fs", re.I)


def _tokens(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, pydicom.multival.MultiValue)):
        return [str(v).upper() for v in value]
    s = str(value)
    try:
        v = ast.literal_eval(s)
        return [str(t).upper() for t in (v if isinstance(v, list) else [v])]
    except Exception:
        return [t.upper() for t in re.split(r"[\\, ]+", s) if t]


def fat_suppressed(ds) -> tuple[bool, str]:
    desc = str(ds.get("SeriesDescription", "") or "")
    if desc and desc != "DummySeriesDesc!":
        if FS_NEG.search(desc):
            return False, f'description "{desc}" says non fat-suppressed'
        if FS_POS.search(desc):
            return True, f'description "{desc}"'
    if "FS" in _tokens(ds.get("ScanOptions")):
        return True, "ScanOptions FS"
    ti = float(ds.get("InversionTime", 0) or 0)
    if ti > 50:
        return True, f"inversion time {ti:.0f} ms (STIR)"
    return False, "no fat-suppression marker in the header"


def plane_of(iop) -> str:
    n = np.cross(np.asarray(iop[:3], float), np.asarray(iop[3:], float))
    return ("Sagittal", "Coronal", "Axial")[int(np.argmax(np.abs(n)))]


@dataclass
class Series:
    uid: str
    study: str
    plane: str
    fat_suppression: bool
    fs_reason: str
    description: str
    n_slices: int
    rows: int
    cols: int
    pixel_spacing: tuple[float, float]       # (row spacing, column spacing) mm
    slice_spacing: float
    iop: list[float]                         # row direction (3), column direction (3)
    positions: list[list[float]]             # ImagePositionPatient per slice, sorted along the normal
    display: np.ndarray | None = None        # uint8 [D, h, w]
    window: tuple[float, float] = (0.0, 1.0)
    tr: float | None = None
    te: float | None = None
    decompressed: bool = False
    include: bool = True
    note: str = ""

    def meta(self) -> dict:
        d = self.display
        return dict(uid=self.uid, plane=self.plane, fatSuppression=self.fat_suppression, fsReason=self.fs_reason,
                    description=self.description, slices=self.n_slices, rows=self.rows, cols=self.cols,
                    pixelSpacing=list(self.pixel_spacing), sliceSpacing=self.slice_spacing, iop=self.iop,
                    positions=self.positions, display=list(d.shape) if d is not None else None,
                    window=list(self.window), tr=self.tr, te=self.te, decompressed=self.decompressed,
                    include=self.include, note=self.note)


@dataclass
class Study:
    sid: str                                  # local session id
    uid: str                                  # StudyInstanceUID
    root: Path                                # <root>/<uid>/<series uid>/*.dcm
    series: list[Series] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)
    source: str = "upload"
    reference: dict | None = None             # reference labels (local sample studies only)
    created: float = field(default_factory=time.time)
    stats: dict = field(default_factory=dict)

    def manifest(self) -> dict:
        return dict(sid=self.sid, studyUid=self.uid, shortId=hashlib.sha1(self.uid.encode()).hexdigest()[:8].upper(),
                    series=[s.meta() for s in self.series], skipped=self.skipped, source=self.source,
                    reference=self.reference, stats=self.stats)


def iter_zip(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if not info.is_dir():
                yield info.filename, z.read(info)


def iter_dir(path: Path):
    for p in sorted(path.rglob("*")):
        if p.is_file():
            yield str(p), p.read_bytes()


def _sort_key(ds, normal):
    return float(np.dot(normal, np.asarray(ds.ImagePositionPatient, float))), int(ds.get("InstanceNumber", 0) or 0)


def build_study(files, cache_root: Path, sid: str, source: str = "upload") -> Study:
    """Parse every file, keep the study with the most series, write its series to the cache, build display volumes."""
    t0 = time.time()
    by_study: dict[str, dict[str, list]] = {}
    n_files, n_bad = 0, 0
    for name, data in files:
        n_files += 1
        if name.lower().endswith(("dicomdir", ".txt", ".json", ".csv", ".xml", ".jpg", ".png", ".pdf")):
            n_bad += 1
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(data), force=True)
            if "PixelData" not in ds or "SeriesInstanceUID" not in ds:
                n_bad += 1
                continue
        except Exception:
            n_bad += 1
            continue
        ds._raw = data                         # original bytes, written to the cache unchanged when not compressed
        by_study.setdefault(str(ds.get("StudyInstanceUID", "unknown")), {}).setdefault(str(ds.SeriesInstanceUID), []).append(ds)
    if not by_study:
        raise ValueError(f"no DICOM images found in {n_files} files")
    study_uid = max(by_study, key=lambda s: (len(by_study[s]), sum(map(len, by_study[s].values()))))
    study = Study(sid=sid, uid=study_uid, root=cache_root / sid, source=source)
    for other in by_study:
        if other != study_uid:
            study.skipped.append(dict(uid=other, reason="another study in the same upload (only one study is read)"))

    def one(uid, slices):
        """Checks, writes and summarizes one series; returns a Series, or a skip record."""
        first = slices[0]
        desc = str(first.get("SeriesDescription", "") or "")
        reason = None
        if str(first.get("Modality", "MR")) != "MR":
            reason = f"modality {first.get('Modality')} (MR only)"
        elif int(first.get("NumberOfFrames", 1) or 1) > 1:
            reason = "multi-frame DICOM (one image per file expected)"
        elif "ImageOrientationPatient" not in first or any("ImagePositionPatient" not in d for d in slices):
            reason = "no patient geometry (ImageOrientation/PositionPatient)"
        elif any("LOCALIZER" in _tokens(d.get("ImageType")) for d in slices[:1]):
            reason = "localizer"
        elif len(slices) < MIN_SLICES:
            reason = f"{len(slices)} slices (a localizer or scout)"
        else:
            iop0 = np.asarray(first.ImageOrientationPatient, float)
            if any(np.abs(np.asarray(d.ImageOrientationPatient, float) - iop0).max() > 1e-3 for d in slices):
                reason = "mixed slice orientations"
            elif len({(int(d.Rows), int(d.Columns)) for d in slices}) > 3:
                reason = "inconsistent image sizes"
        if reason:
            return dict(uid=uid, description=desc, slices=len(slices), reason=reason)

        iop = [float(v) for v in first.ImageOrientationPatient]
        normal = np.cross(iop[:3], iop[3:])
        slices.sort(key=lambda d: _sort_key(d, normal))
        # Write the series as the model modules expect it; compressed transfer syntaxes are decoded (lossless) so the
        # modules' own reader, which expects native pixel data, can read them.
        out = study.root / study_uid / uid
        out.mkdir(parents=True, exist_ok=True)
        decompressed = False
        rows, cols = int(first.Rows), int(first.Columns)
        scale = min(1.0, DISPLAY_MAX_PX / max(rows, cols))
        h, w = max(1, round(rows * scale)), max(1, round(cols * scale))
        small = []
        for i, d in enumerate(slices):
            tsuid = getattr(getattr(d, "file_meta", None), "TransferSyntaxUID", None)
            if tsuid is not None and tsuid.is_compressed:
                d.decompress()                 # lossless: the stored integers are unchanged
                d.save_as(out / f"{i:05d}.dcm", enforce_file_format=True)
                decompressed = True
            else:
                (out / f"{i:05d}.dcm").write_bytes(d._raw)
            del d._raw
            px = d.pixel_array
            if px.shape != (rows, cols):
                px = cv2.resize(px, (cols, rows), interpolation=cv2.INTER_LINEAR)
            # display copy only: shrink in the stored integer type first (cheap), window afterwards
            small.append(cv2.resize(px, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else px)
        slope = np.float32([float(d.get("RescaleSlope", 1) or 1) for d in slices])
        intercept = np.float32([float(d.get("RescaleIntercept", 0) or 0) for d in slices])
        vol = np.stack(small).astype(np.float32) * slope[:, None, None] + intercept[:, None, None]
        lo, hi = np.percentile(vol[:, ::2, ::2], WINDOW)   # the models' window percentiles, on the display copy
        disp = np.clip((vol - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255)
        positions = [[float(v) for v in d.ImagePositionPatient] for d in slices]
        steps = np.diff([float(np.dot(normal, p)) for p in positions])
        fs, fs_reason = fat_suppressed(first)
        return Series(
            uid=uid, study=study_uid, plane=plane_of(iop), fat_suppression=fs, fs_reason=fs_reason, description=desc,
            n_slices=len(slices), rows=rows, cols=cols,
            pixel_spacing=tuple(float(v) for v in first.PixelSpacing),
            slice_spacing=float(np.median(steps)) if len(steps) else float(first.get("SliceThickness", 3.0) or 3.0),
            iop=iop, positions=positions, display=np.round(disp).astype(np.uint8), window=(float(lo), float(hi)),
            tr=float(first.RepetitionTime) if "RepetitionTime" in first else None,
            te=float(first.EchoTime) if "EchoTime" in first else None, decompressed=decompressed)

    ordered = sorted(by_study[study_uid].items(), key=lambda kv: int(kv[1][0].get("SeriesNumber", 0) or 0))
    with ThreadPoolExecutor(min(6, len(ordered))) as pool:   # numpy / OpenCV / file writes release the GIL
        for r in pool.map(lambda kv: one(*kv), ordered):
            if isinstance(r, Series):
                study.series.append(r)
            else:
                study.skipped.append(r)
    if not study.series:
        shutil.rmtree(study.root, ignore_errors=True)
        raise ValueError("no usable MR series: " + "; ".join(f"{s.get('description') or s['uid'][-8:]}: {s['reason']}" for s in study.skipped))
    study.stats = dict(files=n_files, nonImage=n_bad, seconds=round(time.time() - t0, 2),
                       slices=sum(s.n_slices for s in study.series))
    return study
