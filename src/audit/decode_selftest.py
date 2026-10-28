"""Paste-in self-test for the Kaggle SUBMISSION notebook (internet off).

Why: the hidden test set may contain JPEG-Lossless and JPEG-2000 DICOMs (official
data page). The v2/v3 preprocessing catches decode errors and zero-fills the plane,
so missing decoders silently degrade predictions and hidden-run logs are not
visible. This check fails LOUDLY instead (a scoring error is visible; a silent
0.5/zero-filled prediction is not).

Attach wheels for: pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg (or python-gdcm)
as a Kaggle Dataset and `pip install --no-index --find-links <dir> ...` first.
Verified locally (2026-10-17): a competition slice transcoded with gdcmconv to
JPEG-Lossless SV1 and JPEG-2000 decodes bit-identically once these are installed;
without them JPEG-Lossless has no available decoder.
"""
import os

import pydicom
from pydicom.pixels import get_decoder
from pydicom.uid import JPEG2000, JPEG2000Lossless, JPEGLossless, JPEGLosslessSV1

REQUIRED = [JPEGLosslessSV1, JPEGLossless, JPEG2000Lossless, JPEG2000]


def assert_decoders():
    missing = [u.name for u in REQUIRED if not get_decoder(u).is_available]
    if missing:
        raise RuntimeError(f"No pixel decoder for: {missing}. Install pylibjpeg(-libjpeg,-openjpeg) or gdcm wheels.")


def decode_failure_report(test_root, max_files_per_series=1):
    """Try to decode one slice per series; return (n_checked, failures list)."""
    fails, n = [], 0
    for study in os.listdir(test_root):
        for series in os.listdir(os.path.join(test_root, study)):
            d = os.path.join(test_root, study, series)
            for f in sorted(os.listdir(d))[:max_files_per_series]:
                n += 1
                try:
                    _ = pydicom.dcmread(os.path.join(d, f)).pixel_array
                except Exception as ex:
                    fails.append((study, series, type(ex).__name__))
    return n, fails


if __name__ == "__main__":
    assert_decoders()
    root = "/kaggle/input/rsna-knee-abnormality-detection/test_series" if os.path.isdir("/kaggle") else \
        "D:/Coding/kaggle/RSNA/data/Raw_dicom/test_series"
    n, fails = decode_failure_report(root)
    print(f"decoded {n - len(fails)}/{n} series probes")
    if fails:
        raise RuntimeError(f"{len(fails)} series failed to decode, e.g. {fails[:3]}")
