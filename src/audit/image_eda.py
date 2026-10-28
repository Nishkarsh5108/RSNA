"""Pixel-level EDA on a stratified sample of series (bounded memory; 3 slices/series).

Also analyses in-plane orientation conventions from ImageOrientationPatient for
ALL series (header-only), which decides whether images must be re-oriented
before a model sees them.

Outputs: reports/eda/images/*.csv|png  (git-ignored: derived from competition pixels)
"""
import os
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
DICOM = f"{ROOT}/data/Raw_dicom/train_series"
OUT = f"{ROOT}/reports/eda/images"


def series_slices(study, series):
    d = os.path.join(DICOM, study, series)
    items = []
    for f in os.listdir(d):
        ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        n = np.cross(iop[:3], iop[3:])
        items.append((float(np.dot(n, np.array(ds.ImagePositionPatient, dtype=float))), f))
    items.sort()
    return d, [f for _, f in items]


def slice_stats(arr):
    a = arr.astype(np.float32)
    p1, p50, p99 = np.percentile(a, [1, 50, 99])
    lo, hi = a.min(), a.max()
    u8 = np.clip((a - p1) / max(p99 - p1, 1e-6) * 255, 0, 255).astype(np.uint8)
    thr, fgm = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = fgm > 0
    bg = ~fg
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    return {"min": float(lo), "max": float(hi), "p1": float(p1), "p50": float(p50), "p99": float(p99),
            "fg_frac": float(fg.mean()), "border_fg_frac": float(border.mean()),
            "snr_proxy": float(a[fg].mean() / (a[bg].std() + 1e-6)) if bg.any() and fg.any() else np.nan,
            "bg_mean_over_p99": float(a[bg].mean() / max(p99, 1e-6)) if bg.any() else np.nan}


def work(row):
    try:
        d, files = series_slices(row["StudyInstanceUID"], row["SeriesInstanceUID"])
        out = []
        for q in (0.25, 0.5, 0.75):
            f = files[int(round(q * (len(files) - 1)))]
            ds = pydicom.dcmread(os.path.join(d, f))
            arr = ds.pixel_array.astype(np.float32) * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
            if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
                arr = arr.max() - arr
            st = slice_stats(arr)
            st.update({"SeriesInstanceUID": row["SeriesInstanceUID"], "q": q, "rows": arr.shape[0], "cols": arr.shape[1]})
            out.append(st)
            if q == 0.5:
                p1, p99 = np.percentile(arr, [1, 99])
                thumb = cv2.resize(np.clip((arr - p1) / max(p99 - p1, 1e-6), 0, 1), (128, 128), interpolation=cv2.INTER_AREA)
                np.save(os.path.join(OUT, "thumbs", row["SeriesInstanceUID"] + ".npy"), (thumb * 255).astype(np.uint8))
        return out
    except Exception as ex:
        return [{"SeriesInstanceUID": row["SeriesInstanceUID"], "error": f"{type(ex).__name__}: {ex}"}]


def orientation_table():
    sh = pd.read_csv(f"{ROOT}/reports/eda/dicom/series_headers.csv", dtype=str)
    ts = pd.read_csv(f"{ROOT}/data/Raw_dicom/train_series.csv")
    sh = sh.merge(ts, left_on="SeriesInstanceUID_dir", right_on="SeriesInstanceUID")
    rows = []
    for _, r in sh.iterrows():
        iop = np.array(list(map(float, r.ImageOrientationPatient.split("|"))))
        rv, cv = iop[:3], iop[3:]
        ra, ca = int(np.argmax(np.abs(rv))), int(np.argmax(np.abs(cv)))
        axes = "xyz"
        rows.append({"plane": r.Anatomical_Plane,
                     "row_dir": ("+" if rv[ra] > 0 else "-") + axes[ra],
                     "col_dir": ("+" if cv[ca] > 0 else "-") + axes[ca]})
    o = pd.DataFrame(rows)
    tab = o.groupby(["plane", "row_dir", "col_dir"]).size().rename("n_series").reset_index()
    tab.to_csv(os.path.join(OUT, "orientation_conventions.csv"), index=False)
    return tab


def main():
    os.makedirs(os.path.join(OUT, "thumbs"), exist_ok=True)
    print(orientation_table().to_string(index=False), flush=True)
    st = pd.read_csv(f"{ROOT}/reports/eda/metadata/series_table.csv")
    st = st.rename(columns={"SeriesInstanceUID_dir": "SeriesInstanceUID"})
    rng = random.Random(7)
    samp = []
    for (pl, fs), g in st.groupby(["Anatomical_Plane", "Fluid_Sensitive"]):
        samp.append(g.sample(min(60, len(g)), random_state=rng.randint(0, 10**6)))
    samp = pd.concat(samp)
    rows = samp[["StudyInstanceUID", "SeriesInstanceUID"]].to_dict("records")
    res = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for out in ex.map(work, rows, chunksize=4):
            res.extend(out)
    df = pd.DataFrame(res).merge(samp[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Manufacturer",
                                       "MagneticFieldStrength", "n_files", "PixelSpacing_row", "Rows"]], on="SeriesInstanceUID")
    df.to_csv(os.path.join(OUT, "pixel_stats_sample.csv"), index=False)
    print("errors:", int(df.get("error", pd.Series(dtype=str)).notna().sum()))
    mid = df[df.q == 0.5]
    summ = mid.groupby(["Anatomical_Plane", "Fluid_Sensitive"])[["p99", "fg_frac", "border_fg_frac", "snr_proxy"]].median()
    summ.to_csv(os.path.join(OUT, "pixel_summary_by_plane_seq.csv"))
    print(summ.round(3).to_string())
    man = mid.groupby("Manufacturer")[["p50", "p99", "max"]].median()
    man.to_csv(os.path.join(OUT, "intensity_by_manufacturer.csv"))
    print(man.round(1).to_string())
    # figures
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for m, g in mid.groupby("Manufacturer"):
        ax[0].hist(np.log10(g.p99.clip(lower=1)), bins=30, alpha=0.5, label=m)
    ax[0].set_title("log10 p99 intensity (middle slice) by vendor"); ax[0].legend(fontsize=7)
    ax[1].hist(mid.fg_frac, bins=30, color="#4C72B0"); ax[1].set_title("Foreground fraction (Otsu)")
    ax[2].hist(mid.border_fg_frac, bins=30, color="#C44E52"); ax[2].set_title("Foreground touching image border")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pixel_stats.png"), dpi=110); plt.close()
    # montage: typical + unusual
    pick = []
    for pl in ["Sagittal", "Coronal", "Axial"]:
        g = mid[mid.Anatomical_Plane == pl]
        pick += [(f"{pl} typical", u) for u in g.sample(4, random_state=1).SeriesInstanceUID]
        pick += [(f"{pl} low SNR", g.sort_values("snr_proxy").SeriesInstanceUID.iloc[0])]
        pick += [(f"{pl} edge-touching", g.sort_values("border_fg_frac").SeriesInstanceUID.iloc[-1])]
    fig, ax = plt.subplots(3, 6, figsize=(15, 8))
    for a, (title, u) in zip(ax.ravel(), pick):
        a.imshow(np.load(os.path.join(OUT, "thumbs", u + ".npy")), cmap="gray"); a.set_title(title, fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "montage_typical_unusual.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()
