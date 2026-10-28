"""Metadata EDA from the DICOM header inventory (no pixel decoding).

Inputs : reports/eda/dicom/series_headers.csv (all series, 1 header each)
         reports/eda/dicom/slice_headers_sample.csv (all slices of 2,000 random series)
         data/Raw_dicom/train.csv, train_series.csv, reports/eda/report_profile.csv
Outputs: reports/eda/metadata/*.csv and *.png, plus a study-level completeness
         table (reports/eda/study_inventory.csv) used by later steps.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = "D:/Coding/kaggle/RSNA"
OUT = f"{ROOT}/reports/eda/metadata"
os.makedirs(OUT, exist_ok=True)
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

sh = pd.read_csv(f"{ROOT}/reports/eda/dicom/series_headers.csv", dtype=str)
for c in ["n_files", "n_dcm", "bytes", "Rows", "Columns", "SliceThickness", "SpacingBetweenSlices",
          "MagneticFieldStrength", "RepetitionTime", "EchoTime", "InversionTime", "EchoTrainLength",
          "FlipAngle", "PixelBandwidth", "BitsStored", "RescaleSlope", "RescaleIntercept", "iop_plane_conf"]:
    sh[c] = pd.to_numeric(sh[c], errors="coerce")
sh["PixelSpacing_row"] = pd.to_numeric(sh["PixelSpacing"].str.split("|").str[0], errors="coerce")
sh["PixelSpacing_col"] = pd.to_numeric(sh["PixelSpacing"].str.split("|").str[1], errors="coerce")
ts = pd.read_csv(f"{ROOT}/data/Raw_dicom/train_series.csv", dtype={"Fluid_Sensitive": int, "Fat_Suppression": int})
train = pd.read_csv(f"{ROOT}/data/Raw_dicom/train.csv")
prof = pd.read_csv(f"{ROOT}/reports/eda/report_profile.csv")
prof["lang_final"] = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})

s = sh.merge(ts, left_on="SeriesInstanceUID_dir", right_on="SeriesInstanceUID", how="outer", indicator=True)
checks = {
    "series_in_csv_not_on_disk": int((s._merge == "right_only").sum()),
    "series_on_disk_not_in_csv": int((s._merge == "left_only").sum()),
    "series_status_not_ok": int((s.status != "ok").sum()),
    "empty_or_non_dcm_files": int((s.n_files != s.n_dcm).sum()),
    "header_study_uid_mismatch": int((s.StudyInstanceUID_dir != s["StudyInstanceUID_x"]).sum()) if "StudyInstanceUID_x" in s else None,
    "header_series_uid_mismatch": int((s.SeriesInstanceUID_dir != s["SeriesInstanceUID_x"]).sum()) if "SeriesInstanceUID_x" in s else None,
    "study_dir_vs_csv_study_mismatch": int((s.StudyInstanceUID_dir != s["StudyInstanceUID_y"]).sum()),
    "plane_csv_vs_iop_mismatch": int((s.Anatomical_Plane != s.iop_plane).sum()),
    "oblique_series_iop_conf_lt_0.9": int((s.iop_plane_conf < 0.9).sum()),
    "total_series": int(len(s)), "total_files": int(s.n_files.sum()),
    "total_bytes_GB": round(float(s.bytes.sum()) / 1e9, 1),
}
s = s[s._merge == "both"].copy()
s["StudyInstanceUID"] = s["StudyInstanceUID_dir"]

# ---------------------------------------------------------------- series-level distributions
def vc(col, name=None, top=40):
    v = s[col].fillna("<missing>").value_counts().head(top)
    v.to_csv(f"{OUT}/vc_{name or col}.csv", header=["n_series"])
    return v

for col in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "TransferSyntax",
            "PhotometricInterpretation", "Laterality", "PatientPosition", "MRAcquisitionType",
            "ScanningSequence", "ScanOptions", "BodyPartExamined", "PatientSex", "BitsStored",
            "ContrastBolusAgent", "SeriesDescription", "Anatomical_Plane", "iop_plane"]:
    vc(col)
seq = s.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size().rename("n_series")
seq.to_csv(f"{OUT}/sequence_flags_by_plane.csv")
num = s.groupby("Anatomical_Plane")[["n_files", "SliceThickness", "SpacingBetweenSlices", "PixelSpacing_row",
                                     "Rows", "Columns", "MagneticFieldStrength", "RepetitionTime", "EchoTime"]].describe(
    percentiles=[.01, .05, .5, .95, .99]).T
num.to_csv(f"{OUT}/numeric_by_plane.csv")
s["fov_mm"] = s.PixelSpacing_row * s.Rows
s["slab_mm"] = s.n_files * s.SpacingBetweenSlices.fillna(s.SliceThickness)
s[["SeriesInstanceUID_dir", "StudyInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression",
   "n_files", "Rows", "Columns", "PixelSpacing_row", "SliceThickness", "SpacingBetweenSlices", "fov_mm",
   "slab_mm", "Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "TransferSyntax",
   "PhotometricInterpretation", "Laterality", "PatientPosition", "iop_plane", "iop_plane_conf",
   "SeriesDescription"]].to_csv(f"{OUT}/series_table.csv", index=False)

# ---------------------------------------------------------------- study-level inventory
g = s.groupby("StudyInstanceUID")
study = pd.DataFrame({
    "n_series": g.size(),
    "n_slices": g.n_files.sum(),
    "bytes": g.bytes.sum(),
    "n_sag": g.apply(lambda x: (x.Anatomical_Plane == "Sagittal").sum(), include_groups=False),
    "n_cor": g.apply(lambda x: (x.Anatomical_Plane == "Coronal").sum(), include_groups=False),
    "n_ax": g.apply(lambda x: (x.Anatomical_Plane == "Axial").sum(), include_groups=False),
    "n_fluid": g.Fluid_Sensitive.sum(),
    "has_fluid_sag": g.apply(lambda x: ((x.Anatomical_Plane == "Sagittal") & (x.Fluid_Sensitive == 1)).any(), include_groups=False),
    "has_fluid_cor": g.apply(lambda x: ((x.Anatomical_Plane == "Coronal") & (x.Fluid_Sensitive == 1)).any(), include_groups=False),
    "has_fluid_ax": g.apply(lambda x: ((x.Anatomical_Plane == "Axial") & (x.Fluid_Sensitive == 1)).any(), include_groups=False),
    "manufacturer": g.Manufacturer.agg(lambda x: x.mode().iloc[0] if x.notna().any() else None),
    "model": g.ManufacturerModelName.agg(lambda x: x.mode().iloc[0] if x.notna().any() else None),
    "field_T": g.MagneticFieldStrength.median(),
    "n_manufacturers": g.Manufacturer.nunique(),
    "patient_id": g.PatientID.agg(lambda x: x.mode().iloc[0] if x.notna().any() else None),
    "n_patient_ids": g.PatientID.nunique(),
    "sex": g.PatientSex.agg(lambda x: x.mode().iloc[0] if x.notna().any() else None),
    "laterality_tag": g.Laterality.agg(lambda x: "|".join(sorted(set(v for v in x.dropna() if v)))),
    "transfer_syntaxes": g.TransferSyntax.agg(lambda x: "|".join(sorted(set(x.dropna())))),
    "frame_of_ref_n": g.FrameOfReferenceUID.nunique(),
}).reset_index()
study = study.merge(train[["StudyInstanceUID"] + TARGETS], on="StudyInstanceUID", how="right")
study["is_gold"] = study[TARGETS].notna().all(axis=1)
study = study.drop(columns=TARGETS).merge(prof[["StudyInstanceUID", "lang_final", "n_chars", "dup_group_size", "report_hash"]],
                                          on="StudyInstanceUID", how="left")
study["has_report"] = study.n_chars.fillna(0) > 0
study["has_all_3_planes"] = (study.n_sag > 0) & (study.n_cor > 0) & (study.n_ax > 0)
study.to_csv(f"{ROOT}/reports/eda/study_inventory.csv", index=False)

pid_counts = study.patient_id.value_counts()
patients = {
    "n_studies": int(len(study)),
    "n_unique_patient_ids": int(study.patient_id.nunique()),
    "studies_sharing_patient_id": int((study.patient_id.map(pid_counts) > 1).sum()),
    "max_studies_per_patient_id": int(pid_counts.max()),
    "studies_with_multiple_patient_ids_across_series": int((study.n_patient_ids > 1).sum()),
    "studies_missing_patient_id": int(study.patient_id.isna().sum()),
    "gold_studies_sharing_patient_with_nongold": int(study[study.is_gold].patient_id.isin(
        study[~study.is_gold].patient_id).sum()),
}
multi = study[study.patient_id.map(pid_counts) > 1].sort_values("patient_id")
multi[["patient_id", "StudyInstanceUID", "is_gold", "lang_final", "manufacturer", "field_T", "n_series",
       "report_hash", "dup_group_size"]].to_csv(f"{OUT}/patients_with_multiple_studies.csv", index=False)

# ---------------------------------------------------------------- slice-level geometry (sample)
sl = pd.read_csv(f"{ROOT}/reports/eda/dicom/slice_headers_sample.csv", dtype=str)
geo = []
for sid, x in sl.groupby("SeriesInstanceUID_dir"):
    try:
        ipp = np.array([list(map(float, v.split("|"))) for v in x.ipp])
        iop = np.array(list(map(float, x.iop.iloc[0].split("|"))))
        n_iop = x.iop.nunique()
        normal = np.cross(iop[:3], iop[3:])
        proj = ipp @ normal
        order = np.argsort(proj)
        d = np.diff(proj[order])
        inst = pd.to_numeric(x.InstanceNumber, errors="coerce").values
        inst_sorted_by_pos = inst[order]
        mono = bool(np.all(np.diff(inst_sorted_by_pos) > 0) or np.all(np.diff(inst_sorted_by_pos) < 0))
        geo.append({
            "SeriesInstanceUID": sid, "n_slices": len(x), "n_distinct_iop": n_iop,
            "n_duplicate_positions": int((np.abs(d) < 1e-3).sum()),
            "spacing_median": float(np.median(d)) if len(d) else np.nan,
            "spacing_cv": float(np.std(d) / np.median(d)) if len(d) and np.median(d) > 0 else np.nan,
            "spacing_max_over_median": float(d.max() / np.median(d)) if len(d) and np.median(d) > 0 else np.nan,
            "instance_monotonic_with_position": mono,
            "instance_gaps": int(np.nanmax(inst) - np.nanmin(inst) + 1 - len(np.unique(inst))) if np.isfinite(inst).all() else -1,
            "rows_vary": x.Rows.nunique() > 1, "pixel_spacing_vary": x.PixelSpacing.nunique() > 1,
            "transfer_syntax_vary": x.TransferSyntax.nunique() > 1,
        })
    except Exception as ex:
        geo.append({"SeriesInstanceUID": sid, "error": f"{type(ex).__name__}: {ex}"})
geo = pd.DataFrame(geo).merge(ts[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"]], on="SeriesInstanceUID", how="left")
geo.to_csv(f"{OUT}/slice_geometry_sample.csv", index=False)
geo_summary = {
    "series_sampled": int(len(geo)),
    "series_multiple_orientations": int((geo.n_distinct_iop > 1).sum()),
    "series_with_duplicate_positions": int((geo.n_duplicate_positions > 0).sum()),
    "series_irregular_spacing_cv_gt_0.1": int((geo.spacing_cv > 0.1).sum()),
    "series_gap_gt_1.5x_median": int((geo.spacing_max_over_median > 1.5).sum()),
    "series_instance_not_monotonic_with_position": int((~geo.instance_monotonic_with_position.astype(bool)).sum()),
    "series_rows_vary": int(geo.rows_vary.fillna(False).astype(bool).sum()),
    "series_pixel_spacing_vary": int(geo.pixel_spacing_vary.fillna(False).astype(bool).sum()),
}

# ---------------------------------------------------------------- plots
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for i, pl in enumerate(["Sagittal", "Coronal", "Axial"]):
    v = s[s.Anatomical_Plane == pl]
    ax[0, i].hist(v.n_files.clip(upper=80), bins=40, color="#4C72B0")
    ax[0, i].set_title(f"{pl}: slices per series (n={len(v)})")
    ax[1, i].hist(v.PixelSpacing_row.clip(upper=1.0), bins=40, color="#55A868")
    ax[1, i].set_title(f"{pl}: in-plane pixel spacing (mm)")
plt.tight_layout(); plt.savefig(f"{OUT}/series_slices_and_spacing.png", dpi=110); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
s.groupby("Manufacturer").size().sort_values().plot.barh(ax=ax[0], color="#8172B2"); ax[0].set_title("Series by manufacturer")
s.MagneticFieldStrength.round(1).value_counts().sort_index().plot.bar(ax=ax[1], color="#C44E52"); ax[1].set_title("Series by field strength (T)")
s.SliceThickness.clip(upper=8).plot.hist(bins=32, ax=ax[2], color="#CCB974"); ax[2].set_title("Slice thickness (mm)")
plt.tight_layout(); plt.savefig(f"{OUT}/scanner_and_thickness.png", dpi=110); plt.close()

pd.Series({**checks, **patients, **geo_summary}).to_csv(f"{OUT}/metadata_checks.csv", header=["value"])
print(pd.Series({**checks, **patients, **geo_summary}).to_string())
