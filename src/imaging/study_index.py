"""Build the per-study index used by the image models (train) and the same logic for test.

For every study it decides:
  * which series fills each slot  (plane x {fluid-sensitive, non-fluid-sensitive}),
  * the knee side used for orientation normalisation (+ provenance and conflict flags),
  * the 2.5D neighbour offset per series from its physical slice spacing,
  * the CV fold (train studies only).

Series *metadata* (plane, Fluid_Sensitive flag, slice count, spacing, geometry) is used only
to choose WHICH images are read. It is never fed to the classifier as a feature.

Laterality policy (per study):
  side = majority DICOM Laterality tag if any series carries one, else
         majority geometric side over series whose volume centre is >= 20 mm from x=0
         (sign of patient-x: + -> left knee), else majority over all series.
  Flags: 'conflict' if a confident geometric side disagrees with the tag side;
         'ambiguous' if no series is >= 20 mm off the midline.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PLANES = ["Sagittal", "Coronal", "Axial"]
SLOTS = [(p, fs) for fs in (1, 0) for p in PLANES]          # 0..2 = FS sag/cor/ax, 3..5 = non-FS
SLOT_NAMES = [f"{p[:3].lower()}_{'fs' if fs else 'nfs'}" for p, fs in SLOTS]
TARGET_STEP_MM = 3.5      # physical distance between the 3 slices of a 2.5D window
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def choose_slots(ser):
    """ser: DataFrame of one study's series with columns SeriesInstanceUID, Anatomical_Plane,
    Fluid_Sensitive, n_slices. Returns dict slot_idx -> SeriesInstanceUID (or None) and the
    candidate lists (for optional random choice during training)."""
    out, cands = {}, {}
    for si, (plane, fs) in enumerate(SLOTS):
        c = ser[(ser.Anatomical_Plane == plane) & (ser.Fluid_Sensitive == fs) & (ser.n_slices >= 3)]
        c = c.sort_values(["n_slices", "SeriesInstanceUID"], ascending=[False, True])
        cands[si] = list(c.SeriesInstanceUID)
        out[si] = cands[si][0] if cands[si] else None
    return out, cands


def study_side(lat_rows):
    """lat_rows: rows of laterality_by_series for one study."""
    tags = lat_rows.tag_side.dropna()
    tags = tags[tags.isin(["L", "R"])]
    # Geometry only from coronal/axial series: their in-plane centre-x is exact for every slice,
    # whereas sagittal slices are spread along x (v2 index fix; see image_pipeline_audit.md).
    ca = lat_rows[lat_rows.plane.isin(["Coronal", "Axial"])]
    ca = ca if len(ca) else lat_rows
    conf = ca[ca.center_x_mm.abs() >= 20]
    geo_src = conf if len(conf) else ca
    geo = geo_src.geo_side.value_counts().idxmax()
    ambiguous = len(conf) == 0
    if len(tags):
        tag = tags.value_counts().idxmax()
        conflict = (not ambiguous) and (conf.geo_side.value_counts().idxmax() != tag)
        return tag, "tag", bool(conflict), ambiguous
    return geo, "geometry", False, ambiguous


def build(series_df, headers_df, lat_df):
    s = series_df.merge(headers_df, on="SeriesInstanceUID", how="left")
    rows = []
    for uid, ser in s.groupby("StudyInstanceUID"):
        slots, cands = choose_slots(ser)
        side, src, conflict, amb = study_side(lat_df[lat_df.StudyInstanceUID == uid])
        r = {"StudyInstanceUID": uid, "side": side, "side_source": src, "side_conflict": conflict,
             "side_ambiguous": amb}
        for si, name in enumerate(SLOT_NAMES):
            sid = slots[si]
            r[f"{name}_uid"] = sid
            r[f"{name}_n_cands"] = len(cands[si])
            r[f"{name}_cands"] = "|".join(cands[si])
            if sid is not None:
                h = ser[ser.SeriesInstanceUID == sid].iloc[0]
                sp = h.spacing_mm if np.isfinite(h.spacing_mm) and h.spacing_mm > 0 else 3.5
                r[f"{name}_n"] = int(h.n_slices)
                r[f"{name}_spacing"] = float(sp)
                r[f"{name}_step"] = int(max(1, round(TARGET_STEP_MM / sp)))
            else:
                r[f"{name}_n"], r[f"{name}_spacing"], r[f"{name}_step"] = 0, np.nan, 0
        rows.append(r)
    return pd.DataFrame(rows)


def add_folds(idx, labels, lang, dup_groups, n_folds=5, seed=42):
    """Stratify non-gold studies by language x number of report-positive findings.
    Gold studies get fold -1 (never in a training/validation fold). Duplicate studies share a fold."""
    lab = labels.set_index("StudyInstanceUID")
    gold = set(lab.index[lab.label_source == "gold"])
    ng = idx[~idx.StudyInstanceUID.isin(gold)].copy()
    burden = (lab.loc[ng.StudyInstanceUID, TARGETS] >= 0.5).sum(axis=1).clip(upper=6).astype(str).values
    key = (lang.reindex(ng.StudyInstanceUID).fillna("NA").values + "_" + burden)
    vc = pd.Series(key).value_counts()
    key = np.where(pd.Series(key).map(vc).values < n_folds, "rare", key)
    folds = np.full(len(ng), -1)
    for f, (_, va) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(ng, key)):
        folds[va] = f
    ng["fold"] = folds
    fmap = dict(zip(ng.StudyInstanceUID, ng.fold))
    for grp in dup_groups:                       # force duplicates into the first member's fold
        f0 = fmap.get(grp[0])
        for u in grp[1:]:
            if u in fmap and f0 is not None:
                fmap[u] = f0
    idx["fold"] = idx.StudyInstanceUID.map(fmap).fillna(-1).astype(int)
    idx["is_gold"] = idx.StudyInstanceUID.isin(gold)
    return idx


def headers_table(series_headers_csv):
    h = pd.read_csv(series_headers_csv, dtype=str)
    out = pd.DataFrame({"SeriesInstanceUID": h.SeriesInstanceUID_dir,
                        "n_slices": pd.to_numeric(h.n_files),
                        "spacing_mm": pd.to_numeric(h.SpacingBetweenSlices, errors="coerce").fillna(
                            pd.to_numeric(h.SliceThickness, errors="coerce"))})
    return out


if __name__ == "__main__":
    R = "D:/Coding/kaggle/RSNA"
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{R}/data/imaging_v5/study_index_train_v2.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    series = pd.read_csv(f"{R}/data/Raw_dicom/train_series.csv")
    hdr = headers_table(f"{R}/reports/eda/dicom/series_headers.csv")
    lat = pd.read_csv(f"{R}/reports/eda/metadata/laterality_by_series.csv")
    idx = build(series, hdr, lat)
    labels = pd.read_csv(f"{R}/data/labels_v3/training_labels_v3.csv")
    prof = pd.read_csv(f"{R}/reports/eda/report_profile.csv").set_index("StudyInstanceUID")
    lang = prof["lang"].where(prof["lang"] != "Latin", prof["lang2"]).replace({"Bosnian": "Croatian"})
    dups = pd.read_csv(f"{R}/reports/eda/metadata/duplicate_study_pairs_pixel_verified.csv")
    dup_groups = [[r.study_a, r.study_b] for r in dups.itertuples() if r.shared_identical_slices > 0]
    idx = add_folds(idx, labels, lang, dup_groups)
    idx["lang"] = idx.StudyInstanceUID.map(lang)
    idx.to_csv(a.out, index=False)
    print("studies", len(idx), "| fold sizes", idx.fold.value_counts().sort_index().to_dict())
    print("side:", idx.side.value_counts().to_dict(), "| source:", idx.side_source.value_counts().to_dict(),
          "| conflicts:", int(idx.side_conflict.sum()), "| ambiguous:", int(idx.side_ambiguous.sum()))
    for n in SLOT_NAMES:
        print(f"{n}: present {int(idx[f'{n}_uid'].notna().sum())}  multi-candidate {int((idx[f'{n}_n_cands'] > 1).sum())}  "
              f"step>1 {int((idx[f'{n}_step'] > 1).sum())}")
    for g in dup_groups:
        print("duplicate pair folds:", idx.set_index("StudyInstanceUID").loc[g, "fold"].tolist())
