"""Own idea N4a: sharpen the box-pooling model's in-box attention at inference, with no retraining.

The model weights every cell inside a finding's ROI box by exp(a); here exp(s * a) with s > 1 concentrates the pooling
on the cells the model already rates highest (s = 1 is the trained behaviour). Run on the held-out fold + gold-58 with
one or more trained models; the encoder runs once per batch and every s reuses its feature maps.

  python sweep_sharpen.py --modules <dir with the patched classifier.py> --weights a1=<fold0_ema.pth> --fold 0
                          --values 1,1.25,1.5,2 --out <dir>          (writes series_sharp.csv)
  python sweep_sharpen.py --score <dir>                              (tables + paired bootstrap vs s = 1)
"""
import argparse
import os
import sys
import time


def run(a):
    os.environ.setdefault("COMP_DIR", "D:/Coding/kaggle/RSNA/data/Raw_dicom")
    os.environ.setdefault("DATA_DIR", "D:/rsna_work/ext_labels/tascj")
    os.environ["FOLD"] = str(a.fold)
    sys.path.insert(0, a.modules)
    import pandas as pd
    import torch

    import classifier as C

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    _, valid, label_of, _, gold, _ = C.tables()
    if a.gold_only:
        valid = valid[valid.StudyInstanceUID.isin(gold.index)]
    valid = valid.reset_index(drop=True)
    if a.limit:
        valid = valid.iloc[: a.limit]
    ds = C.SeriesDataset(valid, label_of, C.knee_sides(), pd.read_parquet(C.ROI_BOXES))
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, collate_fn=C.collate,
                                         prefetch_factor=4 if a.workers else None)
    models = {}
    for spec in a.weights.split(","):
        name, path = spec.split("=", 1)
        m = C.BoxPoolClassifier(pretrained=False)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models[name] = m.to(dev).eval().to(memory_format=torch.channels_last)
    values = [float(v) for v in a.values.split(",")]
    print(f"{len(valid)} series of {valid.StudyInstanceUID.nunique()} studies | models {list(models)} | s {values}", flush=True)

    rows, t0, done = [], time.time(), 0
    with torch.no_grad():
        for x in loader:
            image = C.frame_volumes(x, dev)
            ss, depth, bx = (x[k].to(dev, non_blocking=True) for k in ("slice_series", "series_depth", "boxes"))
            theta = C.crop_theta(bx)
            inp = C.sample_slices(image, ss, theta, C.CROP_PX)
            with torch.autocast(dev.type, torch.float16, enabled=dev.type == "cuda"):
                for name, m in models.items():
                    encode, cache = m.encoder.forward_features, {}

                    def cached(t, encode=encode, cache=cache):   # the encoder runs once per batch and model
                        if "f" not in cache:
                            cache["f"] = encode(t)
                        return cache["f"]

                    m.encoder.forward_features = cached
                    try:
                        for s in values:
                            C.ATT_SHARP = s
                            p = torch.sigmoid(m(inp, ss, depth, theta, C.pooling_boxes(bx)).float()).cpu().numpy()
                            rows += [(name, s, uid, study, *q) for uid, study, q in zip(x["series"], x["study"], p)]
                    finally:
                        m.encoder.forward_features = encode
                        C.ATT_SHARP = 1.0
            done += len(depth)
            if done % (50 * a.batch) < a.batch:
                print(f"{done}/{len(valid)} series, {done / (time.time() - t0):.2f} series/s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    df = pd.DataFrame(rows, columns=["model", "sharp", "SeriesInstanceUID", "StudyInstanceUID", *C.LABELS])
    df = df.merge(valid[["SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]], on="SeriesInstanceUID")
    df.to_csv(f"{a.out}/series_sharp.csv", index=False)
    print("wrote", f"{a.out}/series_sharp.csv", len(df), "rows in", round(time.time() - t0), "s", flush=True)


def score(a):
    import numpy as np
    import pandas as pd

    from score_bp import E, L, R, auc_cols, fmt, macro, paired

    df = pd.read_csv(f"{a.score}/series_sharp.csv")
    gold = pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv").set_index("StudyInstanceUID")[L]
    ref = {"7th": pd.read_csv(f"{E}/songqizhou/8_final_label.csv").set_index("StudyInstanceUID")[L],
           "1st": pd.read_csv(f"{E}/tascj/labels_astra_trained_guide.csv").set_index("StudyInstanceUID")[L]}
    for name, d in df.groupby("model"):
        st = {s: g.groupby("StudyInstanceUID")[L].mean() for s, g in d.groupby("sharp")}
        base = st[1.0]
        gi = [u for u in gold.index if u in base.index]
        vi = sorted(set(base.index) - set(gold.index))
        print(f"\n=== {name}: {len(gi)} gold studies, {len(vi)} held-out studies ===")
        print("  s     gold-58   fold vs 7th  fold vs 1st")
        for s, p in sorted(st.items()):
            line = f"{s:5.2f}  {macro(gold.loc[gi].values, p.loc[gi].values):.4f}"
            for k in ("7th", "1st"):
                line += f"    {macro((ref[k].loc[vi].values >= 0.5).astype(int), p.loc[vi].values):.4f}" if vi else "    -"
            print(line)
        t = pd.DataFrame({f"s={s:g}": auc_cols((ref["1st"].loc[vi].values >= 0.5).astype(int), p.loc[vi].values) if vi
                          else auc_cols(gold.loc[gi].values, p.loc[gi].values) for s, p in sorted(st.items())}, index=L)
        print(("per finding, fold vs 1st labels" if vi else "per finding, gold-58") + "\n" + t.round(4).to_string())
        for s, p in sorted(st.items()):
            if s == 1.0:
                continue
            print(f"paired s={s:g} minus s=1 | gold-58:", fmt(paired(gold.loc[gi].values, p.loc[gi].values, base.loc[gi].values, 4000)))
            for k in ("7th", "1st"):
                if vi:
                    print(f"                      | fold vs {k}:", fmt(paired((ref[k].loc[vi].values >= 0.5).astype(int), p.loc[vi].values, base.loc[vi].values, 1000)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="")
    ap.add_argument("--weights", default="", help="comma list name=path")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--values", default="1,1.25,1.5,2")
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gold-only", action="store_true")
    ap.add_argument("--score", default="")
    a = ap.parse_args()
    score(a) if a.score else run(a)
