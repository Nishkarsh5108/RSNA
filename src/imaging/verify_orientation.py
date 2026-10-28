"""Numeric + visual verification of laterality normalisation.

For each plane, average the FS-slot image at fixed relative slice positions over left knees
and over right knees, with and without normalisation. If normalisation is right, left/right
means agree better (higher Pearson r) after it. Sagittal is compared at slice positions 0.2
and 0.8 (lateral vs medial compartment after normalisation), where the medial/lateral
anatomy differs (fibula laterally), so a wrong slice order shows up as low correlation.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_index import SLOT_NAMES  # noqa: E402

R = "D:/Coding/kaggle/RSNA"
OUT = f"{R}/reports/eda/imaging"
os.makedirs(OUT, exist_ok=True)
idx = pd.read_csv(f"{R}/data/imaging_v5/study_index_train.csv")
rng = np.random.default_rng(0)
N = 250
res = []
mean_imgs = {}
for slot, plane in [(0, "Sagittal"), (1, "Coronal"), (2, "Axial")]:
    name = SLOT_NAMES[slot]
    sub = idx[idx[f"{name}_uid"].notna() & (idx[f"{name}_step"] == 1) & (~idx.side_conflict) & (~idx.side_ambiguous)]
    for rel in ([0.2, 0.8] if plane == "Sagittal" else [0.5]):
        acc = {(s, o): [] for s in "LR" for o in (False, True)}
        for side in "LR":
            pick = sub[sub.side == side].sample(N, random_state=1)
            for r in pick.itertuples():
                a = np.load(f"{R}/output/v2/preprocessed/{getattr(r, name + '_uid')}.npy", mmap_mode="r")
                T = a.shape[0]
                raw = np.asarray(a[int(round(rel * (T - 1)))], np.float32)
                if side == "R" and plane == "Sagittal":
                    norm = np.asarray(a[(T - 1) - int(round(rel * (T - 1)))], np.float32)
                elif side == "R":
                    norm = raw[:, ::-1]
                else:
                    norm = raw
                acc[(side, False)].append(raw); acc[(side, True)].append(norm)
        for o in (False, True):
            mL, mR = np.mean(acc[("L", o)], 0), np.mean(acc[("R", o)], 0)
            r_ = float(np.corrcoef(mL.ravel(), mR.ravel())[0, 1])
            res.append({"plane": plane, "rel_slice": rel, "normalised": o, "pearson_r_meanL_vs_meanR": round(r_, 4)})
            mean_imgs[(plane, rel, o)] = (mL, mR)
df = pd.DataFrame(res)
df.to_csv(f"{OUT}/orientation_check.csv", index=False)
print(df.to_string(index=False))
fig, ax = plt.subplots(4, 4, figsize=(12, 12))
for i, key in enumerate([("Coronal", 0.5), ("Axial", 0.5), ("Sagittal", 0.2), ("Sagittal", 0.8)]):
    for j, o in enumerate((False, True)):
        mL, mR = mean_imgs[(key[0], key[1], o)]
        ax[i, 2 * j].imshow(mL, cmap="gray"); ax[i, 2 * j].set_title(f"{key[0]} {key[1]} L {'norm' if o else 'raw'}", fontsize=8)
        ax[i, 2 * j + 1].imshow(mR, cmap="gray"); ax[i, 2 * j + 1].set_title(f"{key[0]} {key[1]} R {'norm' if o else 'raw'}", fontsize=8)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(f"{OUT}/orientation_mean_images.png", dpi=90); plt.close()
