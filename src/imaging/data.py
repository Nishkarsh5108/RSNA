"""Study-level dataset for 2.5D multi-slot MRI models.

Reads per-series uint8 volumes (T, H, W) from a preprocessing cache (default: the verified
v2 cache: per-series 1-99 percentile window -> uint8 -> 224x224 bilinear, slices sorted
along the slice normal). Orientation normalisation (applied here, at load time):

  * coronal / axial, right knee  -> mirror columns  (lateral is then always image-right)
  * sagittal, right knee         -> reverse slice order (slice 0 is then always lateral)

The sagittal rule follows from the data's uniform orientation (rows +y, cols -z, slice
normal -x for every sagittal series): ascending normal projection runs from patient-left
to patient-right, i.e. lateral->medial for a left knee and medial->lateral for a right knee.

Each slot yields K windows of 3 slices [c-step, c, c+step] (step from physical spacing, about
3.5 mm). Missing slots are zeros with mask 0; the model must ignore them.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from study_index import SLOTS, SLOT_NAMES, TARGETS


def window_centres(T, K, lo, hi, train, rng):
    """K window centres in the [lo, hi] span of a T-slice series. K = 0 (evaluation only): every
    slice in the span is a centre (variable count; the dataset pads and returns a window mask)."""
    a, b = lo * (T - 1), hi * (T - 1)
    if K == 0:
        return np.arange(int(np.ceil(a - 1e-6)), int(np.floor(b + 1e-6)) + 1).clip(0, T - 1)
    if K == 1:
        c = np.array([(a + b) / 2])
    elif train:
        edges = np.linspace(a, b, K + 1)
        c = rng.uniform(edges[:-1], edges[1:])
    else:
        c = np.linspace(a, b, K)
    return np.clip(np.round(c).astype(int), 0, T - 1)


class KneeStudyDataset(Dataset):
    def __init__(self, index_df, cache_dir, slots=(0, 1, 2), K=8, train=False, labels=None,
                 weights=None, lo=0.04, hi=0.96, random_series=False, spacing_map=None, seed=0,
                 orient=True, strict=True, fallback=True, sag_pos_fix=False, crop=0, k_max=64):
        self.idx = index_df.reset_index(drop=True)
        # crop > 0 (hr cache): square crop of every window, random per series in training (jitter), centred
        # otherwise. K = 0 (evaluation only): every slice in the span is a window centre; windows are padded
        # to the longest slot (at most k_max, evenly thinned beyond) and a per-window mask "wmask" is returned.
        self.crop, self.k_max = crop, k_max
        self.cache_dir = cache_dir
        self.slots = list(slots)
        self.K, self.train, self.lo, self.hi = K, train, lo, hi
        self.labels = labels            # (N, 12) float array aligned with index_df, or None
        self.weights = weights          # (N,) float array or None
        self.random_series = random_series
        self.spacing_map = spacing_map or {}
        self.seed = seed
        self.orient = orient
        self.strict = strict            # missing cache file for a chosen series -> raise, never zero-fill
        self.fallback = fallback        # empty FS slot (0-2) -> use the same plane's non-FS series (slot+3)
        # Slice-position feature for RIGHT-knee sagittal series. The mirrored window indices already put
        # window k at anatomical position p = c/(T-1) (0 = lateral) for both sides, so the correct feature
        # is p. The original code passed 1 - p (found 2026-10-17 by attention_map.py); False reproduces it
        # for checkpoints trained that way (args without "sag_pos_fix").
        self.sag_pos_fix = sag_pos_fix

    def __len__(self):
        return len(self.idx)

    def _load(self, uid):
        p = os.path.join(self.cache_dir, f"{uid}.npy")
        if not os.path.exists(p):
            if self.strict:
                raise FileNotFoundError(f"cache missing for series {uid}")
            return None
        return np.load(p, mmap_mode="r")

    def __getitem__(self, i):
        row = self.idx.iloc[i]
        rng = np.random.default_rng((self.seed * 1_000_003 + i * 7919 + int(torch.randint(0, 1 << 30, (1,)))) if self.train else i)
        S = len(self.slots)
        crop, k_max = getattr(self, "crop", 0), getattr(self, "k_max", 64)   # objects pickled before these existed
        loaded = []                          # (j, plane, arr, step, centres, crop offsets) per slot with a series
        for j, si in enumerate(self.slots):
            name = SLOT_NAMES[si]
            uid = row[f"{name}_uid"]
            if not isinstance(uid, str) and self.fallback and si < 3 and (si + 3) not in self.slots:
                name = SLOT_NAMES[si + 3]          # same plane, non-fluid-sensitive series
                uid = row[f"{name}_uid"]
            if not isinstance(uid, str):
                continue
            if self.train and self.random_series:
                cands = str(row[f"{name}_cands"]).split("|")
                uid = cands[rng.integers(len(cands))]
            step = int(row[f"{name}_step"]) if uid == row[f"{name}_uid"] else \
                max(1, int(round(3.5 / self.spacing_map.get(uid, 3.5))))
            arr = self._load(uid)
            if arr is None:
                continue
            T, H, W = arr.shape
            cs = window_centres(T, self.K, self.lo, self.hi, self.train, rng)
            if self.K == 0 and len(cs) > k_max:            # evenly thin very long series
                cs = cs[np.round(np.linspace(0, len(cs) - 1, k_max)).astype(int)]
            oy = ox = 0
            if crop:
                if self.train:
                    oy, ox = int(rng.integers(0, H - crop + 1)), int(rng.integers(0, W - crop + 1))
                else:
                    oy, ox = (H - crop) // 2, (W - crop) // 2
            loaded.append((j, SLOTS[si][0], arr, step, cs, oy, ox))
        if not loaded:
            raise RuntimeError(f"study {row['StudyInstanceUID']} has no usable series")
        right = (row["side"] == "R") and self.orient
        Kp = self.K if self.K > 0 else max(len(t[4]) for t in loaded)
        h = w = crop if crop else None
        if h is None:
            h, w = loaded[0][2].shape[1:]
        x = np.zeros((S, Kp, 3, h, w), np.uint8)
        mask, pos, wmask = np.zeros(S, np.float32), np.zeros((S, Kp), np.float32), np.zeros((S, Kp), np.float32)
        for j, plane, arr, step, cs, oy, ox in loaded:
            T, H, W = arr.shape
            for k, c in enumerate(cs):
                ids = np.clip(np.array([c - step, c, c + step]), 0, T - 1)
                if right and plane == "Sagittal":
                    ids = (T - 1) - ids[::-1]          # reversed slice order, channel order lateral->medial
                win = np.asarray(arr[ids])
                if right and plane in ("Coronal", "Axial"):
                    win = win[:, :, ::-1]
                if crop:
                    win = win[:, oy:oy + h, ox:ox + w]
                x[j, k] = win
                p = c / max(T - 1, 1)
                pos[j, k] = (1 - p) if (right and plane == "Sagittal" and not self.sag_pos_fix) else p
                wmask[j, k] = 1.0
            mask[j] = 1.0
        out = {"x": torch.from_numpy(x), "mask": torch.from_numpy(mask), "pos": torch.from_numpy(pos), "i": i}
        if self.K == 0:
            out["wmask"] = torch.from_numpy(wmask)
        if self.labels is not None:
            out["y"] = torch.from_numpy(self.labels[i].astype(np.float32))
        if self.weights is not None:
            out["w"] = torch.tensor(float(self.weights[i]), dtype=torch.float32)
        return out


def pad_collate(batch):
    """Collate for K = 0 (all windows): pad x / pos / wmask along the window axis to the batch maximum."""
    kmax = max(b["x"].shape[1] for b in batch)
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key in ("x", "pos", "wmask"):
            padded = []
            for v in vals:
                pad = kmax - v.shape[1]
                if pad:
                    v = torch.cat([v, v.new_zeros((v.shape[0], pad) + tuple(v.shape[2:]))], 1)
                padded.append(v)
            vals = padded
        out[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else torch.tensor(vals)
    return out


def gpu_augment(x, mask, rot_deg=10.0, scale=0.10, shift=0.06, gamma=0.20, gain=0.10):
    """x: float [B,S,K,3,H,W] in [0,1]. One random affine + gamma/gain per (study, slot),
    shared by all windows of that series (keeps the 3 slices of a window aligned)."""
    B, S, K, C, H, W = x.shape
    n = B * S
    dev = x.device
    ang = (torch.rand(n, device=dev) * 2 - 1) * np.deg2rad(rot_deg)
    sc = 1 + (torch.rand(n, device=dev) * 2 - 1) * scale
    tx, ty = [(torch.rand(n, device=dev) * 2 - 1) * shift for _ in range(2)]
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    xs = x.reshape(n, K * C, H, W)
    grid = torch.nn.functional.affine_grid(theta, xs.shape, align_corners=False)
    xs = torch.nn.functional.grid_sample(xs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    g = torch.exp((torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * gamma)
    a = 1 + (torch.rand(n, 1, 1, 1, device=dev) * 2 - 1) * gain
    xs = (xs.clamp(0, 1) ** g * a).clamp(0, 1)
    return xs.reshape(B, S, K, C, H, W)


def load_labels(label_csv, uids, transform="soft"):
    lab = pd.read_csv(label_csv).set_index("StudyInstanceUID").reindex(uids)
    y = lab[TARGETS].values.astype(np.float32)
    if np.isnan(y).any():
        raise ValueError(f"{int(np.isnan(y).sum())} missing label cells for the requested studies")
    if transform == "hard":
        y = (y >= 0.5).astype(np.float32)
    return y, lab
