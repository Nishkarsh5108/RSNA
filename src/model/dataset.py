"""
Dataset that loads preprocessed .npz MRI volumes at the study level,
grouping multiple series (planes / sequences) per study.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KneeMRIDataset(Dataset):
    """
    Each sample = one *study* (patient exam).
    Loads all available series for that study, samples slices, and pads to
    a fixed total so that standard batching works without a custom collate_fn.
    """

    def __init__(self, df, series_df, volumes_dir, config, is_train=True):
        self.config = config
        self.is_train = is_train
        self.volumes_dir = volumes_dir

        # Unique study IDs in this split
        self.study_ids = df["StudyInstanceUID"].unique().tolist()

        # Label lookup  {uid -> (12,) tensor}
        self.labels = {}
        for _, row in df.iterrows():
            uid = row["StudyInstanceUID"]
            self.labels[uid] = torch.tensor(
                [float(row[c]) for c in config.TARGET_COLS], dtype=torch.float32
            )

        # Group series metadata by study
        self.study_series = {}
        for uid in self.study_ids:
            mask = series_df["StudyInstanceUID"] == uid
            self.study_series[uid] = series_df.loc[mask].to_dict("records")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.study_ids)

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_slices(volume, n):
        """Uniformly sample *n* slices; pad with zeros if fewer."""
        total = volume.shape[0]
        if total == 0:
            return np.zeros((n, volume.shape[1], volume.shape[2]), dtype=np.uint8), \
                   np.zeros(n, dtype=np.float32)
        if total <= n:
            pad = np.zeros((n - total, volume.shape[1], volume.shape[2]), dtype=volume.dtype)
            return np.concatenate([volume, pad], axis=0), \
                   np.array([1.0] * total + [0.0] * (n - total), dtype=np.float32)
        idx = np.linspace(0, total - 1, n, dtype=int)
        return volume[idx], np.ones(n, dtype=np.float32)

    # ------------------------------------------------------------------
    def _load_series(self, series_uid):
        path = os.path.join(self.volumes_dir, f"{series_uid}.npz")
        if not os.path.exists(path):
            return None
        try:
            return np.load(path)["data"]  # (S, 224, 224) uint8
        except Exception:
            return None

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        uid = self.study_ids[idx]
        cfg = self.config
        series_list = self.study_series.get(uid, [])

        # Limit series count; prioritise diversity of planes
        if len(series_list) > cfg.MAX_SERIES_PER_STUDY:
            # Try to keep one per plane
            by_plane = {}
            for s in series_list:
                p = s.get("Anatomical_Plane", "Sagittal")
                by_plane.setdefault(p, []).append(s)
            chosen = []
            for p in ["Sagittal", "Coronal", "Axial"]:
                if p in by_plane:
                    chosen.append(by_plane[p][0])
            # Fill remaining slots
            remaining = [s for s in series_list if s not in chosen]
            chosen += remaining[: cfg.MAX_SERIES_PER_STUDY - len(chosen)]
            series_list = chosen[: cfg.MAX_SERIES_PER_STUDY]

        all_slices, all_planes, all_seqs, all_masks = [], [], [], []

        for s_info in series_list:
            vol = self._load_series(s_info["SeriesInstanceUID"])
            if vol is None:
                continue

            slices, mask = self._sample_slices(vol, cfg.MAX_SLICES_PER_SERIES)

            # Augmentation (training only)
            if self.is_train:
                if np.random.random() > 0.5:
                    slices = slices[:, :, ::-1].copy()

            plane_id = cfg.PLANE_MAP.get(s_info.get("Anatomical_Plane", "Sagittal"), 0)
            seq_id = int(s_info.get("Fluid_Sensitive", 0))

            all_slices.append(slices)
            all_planes.extend([plane_id] * cfg.MAX_SLICES_PER_SERIES)
            all_seqs.extend([seq_id] * cfg.MAX_SLICES_PER_SERIES)
            all_masks.append(mask)

        # Fallback if no series loaded
        if len(all_slices) == 0:
            all_slices.append(np.zeros((cfg.MAX_SLICES_PER_SERIES, cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.uint8))
            all_planes.extend([0] * cfg.MAX_SLICES_PER_SERIES)
            all_seqs.extend([0] * cfg.MAX_SLICES_PER_SERIES)
            all_masks.append(np.zeros(cfg.MAX_SLICES_PER_SERIES, dtype=np.float32))

        # Concatenate across series
        slices_np = np.concatenate(all_slices, axis=0)
        masks_np = np.concatenate(all_masks, axis=0)
        planes_np = np.array(all_planes, dtype=np.int64)
        seqs_np = np.array(all_seqs, dtype=np.int64)
        n = slices_np.shape[0]

        # Pad / truncate to MAX_TOTAL_SLICES for uniform batching
        T = cfg.MAX_TOTAL_SLICES
        if n > T:
            idx_sel = np.linspace(0, n - 1, T, dtype=int)
            slices_np = slices_np[idx_sel]
            masks_np = masks_np[idx_sel]
            planes_np = planes_np[idx_sel]
            seqs_np = seqs_np[idx_sel]
        elif n < T:
            pad = T - n
            slices_np = np.concatenate([slices_np, np.zeros((pad, cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.uint8)], 0)
            masks_np = np.concatenate([masks_np, np.zeros(pad, dtype=np.float32)], 0)
            planes_np = np.concatenate([planes_np, np.zeros(pad, dtype=np.int64)], 0)
            seqs_np = np.concatenate([seqs_np, np.zeros(pad, dtype=np.int64)], 0)

        return {
            "slices":    torch.from_numpy(slices_np).float() / 255.0,  # (T, 224, 224)
            "plane_ids": torch.from_numpy(planes_np),                  # (T,)
            "seq_ids":   torch.from_numpy(seqs_np),                    # (T,)
            "masks":     torch.from_numpy(masks_np),                   # (T,)
            "labels":    self.labels[uid],                              # (12,)
            "study_uid": uid,
        }
