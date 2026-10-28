"""Compute R6 4-fold held-out predictions (P_r6_oof) and s6ft_v2 predictions on s6ft_cache_all.

R6 has 4 fold models: fold(uid) = int(sha1(uid)[:8], 16) % 4.
Cached windows are [W, 3, 320, 320]. They are resized 320 -> 384 and cropped to 352 (16:368, 16:368).
Evaluated with flip TTA (MCL plain-pass only) as in R6's native inference.
Also scores s6ft_v2 on the same cached studies at 288 crop.

Output:
  output/emergency_research/track2/r6_oof_cache.csv
  output/emergency_research/track2/s6ft_v2_cache.csv
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
R6_DIR = f"{PW}/rsna-knee-r6-fe055406"
S6_CODE = "D:/rsna_work/public_code/s6_src"
R6_SRC = "C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6"
OUT = Path(f"{R}/output/emergency_research/track2")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]

# Import R6 trainer
sys.path.insert(0, R6_SRC)
import compact_train as CT_R6
from compact_train import mirror_perm

# Import S6 trainer for s6ft_v2
sys.path.insert(0, S6_CODE)
import compact_train as CT_S6

def r6_fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4

def mirror_windows(img, meta):
    assert meta.shape[-1] >= 8, "v4c flip TTA needs the plane column (meta column 7)"
    sag = meta[..., 7].round() == 0
    img = torch.where(sag[:, :, None, None, None], img.flip(2), img.flip(-1))
    meta = meta.clone()
    meta[..., 2] = torch.where(sag, -meta[..., 2], meta[..., 2])
    v = meta[..., 1]
    meta[..., 1] = torch.where(v == 2, torch.full_like(v, 3.), torch.where(v == 3, torch.full_like(v, 2.), v))
    return img, meta

class CachedStudies(torch.utils.data.Dataset):
    def __init__(self, files, max_windows=160):
        self.files = files
        self.max_windows = max_windows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import cv2
        z = np.load(self.files[i])
        uid, jpg, off, meta = str(z["uid"]), z["jpg"], z["off"], z["meta"]
        k = len(meta)
        sel = np.unique(np.linspace(0, k - 1, min(k, self.max_windows)).round().astype(int)) if k else np.zeros(0, int)
        img = np.empty((len(sel), 3, 320, 320), np.uint8)
        for a, w in enumerate(sel):
            for ch in range(3):
                q = 3 * w + ch
                img[a, ch] = cv2.imdecode(jpg[off[q]:off[q + 1]], cv2.IMREAD_GRAYSCALE)
        return uid, torch.from_numpy(img), torch.from_numpy(meta[sel].astype(np.float32)), k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/rsna_work/s6ft_cache_all")
    ap.add_argument("--s6ft-weights", default="D:/rsna_work/s6ft_v2_local/s6ft_ema.pt")
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-fraction", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)

    # 1. Load R6 4 fold models
    print("Loading R6 4 fold models...", flush=True)
    r6_man = json.loads(Path(R6_DIR, "manifest.json").read_text())
    state_spec = json.load(open(Path(R6_DIR, r6_man["state_spec"]), encoding="utf-8"))
    r6_models = []
    for f in range(4):
        p = Path(R6_DIR, f"E_lrprior_s0_f{f}.pt")
        sd = torch.load(p, map_location="cpu", weights_only=True)
        kw = {}
        if any(k == "sp_q" or k.endswith(".sp_q") for k in sd): kw["spatial"] = True
        if any(k.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for k in sd): kw["extent"] = True
        if any(k == "syn_beta" or k.endswith(".syn_beta") for k in sd): kw["syn_decomp"] = True
        m = CT_R6.Net("convnext_small", None, freeze_blocks=16, crop=352, state_spec=state_spec, in_chans=3, **kw)
        m.load_state_dict(sd, strict=True)
        r6_models.append(m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last))
    print("R6 models loaded.", flush=True)

    # Mirror perm for R6
    perm, bin_perm = mirror_perm(state_spec, list(r6_models[0].bin_names))
    mcl_idx = LAB.index("MCL")

    # 2. Load s6ft_v2 model
    print("Loading s6ft_v2 model...", flush=True)
    s6ft_sd = torch.load(a.s6ft_weights, map_location="cpu", weights_only=True)
    m_s6ft = CT_S6.Net("convnext_small", None, freeze_blocks=16, crop=288, state_spec=None, in_chans=3, spatial=True)
    m_s6ft.load_state_dict(s6ft_sd, strict=True)
    m_s6ft = m_s6ft.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)
    print("s6ft_v2 model loaded.", flush=True)

    # Studies
    files = sorted(glob.glob(os.path.join(a.cache, "studies", "*.npz")))
    if a.limit:
        files = files[:a.limit]
    print(f"Total cached studies to score: {len(files)}", flush=True)

    dl = torch.utils.data.DataLoader(CachedStudies(files, a.windows), batch_size=None, shuffle=False,
                                     num_workers=a.workers, prefetch_factor=2 if a.workers else None)
    OUT.mkdir(parents=True, exist_ok=True)

    r6_rows, s6ft_rows = [], []
    t0 = time.time()

    with torch.inference_mode():
        for n, (uid, img, meta, k_all) in enumerate(dl, 1):
            fold = r6_fold(uid)
            w_count = img.shape[0]
            if w_count == 0:
                continue

            # --- A. Score R6 with flip TTA ---
            x_r6 = img.to(dev, non_blocking=True).float()
            x_r6 = F.interpolate(x_r6, size=(384, 384), mode="bilinear", align_corners=False).round_().clamp_(0, 255)
            x_r6 = x_r6[None, :, :, 16:368, 16:368].to(torch.uint8) # [1, W, 3, 352, 352]
            mt_r6 = meta.to(dev)[None] # [1, W, 9]
            mask = torch.ones((1, w_count), dtype=torch.bool, device=dev)

            # Plain pass
            with torch.autocast("cuda", dtype=torch.float16):
                z_plain = r6_models[fold](x_r6, mt_r6, mask, chunk=a.chunk).float()[0]

            # Mirrored pass
            x_m, mt_m = mirror_windows(x_r6, mt_r6)
            with torch.autocast("cuda", dtype=torch.float16):
                z_mirr = r6_models[fold](x_m, mt_m, mask, chunk=a.chunk).float()[0]

            # Flip TTA combine
            z_r6 = 0.5 * (z_plain + z_mirr[perm])
            z_r6[mcl_idx] = z_plain[mcl_idx]
            p_r6 = torch.sigmoid(z_r6.double()).cpu().numpy()
            r6_rows.append([uid, fold, w_count, int(k_all)] + p_r6.tolist())

            # --- B. Score s6ft_v2 ---
            x_s6 = img.to(dev, non_blocking=True)[None, :, :, 16:304, 16:304] # [1, W, 3, 288, 288]
            mt_s6 = meta.to(dev)[None]
            with torch.autocast("cuda", dtype=torch.float16):
                z_s6ft = m_s6ft(x_s6, mt_s6, mask, chunk=a.chunk).float()[0]
            p_s6ft = torch.sigmoid(z_s6ft.double()).cpu().numpy()
            s6ft_rows.append([uid, w_count, int(k_all)] + p_s6ft.tolist())

            if n <= 5 or n % 250 == 0 or n == len(files):
                elapsed = time.time() - t0
                speed = elapsed / n
                eta_min = (len(files) - n) * speed / 60
                print(f"[{n}/{len(files)}] {uid[:12]} | {speed:.2f} s/study | ETA: {eta_min:.1f} min", flush=True)

    # Save R6 OOF cache
    df_r6 = pd.DataFrame(r6_rows, columns=["StudyInstanceUID", "fold", "windows_used", "windows_total"] + LAB)
    out_r6_path = OUT / "r6_oof_cache.csv"
    df_r6.to_csv(out_r6_path, index=False)
    print(f"Wrote {out_r6_path} ({len(df_r6)} studies)", flush=True)

    # Save s6ft_v2 cache
    df_s6ft = pd.DataFrame(s6ft_rows, columns=["StudyInstanceUID", "windows_used", "windows_total"] + LAB)
    out_s6ft_path = OUT / "s6ft_v2_cache.csv"
    df_s6ft.to_csv(out_s6ft_path, index=False)
    print(f"Wrote {out_s6ft_path} ({len(df_s6ft)} studies)", flush=True)

if __name__ == "__main__":
    main()
