"""Plan section 14 C (mission 2026-10-21, Track 2A): S6's held-out-fold predictions on the all-window cache, the teacher of
s6ft_v2. S6 (heliosli/rsna-knee-compact-v1, fastMRI-free) has 4 fold models; a study's held-out fold is
int(sha1(uid)[:8], 16) % 4, and only that fold model scores it.
The cache (s6ft_cache.py) holds s6fall's 320 px rows; S6 was trained on 384 px rows with a 352 crop of the same physical box
(px_cache.view_geometry maps a fixed crop in mm onto the grid). So each window is resampled 320 -> 384 (bilinear, rounded
back to uint8), centre-cropped to 352, and scored with S6's plain pass (fp16 autocast; S6 has no mirror training).
At most --windows evenly spaced windows per study (the brief: 160).
--only oof300 first scores the 300 oof studies and compares them with S6's native held-out-fold logits from task1/s6
(full 384 render, every window): per-finding Spearman, written to s6_oof_cache_check.json.
Gold_58 is not in the cache. Usage:
  python -I src/emergency/s6_oof_cache.py --only oof300
  python -I src/emergency/s6_oof_cache.py            -> output/emergency_research/track2/s6_oof_cache.csv
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
import torch
import torch.nn.functional as F

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
S6 = f"{PW}/rsna-knee-compact-v1"
CODE = "D:/rsna_work/public_code/s6_src"
OUT = Path(f"{R}/output/emergency_research/track2")
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
       "Baker's", "Contusion", "Fracture"]


def s6_fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


class Studies(torch.utils.data.Dataset):
    def __init__(self, files, windows):
        self.files, self.windows = files, windows

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import cv2
        z = np.load(self.files[i])
        uid, jpg, off, meta = str(z["uid"]), z["jpg"], z["off"], z["meta"]
        k = len(meta)
        sel = np.unique(np.linspace(0, k - 1, min(k, self.windows)).round().astype(int)) if k else np.zeros(0, int)
        img = np.empty((len(sel), 3, 320, 320), np.uint8)
        for a, w in enumerate(sel):
            for ch in range(3):
                q = 3 * w + ch
                img[a, ch] = cv2.imdecode(jpg[off[q]:off[q + 1]], cv2.IMREAD_GRAYSCALE)
        return uid, torch.from_numpy(img), torch.from_numpy(meta[sel].astype(np.float32)), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/rsna_work/s6ft_cache_all")
    ap.add_argument("--only", default=None, choices=[None, "oof300"])
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    a = ap.parse_args()
    sys.path.insert(0, CODE)
    import compact_train as CT
    import pandas as pd
    assert list(CT.LABELS) == LAB
    assert hashlib.sha256(Path(CODE, "compact_train.py").read_bytes()).hexdigest().startswith("14767c7a"), "S6 trainer differs"
    man = json.loads(Path(S6, "manifest.json").read_text())
    for n in man["models"] + [man["state_spec"]]:
        assert hashlib.sha256(Path(S6, n).read_bytes()).hexdigest() == man["sha256"][n], n
    assert (man["size"], man["crop"], man["views"], man["slices"], man["phys_mm"]) == (384, 352, "v4", 3, 3.5)
    gold = set(pd.read_csv(f"{R}/data/labels_v3/gold_labels_original.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID)
    files = sorted(glob.glob(os.path.join(a.cache, "studies", "*.npz")))
    st = pd.read_csv(f"{R}/output/emergency_research/task1/task1_studies.csv", dtype={"StudyInstanceUID": str})
    oof300 = set(st.loc[st["set"] == "oof_300", "StudyInstanceUID"])
    uid_of = {}
    for f in files:
        with np.load(f) as z:
            uid_of[f] = str(z["uid"])
    assert not (set(uid_of.values()) & gold), "a gold study is in the cache"
    if a.only == "oof300":
        files = [f for f in files if uid_of[f] in oof300]
    print(f"S6 OOF on the cache: {len(files)} studies, <= {a.windows} windows each", flush=True)
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    spec = json.load(open(Path(S6, man["state_spec"]), encoding="utf-8"))
    models = []
    for p in sorted(Path(S6).glob(man["arm"] + "_s0_f*.pt")):
        sd = torch.load(p, map_location="cpu", weights_only=True)
        kw = {}
        if any(k == "sp_q" or k.endswith(".sp_q") for k in sd):
            kw["spatial"] = True
        if any(k.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for k in sd):
            kw["extent"] = True
        m = CT.Net("convnext_small", None, freeze_blocks=16, crop=352, state_spec=spec, in_chans=3, **kw)
        m.load_state_dict(sd, strict=True)
        models.append(m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last))
    assert len(models) == 4
    dl = torch.utils.data.DataLoader(Studies(files, a.windows), batch_size=None, shuffle=False, num_workers=a.workers,
                                     prefetch_factor=2 if a.workers else None)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with torch.inference_mode():
        for n, (uid, img, meta, k_all) in enumerate(dl, 1):
            fold = s6_fold(uid)
            x = img.to(dev, non_blocking=True).float()
            x = F.interpolate(x, size=(384, 384), mode="bilinear", align_corners=False).round_().clamp_(0, 255)
            x = x[None, :, :, 16:368, 16:368].to(torch.uint8)
            mt = meta.to(dev)[None]
            mask = torch.ones((1, x.shape[1]), dtype=torch.bool, device=dev)
            with torch.autocast("cuda", dtype=torch.float16):
                z = models[fold](x, mt, mask, chunk=a.chunk).float()
            if not torch.isfinite(z).all():
                z = models[fold](x, mt, mask, chunk=a.chunk).float()
            rows.append([uid, fold, int(x.shape[1]), int(k_all)] + z[0].cpu().tolist())
            if n <= 2 or n % 200 == 0:
                print(f"{n}/{len(files)} windows {x.shape[1]}/{k_all} | {(time.time() - t0) / n:.2f} s/study", flush=True)
    df = pd.DataFrame(rows, columns=["StudyInstanceUID", "s6_fold", "windows_used", "windows_total"] + [f"z|{c}" for c in LAB])
    for c in LAB:
        df[c] = 1 / (1 + np.exp(-df[f"z|{c}"]))
    tag = "_oof300" if a.only else ""
    df.to_csv(OUT / f"s6_oof_cache{tag}.csv", index=False)
    print(f"wrote {OUT / f's6_oof_cache{tag}.csv'}: {len(df)} studies in {(time.time() - t0) / 60:.1f} min", flush=True)
    if a.only == "oof300":
        from scipy.stats import spearmanr
        nat = {}
        for f in glob.glob(f"{R}/output/emergency_research/task1/s6/*.npz"):
            with np.load(f) as z:
                u = str(z["uid"])
                if u in oof300:
                    nat[u] = z["z_plain"][s6_fold(u)].astype(np.float64)
        d = df[df.StudyInstanceUID.isin(nat)]
        zc = d[[f"z|{c}" for c in LAB]].values
        zn = np.stack([nat[u] for u in d.StudyInstanceUID])
        res = {"studies": len(d), "spearman": {c: round(float(spearmanr(zc[:, j], zn[:, j]).correlation), 4) for j, c in enumerate(LAB)},
               "mean_abs_dlogit": {c: round(float(np.abs(zc[:, j] - zn[:, j]).mean()), 4) for j, c in enumerate(LAB)}}
        res["spearman_min"] = min(res["spearman"].values())
        (OUT / "s6_oof_cache_check.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
