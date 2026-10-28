# %% [code]
# %% [code]
#!/usr/bin/env python3
"""Knee MRI: training the twelve-finding model

This is the training half of the model behind the public 0.924 inference notebook. It reads the
precomputed slice stacks, trains a CoAtNet backbone with a per-finding attention pooling head, and
writes a checkpoint you can drop straight into that notebook.

WHAT YOU NEED ATTACHED

  1. The preprocessed corpus, both parts:
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus          (3,200 studies)
       kaggle.com/datasets/dreaddevelopment/knee-raptor-corpus-ext      (1,207 studies)
     Every study is already reduced to a fixed 44 x 336 x 336 uint8 stack, so no DICOM reading
     happens here. The two parts concatenate in order.

  2. The competition data, for train.csv.

  3. Training labels, as a parquet with a StudyInstanceUID column and the twelve finding columns.
     THIS IS NOT PROVIDED, and it is the one thing you have to bring yourself. See below.

THE LABEL PROBLEM, WHICH IS THE REAL PROBLEM

The competition gives you 4,407 studies and structured labels for only 58 of them. Every other
study carries a free-text radiology report and nothing else. So before any of this trains, you need
to turn 4,349 reports into twelve numbers each.

The approach behind the published weights was to read each report with a language model and emit
twelve probabilities rather than twelve yes or no answers: a report that hedges, saying a tear is
suspected, becomes something near 0.8 rather than a 1. Soft targets are far more forgiving than
forcing every hedged sentence into a hard label, and the loss here expects them. The 58 studies
that come with real labels are held out and used only for validation, never trained on.

Point --labels at your own parquet built that way. The format is one row per study: a
StudyInstanceUID column plus the twelve finding columns, values between 0 and 1.

WHAT THE MODEL DOES

Three neighbouring slices are stacked into the three channels of one image, so the network sees a
little of what lies above and below the middle slice: most of the benefit of a 3D model at the cost
of a 2D one. Each of these three-slice windows goes through the backbone, and the windows are then
pooled by an attention layer that has separate weights for each of the twelve findings. That last
part matters more than anything else here. A cruciate tear may be visible on two slices while
osteoarthritis spreads across many, and one shared pooling weight forces those to compete; giving
each finding its own attention lets each draw on the slices that actually show it.

Training samples k windows per study at random and evaluates on k_eval windows spread evenly, so
each epoch sees a different view of the same study. Nothing else is augmented.

At the end it keeps the best epoch by validation macro-AUC, and also writes a checkpoint that
averages the weights of the best three epochs. Weight averaging costs nothing at inference, unlike
averaging predictions from three models, and it usually gives a small gain.

TYPICAL RUN

  python train_knee.py --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 16
      --bs 8 --k 12 --k_eval 24 --grad_ckpt --tag mymodel --labels /kaggle/input/YOURS/labels.parquet

About three hours on one 4090 for 16 epochs at 384. --grad_ckpt trades a little speed for a lot of
memory and is what makes bs 8 fit on a 24 GB card. Use --smoke for a fast wiring check.

MEMORY, MEASURED

The CoAtNet at 384 with --bs 8 and --grad_ckpt sits at about 23.7 GB, so it needs a 24 GB card.
On 16 GB use --bs 2 or 4. Separately from the GPU, this script keeps the best checkpoint plus the
top --topk epochs' weights in system RAM so it can average them at the end; that is about 87 MB a
copy for the small ViT and 293 MB for the CoAtNet, and the list fills over the first few epochs.
If you run out of host memory a few epochs in, set --topk 1 and lower --workers.

The checkpoint it writes is a dict with keys model, arch, res and lab, which is exactly what the
inference notebook expects.
"""
import os, sys, time, json, math, random, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import timm

# --- ROI localizer (anatomical joint crop). Optional so the no-ROI path is untouched. ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from roi_localize import square_box as _roi_square_box, compartments as _roi_compartments
except Exception:
    _roi_square_box = _roi_compartments = None

HERE = os.path.dirname(os.path.abspath(__file__))
RSNA = os.path.dirname(HERE)

# ---------------------------------------------------------------------------
# Input discovery. On Kaggle the corpus arrives as two read-only datasets and the
# competition data as a third, so nothing lives beside this script. Everything in
# this block is discovery only - the training code further down is unchanged.
# ---------------------------------------------------------------------------
def _find(*names, root=os.environ.get("RAPTOR_INPUT", "/kaggle/input")):
    """First path under root whose basename matches one of names."""
    for d, _, fs in os.walk(root):
        for n in names:
            if n in fs:
                return os.path.join(d, n)
    return None


class _TwoPartVols:
    """Presents the two published corpus parts as one array of shape (4407, 44, 336, 336).

    Both parts stay memory-mapped and are never concatenated on disk: copying 22 GB would be
    pointless when every read is a single study. Row order is part 1 then part 2, matching the
    order the id files concatenate in. That ordering is the contract between volumes, masks and
    ids, so do not sort any of them independently.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = a.shape[0]
        self.shape = (a.shape[0] + b.shape[0],) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


_CORPUS = None   # RAPTOR_PRELOAD=1: the whole corpus in RAM, loaded once in the parent so forked loader workers share it


def _open_corpus():
    """Return (vols, masks), from a local single-file corpus or the two public parts.
    RAPTOR_PRELOAD=1 reads the arrays fully into RAM (22 GB) instead of memory-mapping them: on a Kaggle TPU VM
    (377 GB RAM) the input mount serves random slice reads too slowly to keep one TPU core busy."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    mm = None if os.environ.get("RAPTOR_PRELOAD") == "1" else "r"
    local_v = os.path.join(HERE, "all_vols.npy")
    if os.path.exists(local_v):
        out = (np.load(local_v, mmap_mode=mm), np.load(os.path.join(HERE, "all_masks.npy")))
    else:
        av, bv = _find("all_vols.npy"), _find("extra_vols.npy")
        am, bm = _find("all_masks.npy"), _find("extra_masks.npy")
        if not all((av, bv, am, bm)):
            raise SystemExit("Could not find the corpus. Attach both parts: "
                             "dreaddevelopment/knee-raptor-corpus and "
                             "dreaddevelopment/knee-raptor-corpus-ext")
        t0 = time.time()
        vols = _TwoPartVols(np.load(av, mmap_mode=mm), np.load(bv, mmap_mode=mm))
        masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
        if mm is None:
            print(f"[corpus] preloaded into RAM in {time.time()-t0:.0f}s", flush=True)
        out = (vols, masks)
    if mm is None:
        _CORPUS = out
    return out


def _open_ids():
    local = os.path.join(HERE, "all_ids.npy")
    if os.path.exists(local):
        return np.load(local, allow_pickle=True).astype(str)
    a, b = _find("all_ids.npy"), _find("extra_ids.npy")
    if not (a and b):
        raise SystemExit("Could not find all_ids.npy / extra_ids.npy - attach both corpus parts.")
    return np.concatenate([np.load(a, allow_pickle=True).astype(str),
                           np.load(b, allow_pickle=True).astype(str)])
LAB = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA",
       "Effusion","Synovitis","Baker's","Contusion","Fracture"]


def _slug(l):   # "Medial Meniscus" -> "Medial_Meniscus", "Baker's" -> "Bakers" (the m_/y_ naming in labels/*.parquet)
    return l.replace(" ", "_").replace("'", "")


def _read_table(p):
    d = pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
    d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    return d


def _read_uids(p):
    """Set of StudyInstanceUIDs from .parquet/.csv (StudyInstanceUID column), .json (list or {'ids': [...]}) or .txt."""
    if p.endswith(".json"):
        j = json.load(open(p))
        return set(map(str, j["ids"] if isinstance(j, dict) else j))
    if p.endswith((".parquet", ".csv")):
        return set(_read_table(p)["StudyInstanceUID"])
    return set(s.strip() for s in open(p) if s.strip())


_NO_LABELS = """
No training labels found, so there is nothing to train against.

The competition labels only 58 of the 4,407 studies. The other 4,349 carry a free-text
radiology report instead, so before this can train you have to turn those reports into
twelve probabilities per study and pass the result with --labels.

Expected format: a parquet with a StudyInstanceUID column plus the columns
  {cols}
with values between 0 and 1. Soft values work better than hard 0/1 here: the loss is built
for them, and hedged reports are common.

Everything else in this notebook is ready to run once that file exists.
"""


# ------------------------------- data ----------------------------------------
CROP_FRAC = float(os.environ.get("RAPTOR_CROP_FRAC", "1.0"))   # 1.0 = OFF (full 140 mm field)
# plane-aware bag (ported 2026-09-26 from research/label-code-audit/train_knee_audit.py; all OFF by default)
SLOT_SIZES = [int(v) for v in os.environ.get("RAPTOR_SLOTS", "26,22,18,12,18").split(",")]
N_POS_BINS = 8
SAGITTAL_SLOTS, CORAX_SLOTS = (0, 1), (2, 3, 4)
def _slot_index(D):
    b = np.cumsum([0] + SLOT_SIZES)
    if b[-1] != D:
        raise RuntimeError(f"RAPTOR_SLOTS sums to {b[-1]} but the corpus has {D} slices per study")
    s = np.searchsorted(b, np.arange(D), side="right") - 1
    rel = (np.arange(D) - b[s]) / np.maximum(1, np.array(SLOT_SIZES)[s] - 1)
    return s, rel


class StudyWindows(Dataset):
    """Per-study bag of 2.5D windows sampled from all_vols.npy (memmap).
    Each window = 3 physically-consecutive slices -> RGB, resized to `res`, in [0,1]
    (matches the SSL input pipeline: ToTensor, no ImageNet norm)."""
    def __init__(self, root, ids, id2row, labels, res, k, train, aug=True, norm="none",
                 roi=False, roi_mode="tight", roi_pad=0.06, roi_overlap=0.12,
                 roi_sbox=None, roi_cen=None, aug_affine=0.0, aug_rot=10.0, aug_scale=0.10, aug_shift=0.06,
                 clean=False, strat=False, lat=None, pos=False):
        self.root = root
        self.clean, self.strat = bool(clean), bool(strat)
        self.lat, self.pos = lat, bool(pos)          # lat: dict uid -> is_left (only used with --lat_norm)
        # spatial aug (Danial 2026-09-25 "go"): per-window random affine with prob aug_affine; 0 = OFF = unchanged
        self.aug_affine, self.aug_rot, self.aug_scale, self.aug_shift = aug_affine, aug_rot, aug_scale, aug_shift
        self.ids = ids
        self.id2row = id2row
        self.labels = labels              # dict uid -> np.float32[12]
        self.res, self.k, self.train, self.aug = res, k, train, aug
        self.norm = norm                  # "none"=[0,1] (Raptor SSL); "imagenet"=DINOv2 stats
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.vols = None; self.masks = None
        # --- ROI anatomical joint-crop config ---
        self.roi = bool(roi)
        self.roi_mode, self.roi_pad, self.roi_overlap = roi_mode, roi_pad, roi_overlap
        self.roi_sbox = roi_sbox          # (N,D,4) int16 per-slice tissue bbox, or None
        self.roi_cen = roi_cen            # (N,D,2) int16 per-slice joint centroid, or None
        if self.roi and (_roi_square_box is None or roi_sbox is None):
            raise RuntimeError("roi=True but roi_localize or roi_boxes not available")

    def __len__(self): return len(self.ids)

    def _ensure(self):
        if self.vols is None:
            self.vols, self.masks = _open_corpus()   # (N,D,H,W) view, (N,D) u8

    def _centers(self, valid, count):
        # valid slice indices; window centers must have both neighbors valid & in-range
        lo, hi = int(valid.min()), int(valid.max())
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        if not cs: cs = [max(1, min((lo + hi) // 2, self._D - 2))]
        if self.train:
            reps = count // len(cs) + 1
            pool = (cs * reps)
            random.shuffle(pool)
            return pool[:count]
        # eval: evenly spaced deterministic
        idx = np.linspace(0, len(cs) - 1, count).round().astype(int)
        return [cs[i] for i in idx]

    def _affine(self, x):
        """x: (K,3,res,res) in [0,1]. Each window, with prob aug_affine: rotation +-aug_rot deg, isotropic scale
        1 +- aug_scale, shift +- aug_shift of the image width, zero padding, the same transform on its 3 slices; NO flips
        (laterality is signal). Draws come from a separate RNG, so window sampling and the intensity jitter are identical
        to a run without this aug (same seed -> same centers)."""
        seed = torch.initial_seed()
        if getattr(self, "_arng_seed", None) != seed:
            self._arng, self._arng_seed = random.Random(seed ^ 0x5EED5EED), seed
        r = self._arng
        sel = [j for j in range(x.shape[0]) if r.random() < self.aug_affine]
        if not sel:
            return x
        th = torch.zeros(len(sel), 2, 3)
        for n in range(len(sel)):
            a = math.radians(r.uniform(-self.aug_rot, self.aug_rot)); s = 1.0 + r.uniform(-self.aug_scale, self.aug_scale)
            c, si = math.cos(a) / s, math.sin(a) / s   # grid maps output -> input, so the inverse scale enlarges by s
            th[n] = torch.tensor([[c, -si, 2 * r.uniform(-self.aug_shift, self.aug_shift)],
                                  [si, c, 2 * r.uniform(-self.aug_shift, self.aug_shift)]])
        xs = x[sel]
        grid = F.affine_grid(th, list(xs.shape), align_corners=False)
        x = x.clone(); x[sel] = F.grid_sample(xs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return x

    def _clean_centers(self, m, count):
        D = len(m); s, _ = _slot_index(D)
        ok = [c for c in range(1, D - 1) if s[c - 1] == s[c] == s[c + 1] and m[c - 1] and m[c] and m[c + 1]]
        if not ok:
            v = np.where(m > 0)[0]
            return self._centers(v if len(v) >= 3 else np.arange(min(3, D)), count)
        if not self.train:
            idx = np.linspace(0, len(ok) - 1, count).round().astype(int)
            return [ok[i] for i in idx]
        if not self.strat:
            pool = ok * (count // len(ok) + 1); random.shuffle(pool)
            return pool[:count]
        by = {}
        for c in ok:
            by.setdefault(int(s[c]), []).append(c)
        first = [random.choice(v) for v in by.values()][:count]
        rest = [c for c in ok if c not in set(first)]
        random.shuffle(rest)
        out = first + (rest * ((count - len(first)) // max(1, len(rest)) + 1))[:max(0, count - len(first))]
        random.shuffle(out)
        return out

    def _resize(self, tri):
        """tri: (3,h,w) float32 [0,1] -> (3,res,res) float32."""
        if CROP_FRAC < 1.0:   # pixel-size test (2026-09-25): centre crop of the 140 mm field before the resize, train AND eval
            H, W = tri.shape[-2], tri.shape[-1]; ch, cw = int(round(H * CROP_FRAC)), int(round(W * CROP_FRAC))
            y0, x0 = (H - ch) // 2, (W - cw) // 2; tri = tri[:, y0:y0 + ch, x0:x0 + cw]
        t = torch.from_numpy(np.ascontiguousarray(tri))
        if t.shape[-1] != self.res or t.shape[-2] != self.res:
            t = F.interpolate(t[None], size=(self.res, self.res), mode="bilinear",
                              align_corners=False)[0]
        return t.numpy()

    def __getitem__(self, i):
        self._ensure()
        uid = self.ids[i]; row = self.id2row[uid]
        self._D = self.vols.shape[1]
        m = self.masks[row]
        valid = np.where(m > 0)[0]
        if len(valid) < 3: valid = np.arange(min(3, self._D))
        # compartment mode emits 2 crops/center -> sample ceil(k/2) centers to keep #windows==k
        compart = self.roi and self.roi_mode == "compartment"
        n_centers = (self.k + 1) // 2 if compart else self.k
        cs = self._clean_centers(m, n_centers) if self.clean else self._centers(valid, n_centers)
        vol = self.vols[row]  # (D,H,W) u8  (single study read)
        tiles = []            # list of (3,res,res) float32
        left = bool(self.lat.get(uid, False)) if self.lat is not None else False
        need_slots = left or self.pos
        s_idx, rel = _slot_index(self._D) if need_slots else (None, None)
        pids = []
        for c in cs:
            c = max(1, min(c, self._D - 2))
            tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0  # (3,H,W)
            if need_slots:
                q = int(s_idx[c]); r = float(rel[c])
                if left and q in CORAX_SLOTS:
                    tri = tri[:, :, ::-1]                    # mirror to right-knee layout (medial on image right)
                if left and q in SAGITTAL_SLOTS:
                    tri = tri[::-1]; r = 1.0 - r            # stack direction medial->lateral for every knee
                pids.append(q * N_POS_BINS + min(N_POS_BINS - 1, int(r * N_POS_BINS)))
            H, W = tri.shape[-2], tri.shape[-1]
            if not self.roi:
                tiles.append(self._resize(tri))
                continue
            # one joint box from the CENTER slice, applied to all 3 slices (keeps RGB registered)
            sq_mode = "tight" if compart else self.roi_mode
            sq = _roi_square_box(tuple(int(v) for v in self.roi_sbox[row, c]), W=W, H=H,
                                 pad=self.roi_pad, mode=sq_mode,
                                 centroid=tuple(float(v) for v in self.roi_cen[row, c]))
            if compart:
                left, right = _roi_compartments(sq, overlap=self.roi_overlap)
                for box in (left, right):
                    x0, y0, x1, y1 = box
                    tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
            else:
                x0, y0, x1, y1 = sq
                tiles.append(self._resize(tri[:, y0:y1, x0:x1]))
        if len(tiles) > self.k:
            tiles = tiles[:self.k]
        wins = np.stack(tiles, 0)          # (K,3,res,res)
        x = torch.from_numpy(wins)
        if self.train and self.aug:
            # light medical-safe aug: NO flips (laterality is signal); mild intensity jitter
            g = 1.0 + (random.random() - 0.5) * 0.20
            x = (x * g).clamp(0, 1)
        if self.train and self.aug and self.aug_affine > 0:
            x = self._affine(x)
        if self.norm == "imagenet":        # each backbone at its correct input distribution
            x = (x - self._mean) / self._std
        y = torch.from_numpy(self.labels[uid])
        if self.pos:
            return x, y, torch.tensor(pids[:self.k], dtype=torch.long)
        return x, y


def _winit(_): torch.set_num_threads(1)


def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,K,3,res,res)
    ys = torch.stack([b[1] for b in batch])   # (B,12)
    if len(batch[0]) > 2:
        return xs, ys, torch.stack([b[2] for b in batch])
    return xs, ys


# ------------------------------- model ---------------------------------------
def build_backbone(arch="vit_small_patch16_224", pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    if is_vit:
        kw.update(global_pool="token", dynamic_img_size=True)
    else:
        kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


def load_raptor(bb, ckpt_path):
    if ckpt_path in ("timm", "pretrained"):
        return "timm-pretrained"
    if ckpt_path in ("", "none", "None"):
        print("[raptor] RANDOM-INIT control (no SSL weights)", flush=True)
        return "random-init"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    st = ck["student"] if "student" in ck else (ck["model"] if "model" in ck else ck)   # SSL student, or a released RaptorClassifier
    bbst = {k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")}
    missing, unexpected = bb.load_state_dict(bbst, strict=False)
    ep = ck.get("epoch", "?")
    print(f"[raptor] loaded backbone from {os.path.basename(ckpt_path)} (ssl epoch {ep}) | "
          f"loaded {len(bbst)} tensors, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    return f"{os.path.basename(ckpt_path)}@ep{ep}"


class RaptorClassifier(nn.Module):
    """Raptor encoder + per-diagnosis attention-MIL head (12 findings)."""
    def __init__(self, backbone, F_dim=384, n=12, drop=0.2, n_pos=0):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop),
                                 nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n
        if n_pos > 0:                              # slot x position-bin embedding for the attention input only
            self.pos_emb = nn.Embedding(n_pos, F_dim); nn.init.zeros_(self.pos_emb.weight)

    win_chunk = 0   # >0: run the backbone over chunks of this many windows under activation checkpointing (8 GB cards)

    def encode(self, x):
        B, K = x.shape[:2]
        flat = x.flatten(0, 1)
        if self.win_chunk > 0 and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            f = torch.cat([checkpoint(self.backbone, c, use_reentrant=False) for c in flat.split(self.win_chunk, 0)], 0)
        elif self.win_chunk > 0:
            f = torch.cat([self.backbone(c) for c in flat.split(self.win_chunk, 0)], 0)
        else:
            f = self.backbone(flat)
        return f.view(B, K, -1)

    def head(self, feats, pos=None):
        h = self.norm(feats)
        a = self.att(h + self.pos_emb(pos)) if (pos is not None and hasattr(self, "pos_emb")) else self.att(h)
        a = torch.softmax(a, dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        logits = (pooled * self.clsW).sum(-1) + self.clsb
        return logits

    def forward(self, x, pos=None):
        return self.head(self.encode(x), pos)


# ------------------------------- train ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Defaults to ImageNet-pretrained weights, which is what every released arm was trained from.
    # The previous default pointed at a self-supervised checkpoint that is not distributed here,
    # so following the example run above failed on a missing file.
    ap.add_argument("--ckpt", default="timm",
                    help="'timm' for ImageNet-pretrained weights, or a path to your own "
                         "self-supervised checkpoint")
    ap.add_argument("--arch", default="vit_small_patch16_224")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_eval", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (effective batch = bs * accum)")
    ap.add_argument("--win_chunk", type=int, default=0, help="checkpointed backbone chunks of N windows (memory); 0 = whole study at once")
    ap.add_argument("--oof_every", type=int, default=0, help="score a fixed N-study subset of the held-out fold every epoch (masked AUC vs soft labels)")
    ap.add_argument("--oof_mask", default=None, help="parquet with m_<label> columns (1 = supervised) for the per-epoch OOF subset")
    ap.add_argument("--fp16", action="store_true", help="fp16 autocast + loss scaling (T4 has no bf16)")
    ap.add_argument("--xla", action="store_true", help="train on a TPU core through torch_xla (bf16 autocast, mark_step per micro-batch, no grad scaler)")
    ap.add_argument("--resume", default=None, help="resume state file written by --time_budget_h")
    ap.add_argument("--time_budget_h", type=float, default=0.0, help="stop after the epoch that crosses this wall-clock budget and write a resume state")
    ap.add_argument("--bb_lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--freeze_blocks", type=int, default=0)
    # Each kept epoch is a full copy of the weights in system RAM (~87 MB small ViT, ~293 MB
    # CoAtNet). Lower to 1 on a memory-tight machine; at 1 no averaged checkpoint is written.
    ap.add_argument("--topk", type=int, default=3,
                    help="top epochs kept for weight averaging (1 disables averaging)")
    ap.add_argument("--norm", default="none", choices=["none", "imagenet"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--labels", default=None,
                    help="parquet of training labels: StudyInstanceUID + the twelve finding "
                         "columns, values 0..1. Not provided with this notebook - see the header.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clean_windows", action="store_true", help="window centres only where c-1,c,c+1 sit in ONE slot and are non-empty")
    ap.add_argument("--strat", action="store_true", help="training bag takes >=1 window from every present slot (needs --clean_windows)")
    ap.add_argument("--lat_file", default=None, help="parquet StudyInstanceUID,is_left")
    ap.add_argument("--lat_norm", action="store_true", help="mirror left knees to the right-knee layout (needs --lat_file)")
    ap.add_argument("--pos_emb", action="store_true", help="slot x depth-bin embedding added to the attention input")
    ap.add_argument("--aug_affine", type=float, default=0.0, help="per-window prob of a random affine (rot/scale/shift, no flip); 0 = off")
    ap.add_argument("--aug_rot", type=float, default=10.0, help="max rotation, degrees")
    ap.add_argument("--aug_scale", type=float, default=0.10, help="max isotropic scale change (fraction)")
    ap.add_argument("--aug_shift", type=float, default=0.06, help="max shift, fraction of the image width")
    # ---- CV fold hook ----
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--fold_file", default=None)
    # ---- anatomical ROI joint-crop (A/B lever). Default OFF -> identical to baseline. ----
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--roi_mode", default="compartment", choices=["tight", "safe", "compartment"])
    ap.add_argument("--roi_pad", type=float, default=0.06)
    ap.add_argument("--roi_overlap", type=float, default=0.12)
    ap.add_argument("--roi_boxes", default=os.path.join(HERE, "roi_boxes.npz"))
    # ---- label-noise levers. Every default is OFF -> the baseline code path runs unchanged. ----
    ap.add_argument("--train_subset", default=None, help="StudyInstanceUIDs to TRAIN on (.parquet/.csv with a StudyInstanceUID column, "
                    ".json list or {'ids': [...]}, or .txt one per line). Applied AFTER the fold split: the OOF fold stays whole")
    ap.add_argument("--cell_mask", default=None, help="parquet: StudyInstanceUID + m_<finding> (slug or LAB name); the value is the "
                    "loss weight of that cell, 0 drops it. Studies absent from the file keep weight 1")
    ap.add_argument("--study_weights", default=None, help="parquet/csv: StudyInstanceUID + weight; multiplies all 12 cells of that "
                    "study (0 = drop the study but keep the batch order and schedule of the full run)")
    ap.add_argument("--pw_labels", default=None, help="parquet to take pos_weight prevalence from, over the pre-subset fold-train "
                    "studies (pins pos_weight across subset/mask arms); default = weighted prevalence of the cells trained on")
    ap.add_argument("--init_from", default=None, help="raptor_ft_<tag>.pt / _swa.pt: load the WHOLE model (backbone + head) before "
                    "training; optimizer and schedule start fresh (unlike --resume)")
    ap.add_argument("--pct_start", type=float, default=0.15, help="OneCycle warm-up fraction; 0 = cosine decay from the peak LR")
    ap.add_argument("--loss", default="bce", choices=["bce", "gce", "sce", "boot"], help="bce = baseline soft BCE; gce = generalized "
                    "CE (--gce_q); sce = symmetric CE (--sce_alpha CE + --sce_beta reverse CE); boot = soft bootstrap (--boot_beta)")
    ap.add_argument("--gce_q", type=float, default=0.7)
    ap.add_argument("--sce_alpha", type=float, default=1.0)
    ap.add_argument("--sce_beta", type=float, default=1.0)
    ap.add_argument("--boot_beta", type=float, default=0.8, help="target = beta*y + (1-beta)*sigmoid(logit).detach()")
    ap.add_argument("--trunc_frac", type=float, default=0.0, help="small-loss truncation: per finding, drop cells whose KL(y||p) is "
                    "above the (1-frac) quantile of the last --trunc_buf cells (0 = off)")
    ap.add_argument("--trunc_ramp", type=float, default=2.0, help="epochs over which the drop fraction ramps from 0 to --trunc_frac")
    ap.add_argument("--trunc_buf", type=int, default=256, help="per-finding loss history for the threshold (micro-batches are tiny)")
    ap.add_argument("--cell_log", action="store_true", help="write raptor_ft_<tag>_cells.npz: per-epoch KL(y||p) and drop flag for "
                    "every trained (study, finding) cell - a loss-trajectory noise ranking (on automatically with --trunc_frac)")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if CROP_FRAC < 1.0: print(f"[crop] centre crop {CROP_FRAC} of the field before resizing (RAPTOR_CROP_FRAC)", flush=True)
    XLA = None
    if a.xla:
        import torch_xla, torch_xla.core.xla_model as xm
        XLA = xm; dev = torch_xla.device()
        print(f"[xla] device {dev} | torch_xla {torch_xla.__version__}", flush=True)
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    DEVT = "xla" if XLA else dev                      # autocast device_type string
    WIF = _winit if XLA else None                     # loader worker_init_fn (XLA path only)
    AMP_DTYPE = torch.float16 if (a.fp16 or (dev == "cuda" and not torch.cuda.is_bf16_supported())) else torch.bfloat16
    AMP_ON = (dev == "cuda") or (XLA is not None)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda" and AMP_DTYPE == torch.float16))
    print(f"[amp] dtype {AMP_DTYPE} | autocast {AMP_ON} | grad scaler {scaler.is_enabled()}", flush=True)
    if a.smoke:
        a.epochs, a.bs, a.k, a.k_eval, a.limit, a.workers = 2, 4, 4, 6, 40, 0
    print(f"device {dev} | res {a.res} | k {a.k}/{a.k_eval} | bs {a.bs} | tag {a.tag} | "
          f"roi={a.roi}({a.roi_mode})", flush=True)

    # ---- ids / labels ----
    ids = _open_ids()
    id2row = {u: i for i, u in enumerate(ids)}
    idset = set(ids)
    _tcsv = os.path.join(RSNA, "train.csv")
    if not os.path.exists(_tcsv):
        _tcsv = _find("train.csv")
    if not _tcsv:
        raise SystemExit("Could not find train.csv - attach the competition data.")
    tr = pd.read_csv(_tcsv); tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold_df = tr[tr[LAB].notna().all(axis=1)].copy().set_index("StudyInstanceUID")
    gold_ids = [u for u in gold_df.index if u in idset]
    _lab = a.labels or _find("labels_llm_soft.parquet")
    if not _lab or not os.path.exists(_lab):
        # Exit cleanly rather than as a failure: running this notebook as published, with no
        # labels attached, is the expected path and should read as an explanation, not a crash.
        print(_NO_LABELS.format(cols=", ".join(LAB)), flush=True)
        raise SystemExit(0)
    soft = pd.read_parquet(_lab)
    soft["StudyInstanceUID"] = soft["StudyInstanceUID"].astype(str); soft = soft.set_index("StudyInstanceUID")
    goldset = set(gold_ids)
    train_ids = [u for u in ids if u in soft.index and u not in goldset]
    # ---- CV fold hook: hold out fold `a.fold`, train on the rest ----
    oof_ids = []
    if a.folds > 0:
        assert 0 <= a.fold < a.folds, f"--fold must be in [0,{a.folds}) when --folds>0"
        fmap = json.load(open(a.fold_file))["folds"]
        held = set(u for u in train_ids if fmap.get(u, -1) == a.fold)
        oof_ids = [u for u in train_ids if u in held]
        train_ids = [u for u in train_ids if u not in held]
        print(f"[cv] fold {a.fold}/{a.folds}: train {len(train_ids)} | OOF held-out {len(oof_ids)} "
              f"| fold_file {os.path.basename(a.fold_file)}", flush=True)
    train_ids_full = list(train_ids)                                  # fold-train set before any subset (for --pw_labels)
    if a.train_subset:
        keep = _read_uids(a.train_subset)
        train_ids = [u for u in train_ids if u in keep]
        print(f"[subset] {os.path.basename(a.train_subset)}: training on {len(train_ids)}/{len(train_ids_full)} studies "
              f"({len(keep)} ids in file; OOF held-out unchanged)", flush=True)
    if a.limit: train_ids = train_ids[:a.limit]
    labels = {u: soft.loc[u, LAB].values.astype(np.float32) for u in train_ids}
    for u in oof_ids: labels[u] = soft.loc[u, LAB].values.astype(np.float32)
    for u in gold_ids: labels[u] = gold_df.loc[u, LAB].values.astype(np.float32)
    print(f"train {len(train_ids)} | gold-val {len(gold_ids)}"
          + (f" | oof {len(oof_ids)}" if oof_ids else ""), flush=True)

    # ---- per-cell weights: NaN label, --cell_mask, --study_weights. CELL=False keeps the baseline loss verbatim. ----
    n_nan = int(sum(np.isnan(labels[u]).sum() for u in train_ids))
    CELL = bool(n_nan or a.cell_mask or a.study_weights or a.loss != "bce" or a.trunc_frac > 0 or a.cell_log)
    cw = None
    if CELL:
        cw = {u: np.where(np.isnan(labels[u]), 0.0, 1.0).astype(np.float32) for u in train_ids}
        if a.cell_mask:
            mk = pd.read_parquet(a.cell_mask); mk["StudyInstanceUID"] = mk["StudyInstanceUID"].astype(str)
            mk = mk.set_index("StudyInstanceUID")
            mcols = [next((c for c in (f"m_{_slug(l)}", f"m_{l}") if c in mk.columns), None) for l in LAB]
            if None in mcols:
                raise SystemExit(f"--cell_mask lacks m_ columns for {[l for l, c in zip(LAB, mcols) if c is None]}")
            for u in train_ids:
                if u in mk.index: cw[u] *= mk.loc[u, mcols].values.astype(np.float32)
        if a.study_weights:
            sw = _read_table(a.study_weights).set_index("StudyInstanceUID")["weight"].astype(float).to_dict()
            for u in train_ids: cw[u] *= np.float32(sw.get(u, 1.0))
        Wm = np.stack([cw[u] for u in train_ids])
        print(f"[cells] loss {a.loss} | trunc {a.trunc_frac} | NaN label cells {n_nan} | weight-0 cells "
              f"{int((Wm == 0).sum())}/{Wm.size} | kept per finding {np.round((Wm > 0).mean(0), 3).tolist()}", flush=True)
    if a.pw_labels:
        pl = pd.read_parquet(a.pw_labels); pl["StudyInstanceUID"] = pl["StudyInstanceUID"].astype(str)
        pl = pl.set_index("StudyInstanceUID"); pl = pl.loc[[u for u in train_ids_full if u in pl.index]]
        prev = np.clip(np.nanmean(pl[LAB].values.astype(np.float64), 0), 0.03, 0.7)
        print(f"[pw] prevalence from {os.path.basename(a.pw_labels)} over {len(pl)} fold-train studies", flush=True)
    elif CELL:
        Ym = np.nan_to_num(np.stack([labels[u] for u in train_ids]))
        prev = np.clip((Ym * Wm).sum(0) / np.maximum(Wm.sum(0), 1e-6), 0.03, 0.7)
    else:
        prev = np.clip(np.stack([labels[u] for u in train_ids]).mean(0), 0.03, 0.7)
    pw = torch.tensor(np.clip((1 - prev) / prev, 1, 10), dtype=torch.float32, device=dev)
    if (CELL or a.pw_labels) and not torch.isfinite(pw).all(): raise SystemExit(f"non-finite pos_weight from prevalence {prev}")
    if CELL or a.pw_labels: print(f"[pw] pos_weight {np.round(pw.cpu().numpy(), 2).tolist()}", flush=True)
    # The training set sees [y (NaN->0) | w | row index] so StudyWindows and collate stay untouched.
    tlabels = labels if not CELL else {u: np.concatenate([np.nan_to_num(labels[u]), cw[u], [i]]).astype(np.float32)
                                       for i, u in enumerate(train_ids)}

    # ---- ROI boxes (only when --roi) ----
    roi_sbox = roi_cen = None
    if a.roi:
        rb = np.load(a.roi_boxes)
        rb_ids = rb["ids"].astype(str)
        if not np.array_equal(rb_ids, ids):
            rmap = {u: i for i, u in enumerate(rb_ids)}
            missing = [u for u in ids if u not in rmap]
            if missing:
                raise RuntimeError(f"roi_boxes missing {len(missing)} corpus ids (e.g. {missing[:2]})")
            order = np.array([rmap[u] for u in ids])
            roi_sbox = rb["sbox"][order]; roi_cen = rb["cen"][order]
        else:
            roi_sbox = rb["sbox"]; roi_cen = rb["cen"]
        print(f"[roi] ENABLED mode={a.roi_mode} pad={a.roi_pad} overlap={a.roi_overlap} "
              f"boxes={os.path.basename(a.roi_boxes)} sbox={roi_sbox.shape}", flush=True)
    _roi_kw = dict(roi=a.roi, roi_mode=a.roi_mode, roi_pad=a.roi_pad, roi_overlap=a.roi_overlap,
                   roi_sbox=roi_sbox, roi_cen=roi_cen)

    if os.environ.get("RAPTOR_PRELOAD") == "1":
        _open_corpus()                                # parent holds the RAM copy; forked workers inherit it
    def _P(pp): return pp[0].to(dev) if pp else None
    LAT = None
    if a.lat_norm:
        _lf = pd.read_parquet(a.lat_file); LAT = dict(zip(_lf["StudyInstanceUID"].astype(str), _lf["is_left"].astype(bool)))
        print(f"[plane] laterality normalisation ON: {sum(LAT.values())} left knees of {len(LAT)}", flush=True)
    _roi_kw.update(clean=a.clean_windows, strat=a.strat, lat=LAT, pos=a.pos_emb)
    if a.clean_windows or a.lat_norm or a.pos_emb: print(f"[plane] clean_windows={a.clean_windows} strat={a.strat} lat_norm={a.lat_norm} pos_emb={a.pos_emb}", flush=True)
    tds = StudyWindows(HERE, train_ids, id2row, tlabels, a.res, a.k, train=True, norm=a.norm, **_roi_kw,
                       aug_affine=a.aug_affine, aug_rot=a.aug_rot, aug_scale=a.aug_scale, aug_shift=a.aug_shift)
    if a.aug_affine > 0: print(f"[aug] affine p={a.aug_affine} rot +-{a.aug_rot} deg, scale +-{a.aug_scale}, shift +-{a.aug_shift}", flush=True)
    vds = StudyWindows(HERE, gold_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
    tl = DataLoader(tds, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    collate_fn=collate, pin_memory=(dev == "cuda"), persistent_workers=a.workers > 0, worker_init_fn=WIF)
    vl = DataLoader(vds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                    collate_fn=collate, persistent_workers=a.workers > 0, worker_init_fn=WIF)
    sl = None; sub_mask = None
    if a.oof_every > 0 and a.folds > 0 and oof_ids:
        sub_ids = random.Random(0).sample(list(oof_ids), min(a.oof_every, len(oof_ids)))
        sds = StudyWindows(HERE, sub_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        sl = DataLoader(sds, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers, collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
        if a.oof_mask:
            _mk = pd.read_parquet(a.oof_mask).set_index("StudyInstanceUID")
            _mc = [c for c in _mk.columns if c.startswith("m_")]
            sub_mask = np.stack([_mk.loc[u, _mc].values.astype(np.float32) if u in _mk.index else np.ones(12, np.float32) for u in sub_ids])
        else:
            sub_mask = np.ones((len(sub_ids), 12), np.float32)
        print(f"[oof-sub] {len(sub_ids)} held-out studies scored every epoch (mask: {'file' if a.oof_mask else 'none'})", flush=True)

    use_timm = a.ckpt in ("timm", "pretrained") and not a.init_from
    bb = build_backbone(a.arch, pretrained=use_timm)
    src = load_raptor(bb, a.ckpt)
    if use_timm: print(f"[raptor] timm-pretrained backbone: {a.arch}", flush=True)
    F_dim = bb.num_features
    if a.grad_ckpt:
        try:
            bb.set_grad_checkpointing(True); print("[raptor] gradient checkpointing ON", flush=True)
        except Exception as e:
            print(f"[raptor] grad_ckpt unsupported for {a.arch}: {e}", flush=True)
    model = RaptorClassifier(bb, F_dim=F_dim, n_pos=(len(SLOT_SIZES) * N_POS_BINS if a.pos_emb else 0)).to(dev)
    model.win_chunk = a.win_chunk
    if a.init_from:
        ck0 = torch.load(a.init_from, map_location="cpu", weights_only=False)
        if ck0.get("arch", a.arch) != a.arch:
            raise SystemExit(f"--init_from was trained as {ck0['arch']}, this run is {a.arch}")
        model.load_state_dict(ck0.get("model", ck0), strict=True)   # backbone + norm + att + cls: all of it
        src = f"{src}+init:{os.path.basename(a.init_from)}@ep{ck0.get('epoch', '?')}"
        print(f"[init] whole model from {os.path.basename(a.init_from)} (res {ck0.get('res')}, epoch {ck0.get('epoch')}, "
              f"its gold {ck0.get('gold_auc')}); fresh optimizer/schedule", flush=True)
        del ck0
    if a.freeze_blocks > 0:
        # Backbones name their layer groups differently: ViT-style models use blocks.N., while
        # CoAtNet, MaxViT and ConvNeXt use stages.N. Matching only one of those froze nothing at
        # all on the other family, and did it silently. Try both, and always say what happened.
        pref = tuple([f"blocks.{b}." for b in range(a.freeze_blocks)]
                     + [f"stages.{b}." for b in range(a.freeze_blocks)])
        frozen = 0
        for nm, p in model.backbone.named_parameters():
            if nm.startswith(pref):
                p.requires_grad = False
                frozen += p.numel()
        n_t = sum(1 for p in model.backbone.parameters() if not p.requires_grad)
        if frozen:
            print(f"[raptor] froze the first {a.freeze_blocks} layer group(s): "
                  f"{n_t} tensors, {frozen/1e6:.2f}M parameters", flush=True)
        else:
            print(f"[raptor] WARNING: --freeze_blocks {a.freeze_blocks} matched no parameters on "
                  f"{a.arch}; every parameter is still being trained", flush=True)

    head_params = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.") and p.requires_grad]
    bb_params = [p for n_, p in model.named_parameters() if n_.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": a.bb_lr},
                             {"params": head_params, "lr": a.head_lr}], weight_decay=a.wd)
    steps = max((len(tl) // a.accum) * a.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.bb_lr, a.head_lr],
                                                total_steps=steps, pct_start=a.pct_start)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    # ---- per-cell loss (only when CELL): weights, robust losses, small-loss truncation, cell log ----
    tbuf = torch.full((max(1, a.trunc_buf), len(LAB)), float("nan"), device=dev) if CELL else None; tptr = [0, 0]   # [write ptr, rows filled]
    LOG = a.cell_log or a.trunc_frac > 0
    clog_kl = np.full((a.epochs, len(train_ids), len(LAB)), np.nan, np.float32) if LOG else None
    clog_drop = np.zeros((a.epochs, len(train_ids), len(LAB)), np.uint8) if LOG else None

    def cell_loss(z, y, w, frac):
        """z logits (B,12) fp32, y soft targets (NaN already 0), w cell weights. Mean over ALL B*12 cells, so with
        w = 1, loss bce and no truncation it equals BCEWithLogitsLoss(pos_weight=pw)."""
        if a.loss == "bce":
            L = F.binary_cross_entropy_with_logits(z, y, pos_weight=pw.expand_as(z), reduction="none")
        elif a.loss == "boot":
            yt = a.boot_beta * y + (1 - a.boot_beta) * torch.sigmoid(z).detach()
            L = F.binary_cross_entropy_with_logits(z, yt, pos_weight=pw.expand_as(z), reduction="none")
        elif a.loss == "gce":   # (1 - p^q)/q per side, in log space; q -> 0 is CE, q = 1 is MAE
            q = a.gce_q
            L = (pw * y * (1 - torch.exp(q * F.logsigmoid(z))) + (1 - y) * (1 - torch.exp(q * F.logsigmoid(-z)))) / q
        else:                   # sce: alpha*CE + beta*reverse CE, log 0 clipped at -4 (Wang et al. 2019)
            p = torch.sigmoid(z); A = math.exp(-4.0)
            rce = -(p * torch.log(y.clamp_min(A)) + (1 - p) * torch.log((1 - y).clamp_min(A)))
            L = a.sce_alpha * F.binary_cross_entropy_with_logits(z, y, pos_weight=pw.expand_as(z), reduction="none") + a.sce_beta * rce
        kl = drop = None
        if LOG or frac > 0:     # rank by KL(y||p) = unweighted BCE - H(y): 0 when p = y, so hedged soft cells are not "hard"
            with torch.no_grad():
                kl = (F.binary_cross_entropy_with_logits(z, y, reduction="none")
                      + torch.xlogy(y, y) + torch.xlogy(1 - y, 1 - y)).clamp_min(0)
                if a.trunc_frac > 0:
                    b = kl.masked_fill(w <= 0, float("nan")); n = b.shape[0]
                    idx = (torch.arange(n, device=z.device) + tptr[0]) % tbuf.shape[0]
                    tbuf[idx] = b; tptr[0] = (tptr[0] + n) % tbuf.shape[0]; tptr[1] = min(tptr[1] + n, tbuf.shape[0])
                    if frac > 0 and tptr[1] >= tbuf.shape[0]:   # threshold only once the history is full
                        thr = torch.nan_to_num(torch.nanquantile(tbuf, 1 - frac, dim=0), nan=float("inf"))
                        drop = (kl > thr) & (w > 0)
                        w = w * (~drop)
        return (L * w).sum() / L.numel(), kl, drop

    @torch.no_grad()
    def evaluate():
        model.eval(); P = []; Y = []
        for x, y, *pp in vl:
            with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                o = torch.sigmoid(model(x.to(dev), _P(pp)).float())
            P.append(o.cpu().numpy()); Y.append(y.numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                aucs[name] = float(roc_auc_score(Y[:, j], P[:, j]))
        return float(np.mean(list(aucs.values()))), aucs, P, Y

    best = 0.0; best_state = None; best_P = None; t0 = time.time(); hist = []
    TOPK = max(1, a.topk); topk = []; ep_start = 0
    if a.resume and os.path.exists(a.resume):
        R_ = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(R_["model"]); opt.load_state_dict(R_["opt"]); sched.load_state_dict(R_["sched"])
        if R_.get("scaler") is not None and scaler.is_enabled(): scaler.load_state_dict(R_["scaler"])
        best, best_state, best_P, hist, topk = R_["best"], R_["best_state"], R_["best_P"], R_["hist"], R_["topk"]
        ep_start = int(R_["epoch"]) + 1; t0 -= float(R_.get("elapsed", 0.0))
        if CELL and "clog" in R_:
            if LOG: clog_kl, clog_drop = R_["clog"]
            tbuf.copy_(R_["tbuf"][0].to(dev)); tptr[:] = R_["tbuf"][1]
        print(f"[resume] from {os.path.basename(a.resume)}: continuing at epoch {ep_start}/{a.epochs} (best {best:.4f})", flush=True)
        del R_
    if a.init_from and ep_start == 0:
        au0 = evaluate()[0]
        print(f"[init] GOLD macro-AUC of the stage-1 weights before fine-tuning: {au0:.4f} (not a candidate for best/topk)", flush=True)
    for ep in range(ep_start, a.epochs):
        model.train(); tot = 0.0; tot_t = None
        opt.zero_grad(); n_acc = 0
        _te = time.time(); _ts = time.time(); _step = 0
        for x, y, *pp in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                logits = model(x, _P(pp))
                if not CELL:
                    loss = lossf(logits.float(), y)
                else:
                    nL = len(LAB); rows = y[:, 2 * nL].long().cpu().numpy()
                    frac = a.trunc_frac * (min(1.0, (ep + _step / max(1, len(tl))) / a.trunc_ramp) if a.trunc_ramp > 0 else 1.0)
                    loss, kl, drop = cell_loss(logits.float(), y[:, :nL], y[:, nL:2 * nL], frac)
                    if LOG:
                        clog_kl[ep, rows] = kl.float().cpu().numpy()
                        if drop is not None: clog_drop[ep, rows] = drop.cpu().numpy()
            scaler.scale(loss / a.accum).backward(); n_acc += 1
            if XLA: tot_t = loss.detach() if tot_t is None else tot_t + loss.detach()   # no per-step host sync on XLA
            else: tot += loss.item()
            if XLA and n_acc < a.accum: XLA.mark_step()
            if n_acc == a.accum:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                scaler.step(opt); scaler.update()
                if sched.last_epoch + 1 < steps: sched.step()
                opt.zero_grad(); n_acc = 0
                if XLA: XLA.mark_step()
            if XLA and _step < 12:
                XLA.wait_device_ops(); print(f"  [xla] step {_step} {time.time()-_ts:.2f}s", flush=True); _ts = time.time()
            _step += 1
        if XLA:
            XLA.wait_device_ops(); print(f"  [xla] train loop {time.time()-_te:.0f}s for {_step} micro-steps", flush=True); _te = time.time()
        if XLA and tot_t is not None: tot = float(tot_t.item())
        if LOG:
            _k = clog_kl[ep]; _sup = np.isfinite(_k) & (Wm > 0)                  # cells seen this epoch with weight > 0
            print(f"  [cells] ep{ep} mean KL (weighted cells) {_k[_sup].mean():.4f} | dropped {int(clog_drop[ep].sum())}/"
                  f"{int(_sup.sum())} ({clog_drop[ep].sum() / max(1, _sup.sum()):.1%}) per finding {clog_drop[ep].sum(0).tolist()}", flush=True)
        au, aucs, P, Y = evaluate()
        if XLA: print(f"  [xla] gold eval {time.time()-_te:.0f}s", flush=True); _te = time.time()
        hist.append({"ep": ep, "loss": tot / len(tl), "gold_auc": au})
        if sl is not None:
            model.eval(); Ps = []; Ys = []
            with torch.no_grad():
                for x, y, *pp in sl:
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Ps.append(torch.sigmoid(model(x.to(dev), _P(pp)).float()).cpu().numpy())
                    Ys.append(y.numpy())
            Ps = np.concatenate(Ps); Ys = np.concatenate(Ys); sub = []
            for j in range(len(LAB)):
                kk = (sub_mask[:, j] > 0) & np.isfinite(Ys[:, j]); yb = (Ys[kk, j] >= 0.5).astype(int)
                if len(set(yb)) > 1: sub.append(float(roc_auc_score(yb, Ps[kk, j])))
            so = float(np.mean(sub)) if sub else float("nan"); hist[-1]["oof_sub_auc"] = so
            print(f"ep{ep} OOF{len(Ys)} masked macro-AUC {so:.4f}", flush=True)
            if XLA: print(f"  [xla] oof-sub eval {time.time()-_te:.0f}s", flush=True)
        if XLA and ep == ep_start:
            import torch_xla.debug.metrics as _met; print(_met.short_metrics_report(), flush=True)
        if au > best:
            best = au; best_P = P
            best_state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                          "arch": a.arch, "lab": LAB, "epoch": ep}
        if au >= best and best_state is not None:
            _tmp = os.path.join(HERE, f"raptor_ft_{a.tag}.pt.tmp")
            torch.save(best_state, _tmp)
            os.replace(_tmp, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                     pred=best_P, truth=Y, ids=np.array(gold_ids))
            json.dump({"tag": a.tag, "src": src, "best_gold_auc": best,
                       "aucs": best_state["aucs"], "hist": hist, "res": a.res,
                       "epochs": a.epochs, "bb_lr": a.bb_lr, "head_lr": a.head_lr,
                       "n_train": len(train_ids), "n_gold": len(gold_ids),
                       "roi": a.roi, "roi_mode": a.roi_mode,
                       "partial": True, "epochs_done": ep + 1},
                      open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
            print(f"  [ckpt] best-so-far saved at ep{ep} ({best:.4f})", flush=True)
        sel = hist[-1].get("oof_sub_auc", float("nan"))                 # selection score: masked OOF subset if scored, else gold
        sel = sel if sel == sel else au
        if len(topk) < TOPK or sel > min(t.get("sel", t["gold_auc"]) for t in topk):
            topk.append({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "gold_auc": au, "aucs": aucs, "src": src, "res": a.res,
                         "arch": a.arch, "lab": LAB, "epoch": ep, "P": P, "sel": sel})
            topk.sort(key=lambda t: -t.get("sel", t["gold_auc"]))
            del topk[TOPK:]
        print(f"ep{ep} loss {tot/len(tl):.3f} | GOLD macro-AUC {au:.4f} (best {best:.4f}) | topk by {'OOF-subset' if sl is not None else 'gold'}: {[int(t['epoch']) for t in topk]} | {time.time()-t0:.0f}s",
              flush=True)
        if a.time_budget_h > 0 and (time.time() - t0) > a.time_budget_h * 3600 and ep < a.epochs - 1:
            rp = os.path.join(HERE, f"raptor_resume_{a.tag}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler.is_enabled() else None, "best": best,
                        "best_state": best_state, "best_P": best_P, "hist": hist, "topk": topk, "epoch": ep,
                        "elapsed": time.time() - t0,
                        **({"clog": (clog_kl, clog_drop), "tbuf": (tbuf.cpu(), list(tptr))} if CELL else {})}, rp + ".tmp")
            os.replace(rp + ".tmp", rp)
            print(f"[resume] time budget reached after ep{ep}; state saved to {os.path.basename(rp)}; exiting", flush=True)
            sys.exit(0)
    _, aucs, _, Y = evaluate()
    print(f"\nDONE {src} | BEST GOLD macro-AUC {best:.4f}", flush=True)
    for k, v in (best_state["aucs"] if best_state else aucs).items():
        print(f"   {k:18s} {v:.3f}", flush=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(HERE, f"raptor_ft_{a.tag}.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}.npz"),
                 pred=best_P, truth=Y, ids=np.array(gold_ids))
        json.dump({"tag": a.tag, "src": src, "best_gold_auc": best, "aucs": best_state["aucs"],
                   "hist": hist, "res": a.res, "epochs": a.epochs, "bb_lr": a.bb_lr,
                   "head_lr": a.head_lr, "n_train": len(train_ids), "n_gold": len(gold_ids),
                   "roi": a.roi, "roi_mode": a.roi_mode, "partial": False, "epochs_done": a.epochs},
                  open(os.path.join(HERE, f"raptor_ft_{a.tag}.json"), "w"), indent=1)
        print(f"saved raptor_ft_{a.tag}.pt / raptor_gold_{a.tag}.npz / raptor_ft_{a.tag}.json", flush=True)
    if LOG:   # raptor_ft_ prefix so the Kaggle wrappers' output filter keeps it
        np.savez_compressed(os.path.join(HERE, f"raptor_ft_{a.tag}_cells.npz"), ids=np.array(train_ids), kl=clog_kl,
                            drop=clog_drop, y=np.stack([tlabels[u][:len(LAB)] for u in train_ids]),
                            w=np.stack([cw[u] for u in train_ids]), lab=np.array(LAB))
        print(f"wrote raptor_ft_{a.tag}_cells.npz (per-epoch KL(y||p) + drop flags, {len(train_ids)} studies)", flush=True)

    # ---- CV OOF: predict the held-out fold at the best (gold-selected) weights ----
    if a.folds > 0 and oof_ids and best_state is not None:
        model.load_state_dict(best_state["model"]); model.eval()
        ods = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False, norm=a.norm, **_roi_kw)
        ol = DataLoader(ods, batch_size=max(2, a.bs // 2), shuffle=False, num_workers=a.workers,
                        collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
        Po = []
        with torch.no_grad():
            for x, y, *pp in ol:
                with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                    o = torch.sigmoid(model(x.to(dev), _P(pp)).float())
                Po.append(o.cpu().numpy())
        Po = np.concatenate(Po)
        Yo = np.stack([labels[u] for u in oof_ids]).astype(np.float32)
        np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_fold{a.fold}.npz"),
                 pred=Po, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
        print(f"[cv] wrote raptor_oof_{a.tag}_fold{a.fold}.npz ({len(oof_ids)} studies; "
              f"truth = soft labels)", flush=True)

        # --- top-K epoch OOF (new) ---
        # The epoch-ensemble used to be judged on gold only; that gate is too small to
        # resolve the move. Re-run the held-out fold at each retained epoch instead.
        oof_by_ep = {best_state["epoch"]: Po}
        for t in topk:
            e = int(t["epoch"])
            if e in oof_by_ep:
                continue
            model.load_state_dict(t["model"]); model.eval()
            Pe = []
            with torch.no_grad():
                for x, y, *pp in ol:
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Pe.append(torch.sigmoid(model(x.to(dev), _P(pp)).float()).cpu().numpy())
            Pe = np.concatenate(Pe); oof_by_ep[e] = Pe
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_ep{e}_fold{a.fold}.npz"),
                     pred=Pe, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote top-K epoch OOF ep{e}", flush=True)
        if len(oof_by_ep) > 1:
            Pens = np.mean(list(oof_by_ep.values()), axis=0)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_epens_fold{a.fold}.npz"),
                     pred=Pens, truth=Yo, ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote epoch-ensemble OOF over epochs {sorted(oof_by_ep)}", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    # --- weight-averaged checkpoint (new) ---
    if len(topk) > 1:
        import copy
        sds = [t["model"] for t in topk]
        avg = {}
        for k in sds[0]:
            v0 = sds[0][k]
            if v0.is_floating_point():
                avg[k] = sum(sd[k].double() for sd in sds).div(len(sds)).to(v0.dtype)
            else:
                avg[k] = v0.clone()          # e.g. num_batches_tracked
        swa_state = {"model": avg, "gold_auc": None, "aucs": {}, "src": src, "res": a.res,
                     "arch": a.arch, "lab": LAB, "epoch": [int(t["epoch"]) for t in topk],
                     "swa_over": [int(t["epoch"]) for t in topk]}
        model.load_state_dict(avg); model.eval()
        au_swa, aucs_swa, P_swa, Y_swa = evaluate()
        swa_state["gold_auc"] = au_swa; swa_state["aucs"] = aucs_swa
        torch.save(swa_state, os.path.join(HERE, f"raptor_ft_{a.tag}_swa.pt"))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_swa.npz"),
                 pred=P_swa, truth=Y_swa, ids=np.array(gold_ids))
        print(f"SWA over epochs {[int(t['epoch']) for t in topk]} | gold {au_swa:.4f} "
              f"(best-epoch {best:.4f}) {'BETTER' if au_swa > best else 'no gain'}", flush=True)
        if a.folds > 0 and oof_ids:
            ods2 = StudyWindows(HERE, oof_ids, id2row, labels, a.res, a.k_eval, train=False,
                                norm=a.norm, **_roi_kw)
            ol2 = DataLoader(ods2, batch_size=max(2, a.bs // 2), shuffle=False,
                             num_workers=a.workers, collate_fn=collate, persistent_workers=False, worker_init_fn=WIF)
            Ps = []
            with torch.no_grad():
                for x, y, *pp in ol2:
                    with torch.autocast(DEVT, dtype=AMP_DTYPE, enabled=AMP_ON):
                        Ps.append(torch.sigmoid(model(x.to(dev), _P(pp)).float()).cpu().numpy())
            Ps = np.concatenate(Ps)
            np.savez(os.path.join(HERE, f"raptor_oof_{a.tag}_swa_fold{a.fold}.npz"),
                     pred=Ps, truth=np.stack([labels[u] for u in oof_ids]).astype(np.float32),
                     ids=np.array(oof_ids), fold=a.fold, nfolds=a.folds)
            print(f"[cv] wrote SWA OOF", flush=True)
        model.load_state_dict(best_state["model"]); model.eval()

    if len(topk) > 1:
        eps = [t["epoch"] for t in topk]
        for rank, t in enumerate(topk):
            if rank == 0:
                continue
            P_t = t.pop("P")
            np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_ep{t['epoch']}.npz"),
                     pred=P_t, truth=Y, ids=np.array(gold_ids))
            t["P"] = P_t
        Pens = np.mean([t["P"] for t in topk], axis=0)
        ens_aucs = {}
        for j, name in enumerate(LAB):
            if len(set(Y[:, j].astype(int))) > 1:
                ens_aucs[name] = float(roc_auc_score(Y[:, j], Pens[:, j]))
        ens = float(np.mean(list(ens_aucs.values())))
        np.savez(os.path.join(HERE, f"raptor_gold_{a.tag}_epens.npz"),
                 pred=Pens, truth=Y, ids=np.array(gold_ids))
        print(f"TOPK epochs {eps} | best {best:.4f} | epoch-ensemble {ens:.4f} "
              f"({'BETTER' if ens > best else 'no gain'})", flush=True)


if __name__ == "__main__":
    main()