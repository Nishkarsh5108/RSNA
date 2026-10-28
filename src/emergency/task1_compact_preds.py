"""Task 1 (plan section 12): heliosli S6 and nartaa s6fall predictions on gold_58 + oof_300, run locally with the legs' own
code (heliosli's compact_infer.CStudies render path and compact_train.Net, the orthofoundation `training` package and
anatomy.pt). Nothing is trained; no label is read.

Per study and per fold model it saves:
  z_plain   [M, 12]   logits, plain pass (the quantity the Kaggle stage averages over folds)
  z_mirror  [M, 12]   logits of the mirrored windows, re-indexed by R6's permutation P (medial <-> lateral meniscus / OA);
                      MCL stays the plain logit (R6's v4c rule). NaN where not computed.
  tok       [M, 12, H] class tokens (Net.out_norm output, plain pass; the residual-adapter input of Task 3), float16
  sl, bl              report-state / binary logits (S6's state heads; plain pass), ext [M, 12] OA-extent area (extent models)
Mirror code: R6's compact_infer v4c (mirror_windows, mirror_perm), copied unchanged except LABELS passed in.
Which passes run: gold_58 -> every fold model plain + mirror; oof_300 -> every fold model plain, mirror only for the
study's own held-out S6 fold (fold = int(sha1(uid)[:8], 16) % 4). s6fall (one all-data model): plain + mirror everywhere.

Usage: python -I src/emergency/task1_compact_preds.py --leg s6|s6fall [--limit N] [--chunk 32] [--workers 3]
Writes output/emergency_research/task1/<leg>/<sha16>.npz (resumable) and <leg>_index.jsonl.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

R = "D:/Coding/kaggle/RSNA"
PW = "D:/rsna_work/public_weights"
SRC = "D:/rsna_work/public_code/s6_src"
CODE = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/cloud_code"
ANATOMY = f"{PW}/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt"
DATA = f"{R}/data/Raw_dicom"
LEGS = {
    "s6": dict(models=[f"{PW}/rsna-knee-compact-v1/E_lrprior_s0_f{k}.pt" for k in range(4)], size=384, crop=352,
               state_spec=f"{PW}/rsna-knee-compact-v1/state_spec.json", side_model=f"{PW}/rsna-knee-compact-v1/side_cls.pt"),
    "s6fall": dict(models=[f"{PW}/rsna-knee-s6fall-weights/E_prior_s42_fall.pt"], size=320, crop=288, state_spec=None,
                   side_model=None),
    # plan 13 Track D: our fine-tune of s6fall (s6ft_train.py EMA weights, same Net and keys), scored like s6fall
    "s6ft": dict(models=[os.environ.get("S6FT_WEIGHTS", "D:/rsna_work/s6ft_local/s6ft_ema.pt")], size=320, crop=288,
                 state_spec=None, side_model=None),
    "s6ft_v2": dict(models=[os.environ.get("S6FT_V2_WEIGHTS", "D:/rsna_work/s6ft_v2_local/s6ft_ema.pt")], size=320, crop=288,
                    state_spec=None, side_model=None),
    "s6ft_v3": dict(models=[os.environ.get("S6FT_V3_WEIGHTS", "D:/rsna_work/s6ft_v3_local/s6ft_specialist_ema.pt")], size=320, crop=288,
                    state_spec=None, side_model=None),
}
MIRROR_PAIRS = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))


def mirror_perm(labels):
    swap = {}
    for m_, l_ in MIRROR_PAIRS:
        swap[m_], swap[l_] = l_, m_
    return [labels.index(swap.get(l, l)) for l in labels]


def mirror_windows(img, meta):
    """R6 compact_infer v4c: coronal/axial windows flipped on the width axis; a sagittal window reversed along its slice
    channels and its rel_pos (meta column 2) negated; coronal locals change sides (view id 2 <-> 3, meta column 1)."""
    import torch
    assert meta.shape[-1] >= 8, "needs the plane column (meta column 7)"
    sag = meta[..., 7].round() == 0
    img = torch.where(sag[:, :, None, None, None], img.flip(2), img.flip(-1))
    meta = meta.clone()
    meta[..., 2] = torch.where(sag, -meta[..., 2], meta[..., 2])
    v = meta[..., 1]
    meta[..., 1] = torch.where(v == 2, torch.full_like(v, 3.), torch.where(v == 3, torch.full_like(v, 2.), v))
    return img, meta


def s6_fold(uid):
    return int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", required=True, choices=sorted(LEGS))
    ap.add_argument("--studies", default=f"{R}/output/emergency_research/task1/task1_studies.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-set", default=None, choices=[None, "gold_58", "oof_300"])
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tmp", default="D:/rsna_work/tmp_task1")
    ap.add_argument("--split", default="train", help="test: parity check on the 3 local public test studies")
    ap.add_argument("--cpu-workers", type=int, default=1, help="1: loader workers see no GPU (anchors/side model on CPU)")
    ap.add_argument("--out-tag", default=None, help="output folder name (default: the leg)")
    ap.add_argument("--mem-fraction", type=float, default=0.8, help="cap on this process's share of GPU memory")
    a = ap.parse_args()
    cfg = LEGS[a.leg]
    os.environ["PYTHONPATH"] = os.pathsep.join([SRC, CODE, os.environ.get("PYTHONPATH", "")])   # spawned loader workers
    sys.path[:0] = [SRC, CODE]
    import pandas as pd
    import torch
    import compact_infer as CI
    import compact_train as CT
    LABELS = CT.LABELS
    P = mirror_perm(LABELS)
    MCL = LABELS.index("MCL")
    out_dir = Path(f"{R}/output/emergency_research/task1/{a.out_tag or a.leg}")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = Path(f"{R}/output/emergency_research/task1/{a.out_tag or a.leg}_index.jsonl")
    st = pd.read_csv(a.studies, dtype={"StudyInstanceUID": str})
    if a.only_set:
        st = st[st["set"] == a.only_set]
    st = st[[not (out_dir / (hashlib.sha256(u.encode()).hexdigest()[:16] + ".npz")).exists() for u in st.StudyInstanceUID]]
    if a.limit:
        st = st.head(a.limit)
    uids = st.StudyInstanceUID.tolist()
    setof = dict(zip(st.StudyInstanceUID, st["set"]))
    print(f"{a.leg}: {len(uids)} studies to score", flush=True)
    if not uids:
        return
    dev = torch.device("cuda")
    spec = json.load(open(cfg["state_spec"], encoding="utf-8")) if cfg["state_spec"] else None
    models = []
    for p in cfg["models"]:
        sd = torch.load(p, map_location="cpu", weights_only=True)
        kw = {}
        if any(k == "sp_q" or k.endswith(".sp_q") for k in sd):
            kw["spatial"] = True
        if any(k.split(".")[-1] in ("ext_u", "ext_view", "ext_gamma") for k in sd):
            kw["extent"] = True
        m = CT.Net("convnext_small", None, freeze_blocks=16, crop=cfg["crop"], state_spec=spec, in_chans=3, **kw)
        m.load_state_dict(sd, strict=True)
        m = m.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)
        grab = {}
        m.out_norm.register_forward_hook(lambda mod, inp, out, g=grab: g.__setitem__("tok", out.detach()))
        models.append((m, grab, kw))
    print("models", [(Path(p).name, kw) for p, (_, _, kw) in zip(cfg["models"], models)], flush=True)
    off = (cfg["size"] - cfg["crop"]) // 2
    if a.workers and a.cpu_workers:
        # loader workers are spawned with this environment: they run the anchor V-Net and the side classifier on the CPU
        # (CStudies._setup picks "cuda" only when CUDA is visible), so the 6 GB card holds only the fold models.
        # "-1", not "": on Windows an empty value deletes the variable and the workers would see the GPU again.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # never let the caching allocator grow past physical VRAM: on Windows the driver then pages to system RAM and a
    # study takes hours instead of seconds (seen 2026-10-20: 7,719 s for one 604-window study)
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    tmp = Path(a.tmp) / a.leg
    tmp.mkdir(parents=True, exist_ok=True)
    ds = CI.CStudies(DATA, a.split, uids, tmp, CODE, cfg["size"], 1, "v4", 3, ANATOMY, 3.5,
                     cfg["side_model"], 2.0, 0.0, 20.0, extent_meta=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers,
                                         prefetch_factor=1 if a.workers else None,
                                         multiprocessing_context="spawn" if a.workers else None,
                                         timeout=900 if a.workers else 0)

    @torch.inference_mode()
    def one_pass(m, grab, img, meta, chunk):
        mask = torch.ones((1, img.shape[1]), dtype=torch.bool, device=dev)
        with torch.autocast("cuda", dtype=torch.float16):
            z, sl, bl = m(img, meta, mask, chunk=chunk, return_aux=True)
        z = z.float()
        if not torch.isfinite(z).all():
            z, sl, bl = m(img.float(), meta, mask, chunk=chunk, return_aux=True)
        ext = getattr(m, "_ext_area", None)
        return (z[0].cpu().numpy(), grab["tok"][0].float().cpu().numpy(),
                None if sl is None else sl[0].float().cpu().numpy(), None if bl is None else bl[0].float().cpu().numpy(),
                None if ext is None else ext[0].float().cpu().numpy())

    t0 = time.time()
    with open(index, "a", encoding="utf-8") as ix:
        for n, item in enumerate(loader, 1):
            uid = item["uid"]
            rec = {"uid": uid, "set": setof[uid], "s6_fold": s6_fold(uid), "error": item["error"],
                   "seconds": item["seconds"], "anchor": item.get("anchor"), "side": item.get("side")}
            if item["error"] or not len(item["images"]):
                rec["error"] = rec["error"] or "no usable window"
                ix.write(json.dumps(rec) + "\n"); ix.flush()
                print("FAILED", uid[-12:], rec["error"][:120], flush=True)
                continue
            tg = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            img = item["images"].to(dev)[None, :, :, off:off + cfg["crop"], off:off + cfg["crop"]]
            meta = item["meta"].to(dev)[None]
            img_m, meta_m = mirror_windows(img, meta)
            M = len(models)
            chunk = a.chunk
            while True:
                try:
                    zp = np.full((M, 12), np.nan, np.float32); zm = np.full((M, 12), np.nan, np.float32)
                    toks, sls, bls, exts = [], [], [], []
                    for k, (m, grab, kw) in enumerate(models):
                        z, tok, sl, bl, ext = one_pass(m, grab, img, meta, chunk)
                        zp[k] = z
                        toks.append(tok.astype(np.float16))
                        sls.append(sl); bls.append(bl); exts.append(ext)
                        do_mirror = a.leg in ("s6fall", "s6ft") or setof[uid] == "gold_58" or k == s6_fold(uid)
                        if do_mirror:
                            z2 = one_pass(m, grab, img_m, meta_m, chunk)[0]
                            zt = 0.5 * (z + z2[P])
                            zt[MCL] = z[MCL]
                            zm[k] = zt
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk <= 8:
                        raise
                    chunk //= 2
                    print(f"OOM on {uid[-12:]} ({img.shape[1]} windows): retrying with chunk {chunk}", flush=True)
            n_win = int(img.shape[1])
            del img, meta, img_m, meta_m
            rec["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2 ** 20)
            rec["chunk"] = chunk
            torch.cuda.empty_cache()
            save = {"z_plain": zp, "z_mirror": zm, "tok": np.stack(toks), "n_windows": np.int32(n_win)}
            if sls[0] is not None:
                save["sl"] = np.stack(sls).astype(np.float32)
            if bls[0] is not None:
                save["bl"] = np.stack(bls).astype(np.float32)
            if exts[0] is not None:
                save["ext"] = np.stack(exts).astype(np.float32)
            np.savez_compressed(out_dir / (hashlib.sha256(uid.encode()).hexdigest()[:16] + ".npz"), uid=uid, **save)
            rec["gpu_s"] = round(time.perf_counter() - tg, 2)
            rec["n_windows"] = n_win
            ix.write(json.dumps(rec) + "\n"); ix.flush()
            if n <= 3 or n % 10 == 0:
                el = time.time() - t0
                print(f"{n}/{len(uids)} {setof[uid]} windows {n_win} decode {item['seconds'][0]:.1f}s render "
                      f"{item['seconds'][1]:.1f}s gpu {rec['gpu_s']:.1f}s peak {rec['peak_reserved_mb']} MB | "
                      f"{el / n:.1f} s/study", flush=True)
    print(f"done {len(uids)} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
