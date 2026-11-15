"""Local GPU inference for the workstation: the same three passes as src/postcomp/infer_bp.py (the pipeline that scored
0.958 private on Kaggle), run one series at a time so every intermediate result can be shown as it happens.

  1. side classifier (ResNet-18d) on the coronal / axial series        -> p(right knee) per series, study side
  2. ROI detector (ConvNeXt-T + 3D fusion), flip test-time augmentation -> 12 boxes per series (both passes kept)
  3. box-pooling classifiers (ConvNeXt-T), every checkpoint found      -> 12 logits per series and model

The model code is the public modules' own (side.py, detector.py, classifier.py, imported unchanged); predictions come
from the classifiers' unmodified forward(). The explanation is read from the same forward: the encoder's feature map is
captured and the forward's pooling weights are recomputed from it. Because the head is linear over a weighted mean of
feature cells, logit_k = sum_cells share_kc * (h_k . f_c) + b_k exactly; the per-cell terms are the attribution, and
their sum is checked against the logit the forward returned (reported as attribution_error).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "D:/rsna_work/hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

MODULES_DIR = Path(os.environ.get("KNEE_MODULES", "D:/rsna_work/bp_modules"))
sys.path.insert(0, str(MODULES_DIR))
import classifier as C  # noqa: E402
import detector as D  # noqa: E402
import side as S  # noqa: E402

LABELS = C.LABELS
PUB = Path(os.environ.get("KNEE_PUBLIC", "D:/rsna_work/public_code/tascj"))
SIDE_WEIGHTS = Path(os.environ.get("KNEE_SIDE_WEIGHTS", PUB / "out_rsna-2026-knee-side-classifier/side_ema.pth"))
DET_WEIGHTS = Path(os.environ.get("KNEE_DET_WEIGHTS", PUB / "out_rsna-2026-knee-roi-detector/detector_ema.pth"))
CLS_DIR = Path(os.environ.get("KNEE_CLS_DIR", "D:/rsna_work/kaggle_runs/knee-bp-weights-ds"))

# What each checkpoint is (decision log, reports/POSTCOMP_DECISION_LOG.md). Unknown files are still loaded.
CHECKPOINTS = {
    "a0_f0": dict(labels="1st-place labels", fold=0, note="published recipe, reproduced"),
    "a1_f0": dict(labels="7th-place labels", fold=0, note="label swap"),
    "a2_f0": dict(labels="fused labels", fold=0, note="mean of both label sets"),
    "c1_f1": dict(labels="fused labels", fold=1, note="fold model"),
    "c2_f2": dict(labels="fused labels", fold=2, note="fold model"),
    "c3_f3": dict(labels="fused labels", fold=3, note="fold model"),
}


def _load(model, path, dev):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(dev).eval()


def _np(t):
    return t.detach().float().cpu().numpy()


class Engine:
    def __init__(self, device: str | None = None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = dict(device_type=self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda")
        self.lock = threading.Lock()
        t0 = time.time()
        self.side = _load(S.SideClassifier(pretrained=False), SIDE_WEIGHTS, self.dev).to(memory_format=torch.channels_last)
        self.det = _load(D.Detector(pretrained=False), DET_WEIGHTS, self.dev)
        for part in (self.det.encoder, self.det.projections):
            part.to(memory_format=torch.channels_last)
        self.cls = []
        for p in sorted(CLS_DIR.glob("*.pth")):
            m = _load(C.BoxPoolClassifier(pretrained=False), p, self.dev).to(memory_format=torch.channels_last)
            self.cls.append((p.stem, m))
        if not self.cls:
            raise RuntimeError(f"no classifier checkpoints in {CLS_DIR}")
        self.warmup()
        self.load_seconds = time.time() - t0
        self.weights_mb = sum(sum(t.numel() * t.element_size() for t in m.state_dict().values())
                              for m in [self.side, self.det] + [m for _, m in self.cls]) / 2**20

    @torch.no_grad()
    def warmup(self):
        """One dummy forward per network at typical shapes, so the first real study does not pay for CUDA / cuDNN
        initialisation (about 15 s on this laptop otherwise). Outputs are discarded."""
        if self.dev.type != "cuda":
            return
        dev, n = self.dev, 32
        ss, depth = torch.zeros(n, dtype=torch.long, device=dev), torch.tensor([n], device=dev)
        with torch.autocast(**self.amp):
            self.side(torch.zeros(n, S.SIDE_PX, S.SIDE_PX, device=dev), ss, depth)
            image, meta, depths = torch.zeros(1, 1, n, D.DET_PX, D.DET_PX, device=dev), torch.zeros(1, 10, device=dev), torch.tensor([n], device=dev)
            meta[0, [0, 4, 6]] = 1
            self.det(image, meta, depths)
            self.det(*D.flip_left_right(image, meta, depths)[:2], depths)
            boxes = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], device=dev).repeat(1, len(LABELS), 1) * 2 - torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=dev)
            theta = C.crop_theta(boxes)
            inp = torch.zeros(n, C.CROP_PX, C.CROP_PX, device=dev)
            for _, m in self.cls:
                m(inp, ss, depth, theta, C.pooling_boxes(boxes))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ status
    def status(self) -> dict:
        gpu = None
        if self.dev.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            gpu = dict(name=torch.cuda.get_device_name(0), totalGB=total / 2**30, freeGB=free / 2**30,
                       allocatedGB=torch.cuda.memory_allocated() / 2**30, capability=".".join(map(str, torch.cuda.get_device_capability(0))))
        params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        return dict(device=str(self.dev), gpu=gpu, precision="fp16 autocast (weights fp32)", torch=torch.__version__,
                    weightsMB=round(self.weights_mb, 1), loadSeconds=round(self.load_seconds, 2), busy=self.lock.locked(),
                    models=dict(
                        side=dict(name="ResNet-18d side classifier", params=params(self.side), input=f"{S.SIDE_PX} px, <= {S.MAX_DEPTH} slices"),
                        detector=dict(name="ConvNeXt-T ROI detector + 3D fusion", params=params(self.det), input=f"{D.DET_PX} px, native depth <= {D.DET_MAX_DEPTH}"),
                        classifiers=[dict(id=k, params=params(m), **CHECKPOINTS.get(k, dict(labels="unknown", fold=None, note=""))) for k, m in self.cls],
                        classifierInput=f"{C.CROP_MM:.0f} mm crop at {C.CROP_PX} px, <= {C.MAX_DEPTH} slices; frame {C.FOV_MM:.0f} mm at {C.FRAME_PX} px"),
                    labels=LABELS)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def run(self, root: Path, series: pd.DataFrame, emit) -> dict:
        """series: StudyInstanceUID, SeriesInstanceUID, Anatomical_Plane, Fat_Suppression, Fluid_Sensitive (one study).
        emit(event: dict) is called from this thread for every step. Returns the full result (arrays included)."""
        with self.lock:
            # The three passes each read every series through the modules' own read_dicom_series (identical code in
            # side.py, detector.py and classifier.py). Within one run the result is memoized, so each series is decoded
            # once instead of three times; the function itself is unchanged, so the arrays are identical.
            memo, originals = {}, {m: m.read_dicom_series for m in (S, D, C)}

            def memoized(directory, _read=C.read_dicom_series):
                key = str(directory)
                if key not in memo:
                    memo[key] = _read(directory)
                return memo[key]

            for m in originals:
                m.read_dicom_series = memoized
            try:
                return self._run(root, series.reset_index(drop=True), emit)
            finally:
                for m, f in originals.items():
                    m.read_dicom_series = f
                memo.clear()

    def _run(self, root, series, emit):
        dev, amp = self.dev, self.amp
        for m in (S, D, C):
            m.SERIES_ROOT = Path(root)
        T0 = time.perf_counter()
        cuda = dev.type == "cuda"
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        def now():
            if cuda:
                torch.cuda.synchronize()
            return time.perf_counter() - T0

        def send(stage, kind, **payload):
            ev = dict(stage=stage, kind=kind, t=round(now() * 1000, 1), **payload)
            if cuda:
                ev["vramGB"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            emit(ev)

        study = series.StudyInstanceUID.iloc[0]
        uids = list(series.SeriesInstanceUID)
        failures = []
        send("pipeline", "start", series=uids, models=[k for k, _ in self.cls])

        # 1. knee side ------------------------------------------------------------------------------------------
        send("side", "start")
        ca = series[series.Anatomical_Plane.isin(["Coronal", "Axial"])].reset_index(drop=True)
        side_ds = S.SideDataset(ca, {})
        p_right = []
        for i in range(len(ca)):
            uid = ca.SeriesInstanceUID[i]
            t = now()
            try:
                x = S.collate([side_ds[i]])
                with torch.autocast(**amp):
                    logit = self.side(S.side_volumes(x, dev), x["slice_series"].to(dev), x["series_depth"].to(dev)).float()
                p = float(torch.sigmoid(logit)[0])
                p_right.append(p)
                send("side", "series", series=uid, pRight=p, slices=int(x["series_depth"][0]), ms=round((now() - t) * 1000, 1))
            except Exception as ex:
                failures.append(dict(stage="side", series=uid, error=f"{type(ex).__name__}: {ex}"))
                send("side", "error", series=uid, error=failures[-1]["error"])
        knee = ("right" if np.mean(p_right) >= 0.5 else "left") if p_right else "right"
        side_map = {study: knee}
        send("side", "done", knee=knee, pRight=float(np.mean(p_right)) if p_right else None, byDefault=not p_right)

        # 2. ROI boxes ------------------------------------------------------------------------------------------
        send("detector", "start")
        det_ds = D.DetectorDataset(series, side_map)
        boxes, det_detail = {}, {}
        for i, uid in enumerate(uids):
            t = now()
            try:
                g = D.collate_volumes([det_ds[i]])[0]
                image, meta, depths = D.detector_volumes(g, dev), g["meta"].to(dev), g["depths"].to(dev)
                # == detector.detect(), with both test-time passes kept for display
                with torch.autocast("cuda", torch.float16, enabled=cuda):
                    direct = self.det(image, meta, depths)[1].float()
                    flipped, flipped_meta, sagittal = D.flip_left_right(image, meta, depths)
                    mirrored = self.det(flipped, flipped_meta, depths)[1].float()
                axis = torch.where(sagittal, 2, 0)[:, None, None]
                lo, hi = mirrored[..., :3].clone(), mirrored[..., 3:].clone()
                on = torch.arange(3, device=dev) == axis
                lo, hi = torch.where(on, 1 - mirrored[..., 3:], lo), torch.where(on, 1 - mirrored[..., :3], hi)
                unmirrored = torch.cat([lo, hi], -1)
                mean = (direct + unmirrored) / 2
                boxes[uid] = _np(mean)[0]
                det_detail[uid] = dict(direct=_np(direct)[0], mirrored=_np(unmirrored)[0], depth=int(depths[0]),
                                       bucket=int(g["bucket"]), meta=_np(meta)[0].round(4).tolist())
                send("detector", "series", series=uid, boxes=boxes[uid].round(5).tolist(),
                     direct=det_detail[uid]["direct"].round(5).tolist(), mirrored=det_detail[uid]["mirrored"].round(5).tolist(),
                     depth=int(depths[0]), bucket=int(g["bucket"]), input=[int(depths[0]), D.DET_PX, D.DET_PX],
                     ms=round((now() - t) * 1000, 1))
            except Exception as ex:
                failures.append(dict(stage="detector", series=uid, error=f"{type(ex).__name__}: {ex}"))
                send("detector", "error", series=uid, error=failures[-1]["error"])
        plane_of = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
        med = {pl: np.median(np.stack([b for u, b in boxes.items() if plane_of[u] == pl]), 0)
               for pl in series.Anatomical_Plane.unique() if any(plane_of[u] == pl for u in boxes)}
        whole = np.tile(np.float32([0, 0, 0, 1, 1, 1]), (len(LABELS), 1))
        fallback = [u for u in uids if u not in boxes]
        for u in fallback:
            boxes[u] = med.get(plane_of[u], whole)
        roi = pd.DataFrame([(u, LABELS[k], *b[k]) for u, b in boxes.items() for k in range(len(LABELS))],
                           columns=["SeriesInstanceUID", "label", "x0", "y0", "z0", "x1", "y1", "z1"])
        send("detector", "done", fallback=fallback)
        if cuda:
            torch.cuda.empty_cache()

        # 3. box-pooling classifiers -------------------------------------------------------------------------------
        send("classifier", "start", models=[k for k, _ in self.cls])
        ds = C.SeriesDataset(series, {}, side_map, roi)
        per_series = {}
        for i, uid in enumerate(uids):
            t = now()
            try:
                item = ds[i]
                x = C.collate([item])
                native = int(item["pixels"].shape[0])
                rows, cols = item["pixels"].shape[1:]
                image = C.frame_volumes(x, dev)
                ss, depth, bx = (x[k].to(dev, non_blocking=True) for k in ("slice_series", "series_depth", "boxes"))
                theta = C.crop_theta(bx)
                inp = C.sample_slices(image, ss, theta, C.CROP_PX)
                geo = self._geometry(x, theta, native, rows, cols, bx)
                send("classifier", "frame", series=uid, mirror=bool(x["mirror"][0]), depth=int(depth[0]), native=native,
                     crop=geo["crop"], framePx=C.FRAME_PX, cropPx=C.CROP_PX, ms=round((now() - t) * 1000, 1))
                rec = dict(uid=uid, models={}, inputs=inp.clamp(0, 255).round().to(torch.uint8).cpu().numpy(), **geo)
                share_sum = contrib_sum = box_sum = lesion_sum = None
                for k, m in self.cls:
                    tm = now()
                    logits, share, contrib, boxw, lesion, err = self._explained(m, inp, ss, depth, theta, C.pooling_boxes(bx))
                    if bool(x["mirror"][0]):                 # maps back to the native orientation
                        share, contrib, boxw, lesion = share.flip(-1), contrib.flip(-1), boxw.flip(-1), lesion.flip(-1)
                    probs = torch.sigmoid(logits)
                    rec["models"][k] = dict(logits=_np(logits).tolist(), probs=_np(probs).tolist(), attributionError=err)
                    share_sum = share if share_sum is None else share_sum + share
                    contrib_sum = contrib if contrib_sum is None else contrib_sum + contrib
                    box_sum = boxw if box_sum is None else box_sum + boxw
                    lesion_sum = lesion if lesion_sum is None else lesion_sum + lesion
                    send("classifier", "model", series=uid, model=k, probs=[round(v, 5) for v in _np(probs).tolist()],
                         logits=[round(v, 4) for v in _np(logits).tolist()], slices=int(depth[0]), cells=list(share.shape[-2:]),
                         channels=int(m.head.in_features), attributionError=err, ms=round((now() - tm) * 1000, 1))
                M = len(self.cls)
                rec["share"] = _np(share_sum / M).astype(np.float16)        # [D, 12, 8, 8]
                rec["contrib"] = _np(contrib_sum / M).astype(np.float16)    # [D, 12, 8, 8], logit units
                rec["boxWeight"] = _np(box_sum / M).astype(np.float16)      # geometric box share before attention
                rec["lesion"] = _np(lesion_sum / M).astype(np.float16)      # lesion head: P(cell covered by a lesion)
                rec["probs"] = np.mean([v["probs"] for v in rec["models"].values()], 0).tolist()
                per_series[uid] = rec
                send("classifier", "series", series=uid, probs=[round(v, 5) for v in rec["probs"]], ms=round((now() - t) * 1000, 1))
            except Exception as ex:
                failures.append(dict(stage="classifier", series=uid, error=f"{type(ex).__name__}: {ex}"))
                send("classifier", "error", series=uid, error=failures[-1]["error"])
            if cuda:
                torch.cuda.empty_cache()

        # 4. study = mean over its series per model, then the mean over the models (as infer_bp.py) ----------------------
        model_ids = [k for k, _ in self.cls]
        per_model = {k: np.mean([r["models"][k]["probs"] for r in per_series.values()], 0) for k in model_ids} if per_series else {}
        study_probs = np.mean(list(per_model.values()), 0) if per_model else np.full(len(LABELS), np.nan)
        result = dict(
            study=study, knee=knee, pRight=p_right, failures=failures, models=model_ids, labels=LABELS,
            probs=study_probs.tolist(), perModel={k: v.tolist() for k, v in per_model.items()},
            spread=dict(min=np.min(list(per_model.values()), 0).tolist(), max=np.max(list(per_model.values()), 0).tolist()) if per_model else None,
            series={u: dict(boxes=boxes[u].tolist(), detector={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in det_detail.get(u, {}).items()},
                            fallbackBox=u in fallback) for u in uids},
            seconds=round(now(), 3),
            peakVramGB=round(torch.cuda.max_memory_allocated() / 2**30, 3) if cuda else None,
            maxAttributionError=max((v["attributionError"] for r in per_series.values() for v in r["models"].values()), default=None),
        )
        for u, r in per_series.items():
            result["series"][u].update({k: r[k] for k in ("probs", "crop", "slicePos", "zRange", "mirror", "depth", "native")},
                                       models=r["models"])
        send("ensemble", "done", probs=[round(v, 5) for v in result["probs"]], perModel={k: [round(x, 5) for x in v] for k, v in result["perModel"].items()},
             seconds=result["seconds"], peakVramGB=result["peakVramGB"], failures=failures, maxAttributionError=result["maxAttributionError"])
        return dict(summary=result, arrays=per_series)

    # ------------------------------------------------------------------ helpers
    def _explained(self, m, inp, ss, depth, theta, boxes):
        """Logits from the model's own forward; attribution from its captured encoder output. Also the lesion head
        (a 1 x 1 convolution trained, during training only, to predict each finding's per-slice lesion boxes as cell
        coverage): it reads the same features but does not feed the logit, so it is an independent localisation."""
        captured = {}
        enc = m.encoder
        original = enc.forward_features

        def capture(t):
            out = original(t)
            captured["padded"] = out
            return out

        enc.forward_features = capture
        try:
            with torch.autocast(**self.amp):
                logits = m(inp, ss, depth, theta, boxes).float()[0]
        finally:
            del enc.forward_features                         # back to the class method
        padded = captured["padded"]
        n = len(inp)
        maps = padded[:n]
        _, c, h, w = maps.shape
        # == BoxPoolClassifier.forward: box weights, then in-box attention
        gx, gy = C.cell_points(theta, ss, h, w)
        start = torch.cumsum(depth, 0) - depth
        z = (torch.arange(n, device=inp.device) - start[ss] + 0.5) / depth[ss]
        b = boxes.float()[ss][..., None, None]
        inside = (gx[:, None] >= b[:, :, 0]) & (gx[:, None] <= b[:, :, 3]) & (gy[:, None] >= b[:, :, 1]) & (gy[:, None] <= b[:, :, 4])
        inside &= ((z[:, None] >= b[:, :, 2, 0, 0]) & (z[:, None] <= b[:, :, 5, 0, 0]))[..., None, None]
        box_weight = F.avg_pool2d(inside.float(), 4)                # [n, 12, h, w]
        weight = box_weight
        if m.attention is not None:
            with torch.autocast(**self.amp):
                score = m.attention(padded)[:n].float().masked_fill(box_weight <= 0, float("-inf"))
            top = score.amax((0, 2, 3))                              # one series: its max of a inside the box
            top = torch.where(torch.isfinite(top), top, torch.zeros_like(top))
            weight = box_weight * torch.exp(C.ATT_SHARP * (score - top[None, :, None, None]))
        f = maps.float()
        totals = weight.sum((0, 2, 3))                               # [12]
        uniform = torch.full_like(weight, 1.0 / (n * h * w))         # an empty box pools the series' mean feature
        share = torch.where((totals > 0)[None, :, None, None], weight / totals.clamp(min=1e-12)[None, :, None, None], uniform)
        cell_logit = torch.einsum("nchw,kc->nkhw", f, m.head.weight.float())
        contrib = share * cell_logit
        recon = contrib.sum((0, 2, 3)) + m.head.bias.float()
        err = float((recon - logits).abs().max())
        bsum = box_weight.sum((0, 2, 3))
        box_share = torch.where((bsum > 0)[None, :, None, None], box_weight / bsum.clamp(min=1e-12)[None, :, None, None], uniform)
        if m.lesion_head is not None:
            with torch.autocast(**self.amp):
                lesion = torch.sigmoid(m.lesion_head(padded)[:n].float())
        else:
            lesion = torch.zeros_like(share)
        return logits, share, contrib, box_share, lesion, err

    @staticmethod
    def _geometry(x, theta, native, rows, cols, bx):
        """Where the classifier looked, in the native image: crop rectangle (normalized columns / rows), the native
        slice position of every classifier slice, and per label the native slice range inside its box's z extent."""
        top, left, height, width = x["placement"][0].tolist()
        mirror = bool(x["mirror"][0])
        s, cx, cy = float(theta[0, 0, 0]), float(theta[0, 0, 2]), float(theta[0, 1, 2])
        fx0, fx1 = (-cx - s, -cx + s) if mirror else (cx - s, cx + s)

        def nx(v):
            return ((v + 1) / 2 * C.FRAME_PX - left) / width

        def ny(v):
            return ((v + 1) / 2 * C.FRAME_PX - top) / height

        crop = [nx(fx0), ny(cy - s), nx(fx1), ny(cy + s)]
        d = min(native, C.MAX_DEPTH)
        slice_pos = (np.linspace(0, native - 1, d) if native > d else np.arange(d, dtype=float)).tolist()
        z = (np.arange(d) + 0.5) / d
        b = bx[0].float().cpu().numpy()
        zr = []
        for k in range(len(LABELS)):
            idx = np.nonzero((z >= b[k, 2]) & (z <= b[k, 5]))[0]
            zr.append([slice_pos[idx[0]], slice_pos[idx[-1]]] if len(idx) else None)
        return dict(crop=[round(v, 5) for v in crop], slicePos=[round(v, 3) for v in slice_pos], zRange=zr,
                    mirror=mirror, depth=d, native=native)
