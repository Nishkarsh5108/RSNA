"""Build compact_infer_x.py: heliosli's v4c compact worker (sha256 8401812c, the worker R6 and R7 pin) plus extra single
models scored on a second render of the SAME decoded study (mission 2026-10-21, final-max v2).

Why: in final-max v1, s6fall ran as its own stage after v27. That stage decodes every DICOM again and places the
SKM-TEA anchors again (~47 s of CPU per study on Kaggle), so on ~1,300 hidden studies it needs about as long as R6's stage.
It may only use the time v27 leaves before 8.35 h (v27 alone runs 6.5-8 h), and s6f_solo treats running out of time as a hard
error. So it very likely never reached the blend, which matches v1 scoring exactly v27's 0.958.
Here the R6 worker renders each study a second time at the extra models' size (px_cache.StudySlices from the same native
render, geometry, side decision and anchors; render is ~2.6 s per study) and scores them on it:
  - windows and meta exactly as s6fall's own worker builds them (compact_infer 34f433fe: the trainer's window_rows in eval
    mode, 6 meta columns, no extent columns), with the window list of the extra models' own trainer (--x-trainer);
  - the plain pass of that worker: fp16 autocast (bf16 when --amp says so), fp32 rescue of a non-finite output;
  - one csv per extra model, x_<stem>.csv (StudyInstanceUID + 12 probabilities, %.9g), x_receipt.json, x_logits.json.
Nothing on the R6 path changes: its predictions.csv, stackers, flip TTA, compile check and receipt are as before (checked on
gold_58 by xpass_gold_check.py).
One known difference from s6fall's own stage: a study the side classifier flips is flipped for the extra models too.
Usage: python -I src/emergency/build_compact_infer_x.py  -> src/emergency/compact_infer_x.py
"""
import hashlib
from pathlib import Path

SRC = Path("C:/Users/nishk/AppData/Local/Temp/claude/d--Coding-kaggle-RSNA/6465e451-4b73-4bc6-82a9-e00b7040f17c/scratchpad/src_r6/compact_infer.py")
SRC_SHA = "8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d"
OUT = Path("D:/Coding/kaggle/RSNA/src/emergency/compact_infer_x.py")

PATCHES = [
    ("docstring", '"""', '''"""[x] compact_infer_x.py = heliosli's v4c worker (sha256 8401812c) + --x-models: extra single models scored on a second
render of the same decoded study (see the [x] marks; everything else is unchanged).
''', 1),
    ("args",
     '    ap.add_argument("--models2", default=None, help="second model group (another architecture), run on the same windows")\n',
     '    ap.add_argument("--x-models", default=None, help="[x] comma-separated single-model state dicts scored on a second render of "\n'
     '                    "the same decoded study (--x-size rows, --x-crop centre crop)")\n'
     '    ap.add_argument("--x-trainer", default=None, help="[x] the compact_train.py the extra models were trained with")\n'
     '    ap.add_argument("--x-backbone", default="convnext_small")\n'
     '    ap.add_argument("--x-size", type=int, default=320)\n'
     '    ap.add_argument("--x-crop", type=int, default=288)\n'
     '    ap.add_argument("--x-eval-stride", type=int, default=1)\n'
     '    ap.add_argument("--x-chunk", type=int, default=192)\n'
     '    ap.add_argument("--models2", default=None, help="second model group (another architecture), run on the same windows")\n', None),
    ("init",
     "        self.source_directory = True     # False: read DICOMs from the chunked competition archive under <data>/download\n",
     "        self.source_directory = True     # False: read DICOMs from the chunked competition archive under <data>/download\n"
     "        self.x_cfg = None                # [x] {size, eval_stride, trainer}: the extra models' render (set by main)\n", None),
    ("setup",
     "        self.window_rows = compact_train.window_rows          # the trainer's own window list (eval mode)\n",
     "        self.window_rows = compact_train.window_rows          # the trainer's own window list (eval mode)\n"
     "        if self.x_cfg is not None:                            # [x] the extra models' own trainer (window list) and renderer\n"
     "            import importlib.util as _ilu\n"
     "            _spec = _ilu.spec_from_file_location(\"compact_train_x\", self.x_cfg[\"trainer\"])\n"
     "            _ctx = _ilu.module_from_spec(_spec)\n"
     "            _spec.loader.exec_module(_ctx)\n"
     "            self.window_rows_x = _ctx.window_rows\n"
     "            self.builder_x = px_cache.StudySlices(self.tmp, [], int(self.x_cfg[\"size\"]), self.views)\n", None),
    ("x_item method",
     "    def __getitem__(self, i):\n        uid = self.uids[i]\n",
     "    def _x_item(self, uid, work):\n"
     "        \"\"\"[x] the extra models' windows: px_cache at the x size from this study's native render, geometry and anchors, the\n"
     "        window list of their own trainer, 6 meta columns (their worker, compact_infer 34f433fe, builds them so). Never raises.\"\"\"\n"
     "        t = time.perf_counter()\n"
     "        try:\n"
     "            self.builder_x.cache = str(work)\n"
     "            if self.bank is not None:\n"
     "                self.builder_x.anchors = str(work / \"anchors\")\n"
     "            item = self.builder_x.build(uid)\n"
     "            px, meta = item[\"px\"].numpy(), item[\"meta\"].numpy()\n"
     "            blocks = [{\"n_slices\": b[\"n_slices\"], \"views\": [(0, v, st) for v, st in b[\"views\"]]} for b in item[\"blocks\"]]\n"
     "            cand = self.window_rows_x(blocks, lambda f, r: meta[r], \"eval\", self.slices, int(self.x_cfg[\"eval_stride\"]),\n"
     "                                      phys_mm=self.phys_mm)\n"
     "            if not cand:\n"
     "                raise ValueError(\"no usable window\")\n"
     "            rows = np.array(cand, np.int64)\n"
     "            img = np.stack([px[rows[:, 2 + d]] for d in range(self.slices)], 1)\n"
     "            m = meta[rows[:, 1]]\n"
     "            meta6 = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1).astype(np.float32)\n"
     "            return {\"images_x\": torch.from_numpy(img), \"meta_x\": torch.from_numpy(meta6), \"x_error\": None,\n"
     "                    \"x_seconds\": round(time.perf_counter() - t, 3)}\n"
     "        except Exception as exc:\n"
     "            return {\"images_x\": None, \"meta_x\": None, \"x_error\": repr(exc)[:300], \"x_seconds\": round(time.perf_counter() - t, 3)}\n"
     "\n"
     "    def __getitem__(self, i):\n        uid = self.uids[i]\n", None),
    ("getitem",
     '            out = {"uid": uid, "error": None, "images": torch.from_numpy(img), "meta": torch.from_numpy(meta6), "anchor": anchor_note,\n'
     '                   "side": side_note}\n',
     '            out = {"uid": uid, "error": None, "images": torch.from_numpy(img), "meta": torch.from_numpy(meta6), "anchor": anchor_note,\n'
     '                   "side": side_note}\n'
     '            if self.x_cfg is not None:                        # [x] before the finally below removes the work folder\n'
     '                out.update(self._x_item(uid, work))\n', None),
    ("load x models",
     "    files, models, n_spatial = load_group(a.models, a.arm, a.backbone, a.crop)\n    n_syn1 = N_SYN[0]\n",
     "    files, models, n_spatial = load_group(a.models, a.arm, a.backbone, a.crop)\n    n_syn1 = N_SYN[0]\n"
     "    X_CFG, XMODELS, XFH, XW, XLOG = None, [], {}, {}, {}           # [x]\n"
     "    XREC = {\"scored\": 0, \"failed\": {}, \"gpu_s\": 0.0, \"render_s\": 0.0, \"windows\": 0}\n"
     "    if a.x_models:\n"
     "        import importlib.util as _ilu\n"
     "        if not a.x_trainer:\n"
     "            raise SystemExit(\"--x-models needs --x-trainer\")\n"
     "        _spec = _ilu.spec_from_file_location(\"compact_train_x\", a.x_trainer)\n"
     "        CTX = _ilu.module_from_spec(_spec)\n"
     "        _spec.loader.exec_module(CTX)\n"
     "        if list(CTX.LABELS) != list(LABELS):\n"
     "            raise SystemExit(\"--x-trainer: label order differs from the main trainer's\")\n"
     "        if not 0 < a.x_crop <= a.x_size:\n"
     "            raise SystemExit(\"--x-crop must be in (0, --x-size]\")\n"
     "        for p in a.x_models.split(\",\"):\n"
     "            sd = torch.load(p, map_location=\"cpu\", weights_only=True)\n"
     "            if any(k.split(\".\")[-1] in (\"ext_u\", \"ext_view\", \"ext_gamma\", \"syn_beta\") for k in sd):\n"
     "                raise SystemExit(\"--x-models: extent / syn-decomp models are not supported (%s)\" % p)\n"
     "            kw = {\"spatial\": True} if any(k == \"sp_q\" or k.endswith(\".sp_q\") for k in sd) else {}\n"
     "            mx = CTX.Net(a.x_backbone, None, freeze_blocks=a.freeze_blocks, crop=a.x_crop, state_spec=None, in_chans=a.slices, **kw)\n"
     "            mx.load_state_dict(sd, strict=True)\n"
     "            name = Path(p).stem\n"
     "            if any(name == n_ for n_, _, _ in XMODELS):\n"
     "                raise SystemExit(\"--x-models: two files named %s\" % name)\n"
     "            XMODELS.append((name, p, mx.eval().requires_grad_(False).to(dev).to(memory_format=torch.channels_last)))\n"
     "            XFH[name] = open(out / (\"x_%s.csv\" % name), \"w\", encoding=\"utf-8\", newline=\"\")\n"
     "            XW[name] = csv.writer(XFH[name])\n"
     "            XW[name].writerow([UIDK] + LABELS)\n"
     "        X_CFG = {\"size\": int(a.x_size), \"eval_stride\": int(a.x_eval_stride), \"trainer\": str(Path(a.x_trainer).resolve())}\n"
     "        print(\"X_READY\", json.dumps({\"models\": [n_ for n_, _, _ in XMODELS], \"trainer\": X_CFG[\"trainer\"],\n"
     "                                     \"trainer_sha256\": hashlib.sha256(Path(a.x_trainer).read_bytes()).hexdigest(),\n"
     "                                     \"size\": a.x_size, \"crop\": a.x_crop, \"eval_stride\": a.x_eval_stride, \"chunk\": a.x_chunk}),\n"
     "              flush=True)\n", None),
    ("score_x",
     "    def handle(item):\n        uid = item[\"uid\"]\n",
     "    def score_x(item):\n"
     "        \"\"\"[x] every extra model on the study's own x windows: the plain pass of its worker (autocast, fp32 rescue).\"\"\"\n"
     "        uid = item[\"uid\"]\n"
     "        XREC[\"render_s\"] += float(item.get(\"x_seconds\") or 0.0)\n"
     "        if item.get(\"images_x\") is None or not len(item[\"images_x\"]):\n"
     "            XREC[\"failed\"][uid] = item.get(\"x_error\") or \"no x windows\"\n"
     "            return\n"
     "        t = time.perf_counter()\n"
     "        try:\n"
     "            with torch.inference_mode():\n"
     "                o = (a.x_size - a.x_crop) // 2\n"
     "                img = item[\"images_x\"].to(dev, non_blocking=True)[None, :, :, o:o + a.x_crop, o:o + a.x_crop]\n"
     "                meta = item[\"meta_x\"].to(dev)[None]\n"
     "                mask = torch.ones((1, img.shape[1]), dtype=torch.bool, device=dev)\n"
     "                for name, _, mx in XMODELS:\n"
     "                    with torch.autocast(\"cuda\", dtype=torch.float16 if amp == \"fp16\" else torch.bfloat16):\n"
     "                        z = mx(img, meta, mask, chunk=a.x_chunk).float()\n"
     "                    if not torch.isfinite(z).all():\n"
     "                        z = mx(img, meta, mask, chunk=a.x_chunk).float()\n"
     "                    if not torch.isfinite(z).all():\n"
     "                        raise ValueError(\"non-finite logits from \" + name)\n"
     "                    pz = torch.sigmoid(z[0].double()).cpu().numpy()\n"
     "                    XW[name].writerow([uid] + [\"%.9g\" % float(v) for v in pz])\n"
     "                    XFH[name].flush()\n"
     "                    XLOG.setdefault(name, {})[uid] = z[0].cpu().tolist()\n"
     "            XREC[\"scored\"] += 1\n"
     "            XREC[\"windows\"] += int(img.shape[1])\n"
     "        except Exception as exc:\n"
     "            XREC[\"failed\"][uid] = \"gpu: \" + repr(exc)[:200]\n"
     "        XREC[\"gpu_s\"] += time.perf_counter() - t\n"
     "\n"
     "    def handle(item):\n        uid = item[\"uid\"]\n", None),
    ("handle hook",
     "        wr.writerow([uid] + [\"%.9g\" % float(x) for x in p])\n        fh.flush()\n",
     "        wr.writerow([uid] + [\"%.9g\" % float(x) for x in p])\n        fh.flush()\n"
     "        if XMODELS:                                   # [x] after the main row: the extra models on their own render\n"
     "            score_x(item)\n", None),
    ("dataset cfg",
     "                      fluid=STK is not None or (SST is not None and SST[\"needs_volume\"]))\n",
     "                      fluid=STK is not None or (SST is not None and SST[\"needs_volume\"]))\n"
     "        ds.x_cfg = X_CFG                                  # [x] pickled into the spawned loader workers with the dataset\n", None),
    ("x receipt",
     "    for u in uids[pos:]:\n        fallback[u] = \"not reached before the deadline\"\n    fh.close()\n",
     "    for u in uids[pos:]:\n        fallback[u] = \"not reached before the deadline\"\n    fh.close()\n"
     "    if XMODELS:                                       # [x]\n"
     "        for f_ in XFH.values():\n"
     "            f_.close()\n"
     "        (out / \"x_logits.json\").write_text(json.dumps(XLOG), encoding=\"utf-8\")\n"
     "        (out / \"x_receipt.json\").write_text(json.dumps({\n"
     "            \"models\": {n_: {\"file\": str(p_), \"sha256\": hashlib.sha256(Path(p_).read_bytes()).hexdigest()} for n_, p_, _ in XMODELS},\n"
     "            \"trainer\": X_CFG[\"trainer\"], \"trainer_sha256\": hashlib.sha256(Path(a.x_trainer).read_bytes()).hexdigest(),\n"
     "            \"size\": a.x_size, \"crop\": a.x_crop, \"eval_stride\": a.x_eval_stride, \"chunk\": a.x_chunk, \"amp\": amp,\n"
     "            \"studies\": len(uids), \"scored\": XREC[\"scored\"], \"failed\": XREC[\"failed\"], \"windows\": XREC[\"windows\"],\n"
     "            \"gpu_s\": round(XREC[\"gpu_s\"], 1), \"render_s\": round(XREC[\"render_s\"], 1), \"timed_out\": timed_out}, indent=1),\n"
     "            encoding=\"utf-8\")\n"
     "        print(\"X_DONE\", json.dumps({\"scored\": XREC[\"scored\"], \"failed\": len(XREC[\"failed\"]), \"gpu_s\": round(XREC[\"gpu_s\"], 1),\n"
     "                                    \"render_s\": round(XREC[\"render_s\"], 1)}), flush=True)\n", None),
]


def main():
    s = SRC.read_bytes()
    assert hashlib.sha256(s).hexdigest() == SRC_SHA, "v4c worker differs from the one R6 / R7 pin"
    t = s.decode("utf-8")
    for what, old, new, count in PATCHES:
        n = t.count(old)
        if count is None and n != 1:
            raise SystemExit(f"{what}: expected 1 match, found {n}")
        t = t.replace(old, new, 1)
    compile(t, str(OUT), "exec")
    assert "claude" not in t.lower() and "anthropic" not in t.lower()
    OUT.write_text(t, encoding="utf-8", newline="\n")
    print("wrote", OUT, hashlib.sha256(OUT.read_bytes()).hexdigest()[:16], len(t), "chars")


if __name__ == "__main__":
    main()
