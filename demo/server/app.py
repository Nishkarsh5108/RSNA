"""Local server for the knee MRI workstation. Binds to 127.0.0.1 only; uploaded studies live in a local cache folder and
in memory, and are deleted on purge or shutdown. Run:

    python app.py            (serves the built site from ../web/dist and the API on http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import struct
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import ingest

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("KNEE_CACHE", "D:/rsna_work/demo_cache"))
DIST = HERE.parent / "web" / "dist"
RAW = Path(os.environ.get("KNEE_RAW", "D:/Coding/kaggle/RSNA/data/Raw_dicom"))
HOST, PORT = "127.0.0.1", int(os.environ.get("PORT", 8000))

# Local sample studies (read from this machine's dataset folder; nothing is bundled with the site). The reference
# studies are 4 of the 58 image-read studies that no model was trained on, chosen by their reference labels only (one
# acute trauma, one degenerative, one ligament injury, one single finding) before any prediction was seen. Their list
# lives outside the repository (KNEE_SAMPLES: [{"uid": ..., "label": ...}]), like the data it points to.
SAMPLES_FILE = Path(os.environ.get("KNEE_SAMPLES", "D:/rsna_work/demo_samples.json"))


def reference_samples() -> list[tuple[str, str]]:
    try:
        return [(d["uid"], d["label"]) for d in json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))]
    except Exception:
        return []


STUDIES: dict[str, ingest.Study] = {}
RESULTS: dict[str, dict] = {}
ENGINE = None
ENGINE_ERROR = None


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _clean(o):
    """NaN / inf -> null (JSON has no NaN), recursively."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.floating):
        return _clean(float(o))
    return o


def dumps(o) -> str:
    return json.dumps(_clean(o), default=jsonable, allow_nan=False)


def packed(header: dict, *arrays: np.ndarray) -> Response:
    """uint32 header length (little endian) + JSON header + the arrays' bytes, back to back."""
    head = dumps(header).encode()
    pad = (-(4 + len(head))) % 4                    # keep the payload 4-byte aligned for Float32Array views
    head += b" " * pad
    body = struct.pack("<I", len(head)) + head + b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
    return Response(body, media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


def load_engine():
    global ENGINE, ENGINE_ERROR
    try:
        import engine
        ENGINE = engine.Engine()
    except Exception as ex:                          # the site still serves; the workstation reports the error
        ENGINE_ERROR = f"{type(ex).__name__}: {ex}"


@asynccontextmanager
async def lifespan(app):
    shutil.rmtree(CACHE, ignore_errors=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=load_engine, daemon=True).start()
    yield
    shutil.rmtree(CACHE, ignore_errors=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


def reference_labels():
    try:
        t = pd.read_csv(RAW / "train.csv").set_index("StudyInstanceUID")
        import engine
        return {uid: t.loc[uid, engine.LABELS].astype(float).tolist() for uid, _ in reference_samples() if uid in t.index}
    except Exception:
        return {}


def samples():
    out = []
    test_root = RAW / "test_series"
    if test_root.is_dir():
        for i, p in enumerate(sorted(test_root.iterdir())):
            if p.is_dir():
                out.append(dict(id=f"test-{i}", label=f"Public test study {chr(65 + i)}", kind="unlabelled",
                                series=sum(1 for q in p.iterdir() if q.is_dir()), path=str(p)))
    for i, (uid, label) in enumerate(reference_samples()):
        p = RAW / "train_series" / uid
        if p.is_dir():
            out.append(dict(id=f"ref-{i}", label=label, kind="reference", uid=uid,
                            series=sum(1 for q in p.iterdir() if q.is_dir()), path=str(p)))
    return out


# ---------------------------------------------------------------------------------------------- routes
@app.get("/api/status")
def status():
    base = dict(host=f"{HOST}:{PORT}", cache=str(CACHE), studies=len(STUDIES))
    if ENGINE is None:
        return dict(base, ready=False, error=ENGINE_ERROR)
    return dict(base, ready=True, **ENGINE.status())


@app.get("/api/samples")
def list_samples():
    return [{k: v for k, v in s.items() if k != "path"} for s in samples()]


def _register(study: ingest.Study):
    STUDIES[study.sid] = study
    while len(STUDIES) > 4:                          # keep the cache small: drop the oldest study
        old = min(STUDIES.values(), key=lambda s: s.created)
        purge(old.sid)
    return study.manifest()


@app.post("/api/samples/{sample_id}")
async def load_sample(sample_id: str):
    s = next((s for s in samples() if s["id"] == sample_id), None)
    if s is None:
        raise HTTPException(404, "no such sample")
    sid = uuid.uuid4().hex[:12]
    try:
        study = await run_in_threadpool(ingest.build_study, ingest.iter_dir(Path(s["path"])), CACHE, sid, "sample")
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    study.source = s["label"]
    if s["kind"] == "reference":
        study.reference = dict(labels=reference_labels().get(study.uid),
                               note="Image-read reference labels shipped with the dataset (one of 58 studies excluded from all training).")
    return _register(study)


@app.post("/api/studies")
async def upload(request: Request):
    """Body: a zip archive of the study's DICOM files (the browser packs dropped folders into one)."""
    data = await request.body()
    if not data:
        raise HTTPException(400, "empty upload")
    sid = uuid.uuid4().hex[:12]

    def work():
        try:
            files = list(ingest.iter_zip(data))
        except Exception:
            files = [("upload.dcm", data)]       # a single DICOM file
        return ingest.build_study(files, CACHE, sid, "upload")

    try:
        study = await run_in_threadpool(work)
    except ValueError as ex:
        raise HTTPException(422, str(ex))
    return _register(study)


def _study(sid) -> ingest.Study:
    if sid not in STUDIES:
        raise HTTPException(404, "study not loaded (purged or server restarted)")
    return STUDIES[sid]


def _series(sid, uid):
    s = next((s for s in _study(sid).series if s.uid == uid), None)
    if s is None:
        raise HTTPException(404, "no such series")
    return s


@app.get("/api/studies/{sid}")
def manifest(sid: str):
    return _study(sid).manifest()


@app.patch("/api/studies/{sid}/series/{uid}")
async def edit_series(sid: str, uid: str, request: Request):
    s = _series(sid, uid)
    body = await request.json()
    if "fatSuppression" in body:
        s.fat_suppression = bool(body["fatSuppression"])
        s.fs_reason = "set by the user"
    if "include" in body:
        s.include = bool(body["include"])
    RESULTS.pop(sid, None)
    return s.meta()


@app.get("/api/studies/{sid}/series/{uid}/volume")
def volume(sid: str, uid: str):
    s = _series(sid, uid)
    return packed(dict(shape=list(s.display.shape), dtype="uint8"), s.display)


@app.get("/api/studies/{sid}/series/{uid}/model-input")
def model_input(sid: str, uid: str):
    r = RESULTS.get(sid, {}).get("arrays", {}).get(uid)
    if r is None:
        raise HTTPException(404, "run inference first")
    return packed(dict(shape=list(r["inputs"].shape), dtype="uint8"), r["inputs"])


@app.get("/api/studies/{sid}/series/{uid}/explain")
def explain(sid: str, uid: str):
    r = RESULTS.get(sid, {}).get("arrays", {}).get(uid)
    if r is None:
        raise HTTPException(404, "run inference first")
    names = [k for k in ("share", "contrib", "boxWeight", "lesion") if k in r]
    arrs = [r[k].astype(np.float32) for k in names]
    return packed(dict(shape=list(arrs[0].shape), dtype="float32", arrays=names,
                       units=dict(share="fraction of the label's pooled feature (sums to 1 over the series)",
                                  contrib="logit units, averaged over the models (sums to the mean logit minus bias)",
                                  boxWeight="share of the ROI box before attention",
                                  lesion="lesion head: probability that the cell is covered by a lesion of the finding (not used for the probability)")), *arrs)


@app.delete("/api/studies/{sid}")
def purge(sid: str):
    s = STUDIES.pop(sid, None)
    RESULTS.pop(sid, None)
    if s is not None:
        shutil.rmtree(s.root, ignore_errors=True)
    return dict(purged=s is not None)


@app.post("/api/studies/{sid}/infer")
async def infer(sid: str):
    """Runs the pipeline and streams every step as newline-delimited JSON (one event per line), ending with the result."""
    study = _study(sid)
    if ENGINE is None:
        raise HTTPException(503, ENGINE_ERROR or "models are still loading")
    if ENGINE.lock.locked():
        raise HTTPException(409, "the GPU is busy with another study")
    rows = [dict(StudyInstanceUID=study.uid, SeriesInstanceUID=s.uid, Anatomical_Plane=s.plane,
                 Fat_Suppression=int(s.fat_suppression), Fluid_Sensitive=int(s.fat_suppression)) for s in study.series if s.include]
    if not rows:
        raise HTTPException(422, "no series selected")
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def emit(ev):
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    def work():
        try:
            res = ENGINE.run(study.root, pd.DataFrame(rows), emit)
            RESULTS[sid] = res
            emit(dict(kind="result", summary=res["summary"]))
        except Exception as ex:
            import traceback
            traceback.print_exc()
            emit(dict(kind="error", error=f"{type(ex).__name__}: {ex}"))
        emit(None)

    threading.Thread(target=work, daemon=True).start()

    async def stream():
        while (ev := await queue.get()) is not None:
            yield dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------------------------------------- site
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def site(path: str):
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        # the shell is never cached, so a rebuilt site (new chunk hashes) is picked up on the next load
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
