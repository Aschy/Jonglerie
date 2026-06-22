#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Backend web de l'analyseur de jongle.
==============================================
FastAPI minimal : upload d'une vidéo -> analyse YOLOv8 en tâche de fond ->
métriques + score + vidéo annotée. UI servie depuis ../web/index.html.

Endpoints :
  GET  /                       page web (interface d'upload)
  POST /api/analyze            upload vidéo -> { job_id }
  GET  /api/job/{id}           état du job (queued|running|done|error) + résultat
  GET  /api/job/{id}/video     vidéo annotée (mp4)
  GET  /healthz                ping
"""
import os, sys, uuid, json, shutil, traceback, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- chemins projet -------------------------------------------------------- #
BASE   = Path(__file__).resolve().parent.parent       # racine du dépôt
SRC    = BASE / "src"
WEB    = BASE / "web"
MODEL  = os.environ.get("JONGLE_MODEL", str(BASE / "models" / "yolov8m.onnx"))
DATA   = Path(os.environ.get("JONGLE_DATA", str(BASE / "data")))
JOBS_D = DATA / "jobs"
JOBS_D.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
import analyzer_yolo as AY                              # noqa: E402

# OpenCV : utiliser tous les cœurs pour accélérer l'inférence DNN
try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

MAX_UPLOAD_MB = int(os.environ.get("JONGLE_MAX_MB", "150"))
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}

app = FastAPI(title="Jonglerie — Analyse de jongle")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# --- état des jobs en mémoire (single worker) ------------------------------ #
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)                # analyses sérialisées (CPU-bound)


def _set(job_id: str, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


def _run_analysis(job_id: str, video_path: Path, out_dir: Path):
    _set(job_id, status="running", pct=1, label="Initialisation")
    try:
        def cb(pct, label):
            _set(job_id, pct=int(pct), label=label)
        res = AY.analyze_yolo(str(video_path), str(out_dir), onnx=MODEL,
                              progress_cb=cb)
        _set(job_id, status="done", pct=100, label="Terminé",
             result=res, video_url=f"/api/job/{job_id}/video")
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", label="Erreur d'analyse", error=str(e))
    finally:
        try:
            video_path.unlink(missing_ok=True)          # libère l'upload brut
        except Exception:
            pass


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_present": Path(MODEL).exists()}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)):
    if not Path(MODEL).exists():
        raise HTTPException(503, "Modèle YOLO indisponible sur le serveur.")
    ext = Path(video.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Format non supporté ({ext or '?'}). "
                                 f"Utilise : {', '.join(sorted(ALLOWED_EXT))}")

    job_id = uuid.uuid4().hex[:12]
    out_dir = JOBS_D / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"input{ext}"

    # écriture en streaming + garde-fou taille
    size = 0
    limit = MAX_UPLOAD_MB * 1024 * 1024
    with open(raw, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close(); shutil.rmtree(out_dir, ignore_errors=True)
                raise HTTPException(413, f"Vidéo trop lourde (> {MAX_UPLOAD_MB} Mo).")
            f.write(chunk)

    _set(job_id, status="queued", pct=0, label="En file d'attente",
         filename=video.filename)
    POOL.submit(_run_analysis, job_id, raw, out_dir)
    return {"job_id": job_id}


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Job inconnu")
    out = {k: v for k, v in j.items() if k != "result"}
    if j.get("status") == "done":
        out["result"] = j["result"]
    return JSONResponse(out)


@app.get("/api/job/{job_id}/video")
def job_video(job_id: str):
    path = JOBS_D / job_id / "annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "Vidéo non disponible")
    return FileResponse(str(path), media_type="video/mp4",
                        filename=f"jonglerie_{job_id}.mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
