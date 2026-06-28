#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Backend web de l'analyseur de jongle (avec comptes utilisateurs).
==========================================================================
Auth login/mot de passe (token signe), sessions persistees en SQLite, analyse
calibree par utilisateur (taille + ballon connus). UI servie depuis ../web/index.html.

Endpoints :
  GET  /                       interface (login + app)
  POST /api/login              { username, password } -> { token, user }
  GET  /api/me                 (auth) profil utilisateur
  POST /api/analyze            (auth) upload video -> { job_id }
  GET  /api/job/{id}           etat du job + resultat (+ session_id si sauvee)
  GET  /api/job/{id}/video     video annotee (mp4)
  GET  /api/sessions           (auth) historique des sessions
  GET  /api/sessions/{id}      (auth) detail d'une session
  GET  /healthz                ping
"""
import os, sys, uuid, shutil, traceback, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

BASE  = Path(__file__).resolve().parent.parent
SRC   = BASE / "src"
WEB   = BASE / "web"
MODEL = os.environ.get("JONGLE_MODEL", str(BASE / "models" / "yolov8m.onnx"))
DATA  = Path(os.environ.get("JONGLE_DATA", str(BASE / "data")))
JOBS_D = DATA / "jobs"
JOBS_D.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # pour db/auth
import analyzer_yolo as AY                # noqa: E402
import db, auth, gamify                   # noqa: E402

try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

MAX_UPLOAD_MB = int(os.environ.get("JONGLE_MAX_MB", "150"))
ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}

app = FastAPI(title="Jonglerie — Analyse de jongle")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)


def _set(job_id, **kw):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {}).update(kw)


# --- auth ----------------------------------------------------------------- #
def current_user(authorization: str = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Authentification requise")
    uid = auth.verify_token(authorization.split(" ", 1)[1].strip())
    if not uid:
        raise HTTPException(401, "Session expirée, reconnecte-toi")
    u = db.get_user_public(uid)
    if not u:
        raise HTTPException(401, "Utilisateur inconnu")
    return u


# --- analyse (thread) ----------------------------------------------------- #
def _run_analysis(job_id, video_path: Path, out_dir: Path, user, filename):
    _set(job_id, status="running", pct=1, label="Initialisation")
    try:
        calib = dict(ball_real_cm=user["ball_diam_cm"], player_height_cm=user["height_cm"])
        def cb(pct, label):
            _set(job_id, pct=int(pct), label=label)
        res = AY.analyze_yolo(str(video_path), str(out_dir), onnx=MODEL,
                              progress_cb=cb, calib=calib,
                              player={"display_name": user["display_name"],
                                      "username": user["username"]})
        session_id = db.add_session(user["id"], res, filename=filename)
        _set(job_id, status="done", pct=100, label="Terminé", result=res,
             session_id=session_id, video_url=f"/api/job/{job_id}/video")
    except Exception as e:
        traceback.print_exc()
        _set(job_id, status="error", label="Erreur d'analyse", error=str(e))
    finally:
        try: video_path.unlink(missing_ok=True)
        except Exception: pass


# --- lifecycle ------------------------------------------------------------ #
@app.on_event("startup")
def _startup():
    db.init_db()
    print("[+] DB prête:", db.DB_PATH)
    if Path(MODEL).exists():
        try:
            import numpy as np
            from yolo_onnx import get_detector
            det = get_detector(MODEL, conf=0.10, imgsz=640)
            det.detect_ball(np.zeros((64, 64, 3), dtype="uint8"))
            print(f"[+] modèle YOLO préchauffé (backend={det.backend})")
        except Exception as e:
            print("[warn] préchauffage échoué:", e)
    else:
        print("[warn] modèle absent:", MODEL)


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/healthz")
def healthz():
    return {"ok": True, "model_present": Path(MODEL).exists()}


# --- auth endpoints ------------------------------------------------------- #
@app.post("/api/login")
async def login(payload: dict):
    u = db.authenticate((payload.get("username") or "").strip().lower(),
                        payload.get("password") or "")
    if not u:
        raise HTTPException(401, "Identifiant ou mot de passe incorrect")
    return {"token": auth.make_token(u["id"]), "user": u}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


# --- analyse -------------------------------------------------------------- #
@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...), user=Depends(current_user)):
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
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    with open(raw, "wb") as f:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > limit:
                f.close(); shutil.rmtree(out_dir, ignore_errors=True)
                raise HTTPException(413, f"Vidéo trop lourde (> {MAX_UPLOAD_MB} Mo).")
            f.write(chunk)
    _set(job_id, status="queued", pct=0, label="En file d'attente", filename=video.filename)
    POOL.submit(_run_analysis, job_id, raw, out_dir, user, video.filename)
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
    return FileResponse(str(path), media_type="video/mp4", filename=f"jonglerie_{job_id}.mp4")


# --- historique ----------------------------------------------------------- #
@app.get("/api/sessions")
def sessions(user=Depends(current_user)):
    return db.list_sessions(user["id"])


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int, user=Depends(current_user)):
    s = db.get_session(user["id"], session_id)
    if not s:
        raise HTTPException(404, "Session introuvable")
    return s


@app.get("/api/profile")
def profile(user=Depends(current_user)):
    """Carte joueur facon FC : OVR, attributs, niveau/XP, palier, deblocages."""
    return gamify.build_profile(user, db.list_sessions_full(user["id"]))


# avatars + assets statiques (cartes joueur)
ASSETS = WEB / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
app.mount("/assets", StaticFiles(directory=str(ASSETS)), name="assets")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
