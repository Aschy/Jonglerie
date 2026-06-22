#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyzer_yolo.py — Pipeline d'analyse de jongle, detection YOLOv8 (version amelioree).
=====================================================================================
Remplace la detection couleur de `analyzer.py` par un YOLOv8 pre-entraine COCO
(classe 32 "sports ball"), execute via OpenCV DNN — AUCUNE dependance torch/ultralytics.

Ameliorations cles (cf. docs/METHODOLOGY.md) :
  1. Detection YOLO sur TOUTES les frames, seuil bas (YOLO ne confond pas le ballon
     avec un maillot de meme couleur -> seuil bas sans risque).
  2. Comptage par INVERSION DE VITESSE verticale (plus sensible que les simples pics).
  3. RATTRAPAGE des jongles caches dans les trous de detection (ballon flou) via un
     second signal independant : le mouvement de jambe (kick_motion).

Obtenir le modele ONNX (une fois) :
    pip install ultralytics
    yolo export model=yolov8m.pt format=onnx imgsz=640      # -> yolov8m.onnx

Usage :
    python analyzer_yolo.py ma_video.mp4 out/ yolov8m.onnx
"""
import cv2, numpy as np, json, os, sys
import analyzer as A
from yolo_onnx import YOLOv8ONNX
import kick_motion as KM
from scipy.signal import find_peaks


def track_with_yolo(frames, onnx, conf=0.10, gapfill=True):
    """Detecte le ballon par YOLO sur chaque frame -> trajectoire dense interpolee."""
    det = YOLOv8ONNX(onnx, conf=conf, imgsz=640)
    N = len(frames)
    xs = np.full(N, np.nan); ys = np.full(N, np.nan); meas = np.zeros(N, bool)
    for i, fr in enumerate(frames):
        b = det.detect_ball(fr)
        if b:
            xs[i], ys[i], meas[i] = b[0], b[1], True

    if gapfill:
        # Rattrapage gate : dans les trous, on cherche un candidat YOLO a tres bas
        # seuil PRES de la position predite (evite tout faux positif lointain).
        idx = np.arange(N); g = ~np.isnan(xs)
        px = np.interp(idx, idx[g], xs[g]); py = np.interp(idx, idx[g], ys[g])
        for i in range(N):
            if meas[i]:
                continue
            best, bd = None, 40**2
            for (x, y, r, sc) in det.detect_ball_candidates(frames[i], conf=0.05):
                d = (x - px[i])**2 + (y - py[i])**2
                if d < bd:
                    bd, best = d, (x, y)
            if best:
                xs[i], ys[i], meas[i] = best[0], best[1], True

    idx = np.arange(N); g = ~np.isnan(xs)
    if g.sum() < 2:
        raise RuntimeError("YOLO n'a quasi rien detecte — verifier conf/onnx.")
    xs[~g] = np.interp(idx[~g], idx[g], xs[g])
    ys[~g] = np.interp(idx[~g], idx[g], ys[g])
    ys_s = np.convolve(ys, np.ones(2)/2, mode='same')
    return dict(x=xs, y=ys, ys=ys_s, meas=meas, N=N)


def count_contacts(traj, fps):
    """Contacts = minima de hauteur (inversion de vitesse verticale)."""
    p, _ = find_peaks(traj['ys'], prominence=8, distance=int(0.12 * fps))
    return [int(c) for c in p]


def fuse_contacts(traj, frames, ball_contacts, fps):
    """Ajoute les coups de pied detectes par mouvement de jambe DANS les trous
    de detection du ballon (la ou la trajectoire est peu fiable)."""
    energy = KM.kick_energy(frames)
    kicks = KM.kick_peaks(energy, fps)
    meas = traj['meas']; N = traj['N']
    dens = np.array([meas[max(0, i-6):i+7].mean() for i in range(N)])
    final = set(ball_contacts); MIN = int(0.16 * fps)
    for k in kicks:
        if any(abs(k - c) < MIN for c in final):
            continue
        if dens[k] < 0.45:                 # uniquement dans les trous de detection
            final.add(int(k))
    return sorted(final), energy, kicks


def analyze_yolo(video_path, out_dir, onnx="yolov8m.onnx", use_kick_fusion=True):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    print(f"[+] {len(frames)} frames @ {fps:.0f} fps — detection YOLOv8 (toutes frames)...")

    traj = track_with_yolo(frames, onnx)
    print(f"[+] couverture detection {100*traj['meas'].mean():.0f}%")
    ball = count_contacts(traj, fps)
    if use_kick_fusion:
        contacts, _, _ = fuse_contacts(traj, frames, ball, fps)
        print(f"[+] contacts ballon={len(ball)}  -> fusion mouvement-jambe={len(contacts)}")
    else:
        contacts = ball

    metrics = A.compute_metrics(traj, contacts, fps)
    score = A.compute_score(metrics)
    json.dump(dict(metrics=metrics, score=score),
              open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    A.render_video(frames, traj, contacts, metrics, fps,
                   os.path.join(out_dir, "annotated.mp4"))
    print(f"[+] {metrics['total_juggles']} jongles | score {score['score_0_100']}/100 ({score['grade']})")
    return dict(metrics=metrics, score=score)


if __name__ == "__main__":
    vid  = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    out  = sys.argv[2] if len(sys.argv) > 2 else "out_yolo"
    onnx = sys.argv[3] if len(sys.argv) > 3 else "yolov8m.onnx"
    analyze_yolo(vid, out, onnx)
