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


def track_with_yolo(frames, onnx, conf=0.10, gapfill=True, progress_cb=None):
    """Detecte le ballon par YOLO sur chaque frame -> trajectoire dense interpolee.

    progress_cb(frac) est appele periodiquement avec une fraction 0..1 couvrant
    les DEUX passes YOLO (detection principale + rattrapage des trous), pour que
    la barre de progression ne stagne pas pendant la 2e passe.
    """
    det = YOLOv8ONNX(onnx, conf=conf, imgsz=640)
    N = len(frames)
    xs = np.full(N, np.nan); ys = np.full(N, np.nan); meas = np.zeros(N, bool)

    # --- passe 1 : detection principale (0 .. 0.6 de la progression) ---------
    for i, fr in enumerate(frames):
        b = det.detect_ball(fr)
        if b:
            xs[i], ys[i], meas[i] = b[0], b[1], True
        if progress_cb and (i % 4 == 0 or i == N - 1):
            progress_cb(0.6 * (i + 1) / N)

    if gapfill:
        # --- passe 2 : rattrapage gate dans les trous (0.6 .. 1.0) -----------
        # candidat YOLO a tres bas seuil PRES de la position predite (evite tout
        # faux positif lointain).
        idx = np.arange(N); g = ~np.isnan(xs)
        px = np.interp(idx, idx[g], xs[g]); py = np.interp(idx, idx[g], ys[g])
        gaps = [i for i in range(N) if not meas[i]]
        G = max(len(gaps), 1)
        for j, i in enumerate(gaps):
            best, bd = None, 40**2
            for (x, y, r, sc) in det.detect_ball_candidates(frames[i], conf=0.05):
                d = (x - px[i])**2 + (y - py[i])**2
                if d < bd:
                    bd, best = d, (x, y)
            if best:
                xs[i], ys[i], meas[i] = best[0], best[1], True
            if progress_cb and (j % 4 == 0 or j == G - 1):
                progress_cb(0.6 + 0.4 * (j + 1) / G)
    elif progress_cb:
        progress_cb(1.0)

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


def analyze_yolo(video_path, out_dir, onnx="yolov8m.onnx", use_kick_fusion=True,
                 progress_cb=None, max_side=1280, max_frames=2400):
    """Analyse complete d'une video. progress_cb(pct, label) remonte la progression
    (0-100) avec un libelle d'etape, pour piloter une barre de progression.

    max_side : borne la plus grande dimension des frames (memoire + vitesse) ;
    max_frames : garde-fou sur les videos trop longues."""
    def report(pct, label):
        if progress_cb:
            progress_cb(pct, label)

    os.makedirs(out_dir, exist_ok=True)
    report(2, "Lecture de la vidéo")
    cap = cv2.VideoCapture(video_path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        if max_side:
            h, w = fr.shape[:2]
            m = max(h, w)
            if m > max_side:
                s = max_side / m
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))),
                                interpolation=cv2.INTER_AREA)
        frames.append(fr)
        if len(frames) >= max_frames:
            break
    cap.release()
    if len(frames) < 5:
        raise RuntimeError("Vidéo illisible ou trop courte (aucune frame exploitable).")
    print(f"[+] {len(frames)} frames @ {fps:.0f} fps — detection YOLOv8 (toutes frames)...")

    # Detection = phase la plus longue (2 passes YOLO) -> mappee sur 5%..72%
    def det_cb(frac):
        report(5 + int(67 * max(0.0, min(1.0, frac))), "Détection du ballon (YOLOv8)")
    traj = track_with_yolo(frames, onnx, progress_cb=det_cb)
    print(f"[+] couverture detection {100*traj['meas'].mean():.0f}%")

    report(74, "Comptage des touches")
    ball = count_contacts(traj, fps)
    if use_kick_fusion:
        contacts, _, _ = fuse_contacts(traj, frames, ball, fps)
        print(f"[+] contacts ballon={len(ball)}  -> fusion mouvement-jambe={len(contacts)}")
    else:
        contacts = ball

    report(80, "Calcul des métriques")
    metrics = A.compute_metrics(traj, contacts, fps)
    score = A.compute_score(metrics)
    json.dump(dict(metrics=metrics, score=score),
              open(os.path.join(out_dir, "metrics.json"), "w"), indent=2, ensure_ascii=False)

    report(84, "Génération de la vidéo annotée")
    A.render_video(frames, traj, contacts, metrics, fps,
                   os.path.join(out_dir, "annotated.mp4"))
    report(100, "Terminé")
    print(f"[+] {metrics['total_juggles']} jongles | score {score['score_0_100']}/100 ({score['grade']})")
    return dict(metrics=metrics, score=score)


if __name__ == "__main__":
    vid  = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    out  = sys.argv[2] if len(sys.argv) > 2 else "out_yolo"
    onnx = sys.argv[3] if len(sys.argv) > 3 else "yolov8m.onnx"
    analyze_yolo(vid, out, onnx)
