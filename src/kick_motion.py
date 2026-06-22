#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kick_motion.py — Second signal independant de comptage des jongles.
==================================================================
A chaque touche, la jambe/le pied du joueur fait un mouvement bref vers le haut.
On mesure l'energie de mouvement (difference inter-frames) dans le BAS du cadre :
chaque pic = un coup de pied = un contact candidat.

Ce signal est COMPLEMENTAIRE de la trajectoire du ballon :
  - la trajectoire du ballon donne des contacts precis MAIS rate les jongles
    dont le ballon est en flou de mouvement (trous de detection).
  - le mouvement de jambe capte CHAQUE coup de pied, meme ballon invisible,
    mais il est plus bruite (decalage temporel, repositionnements).

=> On l'utilise pour RATTRAPER les contacts situes dans les trous de detection
   du ballon (cf. analyzer_yolo.fuse_contacts). Pour un signal propre et bien
   date, remplacer cette heuristique par le suivi de la cheville (MediaPipe Pose).
"""
import cv2, numpy as np
from scipy.signal import find_peaks


def kick_energy(frames, roi_top=0.45):
    """Energie de mouvement dans le bas du cadre, par frame (0..1)."""
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in frames]
    h = g[0].shape[0]; roi = slice(int(h * roi_top), h)
    e = np.zeros(len(frames))
    for i in range(1, len(frames)):
        e[i] = (np.abs(g[i][roi] - g[i-1][roi]) > 20).mean()
    return np.convolve(e, np.ones(3)/3, mode='same')


def kick_peaks(energy, fps, min_gap_s=0.18):
    pk, _ = find_peaks(energy, prominence=energy.std() * 0.5,
                       distance=int(min_gap_s * fps))
    return [int(p) for p in pk]
