#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cutout_avatar.py — Détoure un avatar joueur sur fond damier (PNG sans alpha)
============================================================================
Les renders fournis ont le damier de transparence APLATI en pixels gris clair
(pas de canal alpha). On reconstruit une vraie transparence :
  1. GrabCut (modèle couleur + cohérence spatiale) -> silhouette propre,
     sépare correctement les vêtements blancs du fond clair (gradients de bord).
  2. Nettoyage des résidus de damier que GrabCut garde parfois près des jambes :
     le damier est clair-neutre ET très texturé (std locale ~40 sur fenêtre 49px),
     alors que le tissu lisse est ~8 -> seuil à 24 les sépare sans toucher aux habits.
  3. Plus grand composant + bouchage des trous + léger feather -> RGBA.

Usage : cutout_avatar.py <entrée.png> <sortie.png>
"""
import sys, cv2, numpy as np


def cutout(src):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"illisible: {src}")
    h, w = img.shape[:2]
    a = img.astype(np.int16); b, g, r = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mn = np.minimum(np.minimum(b, g), r); mx = np.maximum(np.maximum(b, g), r)

    # bbox du sujet (tout ce qui n'est pas clair-neutre)
    subj = ~((mn > 195) & ((mx - mn) < 22))
    ys, xs = np.where(subj)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    rect = (max(0, x0 - 8), max(0, y0 - 8),
            min(w, x1 + 8) - max(0, x0 - 8), min(h, y1 + 8) - max(0, y0 - 8))

    # 1. GrabCut
    mask = np.zeros((h, w), np.uint8)
    cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 2. résidus de damier que GrabCut a gardés (concavités entre les jambes…).
    #    Le damier est clair-neutre ET texturé (std~40 / fenêtre 49) ; le tissu
    #    blanc est lisse (~8). On PROPAGE la transparence du fond dans le damier
    #    qui lui est connecté : on retire tout damier relié au fond, mais le blanc
    #    lisse (short/chaussettes) sert de barrière -> vêtements intacts, et les
    #    patchs de damier intérieurs (ombres du short) ne sont pas touchés.
    light = (mn > 205) & ((mx - mn) < 24)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    me = cv2.blur(gray, (49, 49)); sq = cv2.blur(gray * gray, (49, 49))
    std = cv2.sqrt(np.maximum(sq - me * me, 0))
    checker = (light & (std > 18)).astype(np.uint8)
    allowed = (((fg == 0) | (checker > 0))).astype(np.uint8)    # fond + damier traversables
    nl, lab = cv2.connectedComponents(allowed)
    bl = (set(lab[0, :].tolist()) | set(lab[-1, :].tolist()) |
          set(lab[:, 0].tolist()) | set(lab[:, -1].tolist())); bl.discard(0)
    fg[np.isin(lab, list(bl))] = 0                              # fond + damier connecté -> retirés

    # 3. plus grand composant + bouchage trous + nettoyage + feather
    nl, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if nl > 1:
        fg = np.where(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    ff = fg.copy(); m2 = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255); fg = fg | cv2.bitwise_not(ff)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), 1)
    alpha = cv2.GaussianBlur(fg, (0, 0), 1.3)
    return np.dstack([img, alpha]).astype(np.uint8)


def autocrop(r, padx=14, padbot=6):
    """Recadre sur le contenu (boîte alpha) avec une petite marge."""
    al = r[:, :, 3]; ys, xs = np.where(al > 20)
    y0, y1 = ys.min(), min(r.shape[0] - 1, ys.max() + padbot)
    x0, x1 = max(0, xs.min() - padx), min(r.shape[1] - 1, xs.max() + padx)
    return r[y0:y1 + 1, x0:x1 + 1]


def finalize(src, dst, crop_bottom=None):
    """Détoure + (option) coupe sous une ligne (cadrage buste) + autocrop -> PNG RGBA.
    crop_bottom évite le bas des jambes (segmentation ambiguë blanc/fond) et donne
    un cadrage façon carte FC."""
    r = cutout(src)
    if crop_bottom:
        r[int(crop_bottom):, :, 3] = 0
    cv2.imwrite(dst, autocrop(r))
    print(f"[✓] {dst}")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    cb = int(sys.argv[3]) if len(sys.argv) > 3 else None
    finalize(src, dst, cb)
