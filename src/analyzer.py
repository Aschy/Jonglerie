#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POC — Analyse de session de jongle foot a partir d'une video.
=============================================================
Pipeline 100% vision classique (auto-suffisant, sans modele a telecharger) :

  1. BallTracker      : detection ballon (couleur + mouvement) -> chainage Viterbi
  2. ContactDetector  : contacts = minima de hauteur du ballon (signal processing)
  3. BodyZone         : classification de la partie du corps par hauteur de contact
  4. Metrics          : duree, nb jongles, tempo, equilibre G/D, regularite, controle...
  5. Score            : score composite ponderE par difficulte
  6. Renderer         : video annotee + tableau de bord PNG + metrics.json

>>> EN PRODUCTION : remplacer `BallTracker.candidates()` par un YOLOv8 fine-tune
    sur ballons de foot, et `BodyZone` par MediaPipe Pose. TOUT le reste (contacts,
    metriques, score, rendu) reste identique. Ce sont les 2 seuls points a upgrader.
"""
import cv2, numpy as np, json, sys, os
from scipy.signal import find_peaks

# --------------------------------------------------------------------------- #
#  1. SUIVI DU BALLON                                                          #
# --------------------------------------------------------------------------- #
class BallTracker:
    """Detection multi-indices + association globale par Viterbi."""

    def __init__(self, fps):
        self.fps = fps

    # --- candidats couleur : ballon clair, non-vert, rond -------------------
    @staticmethod
    def _color_candidates(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, (28, 30, 30), (92, 255, 255))
        dark  = cv2.inRange(hsv, (0, 0, 0), (180, 255, 65))
        mask  = cv2.bitwise_not(cv2.bitwise_or(green, dark))
        mask  = cv2.bitwise_and(mask, cv2.inRange(hsv, (0, 0, 110), (180, 255, 255)))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return BallTracker._blobs(mask)

    # --- candidats mouvement : rattrape le ballon devant le corps -----------
    @staticmethod
    def _motion_candidates(bgr, prev_gray):
        if prev_gray is None:
            return []
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        diff = np.abs(gray - prev_gray).astype(np.uint8)
        _, mv = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        notgreen = cv2.bitwise_not(cv2.inRange(hsv, (28, 30, 30), (92, 255, 255)))
        bright   = cv2.inRange(hsv, (0, 0, 95), (180, 255, 255))
        m = cv2.bitwise_and(cv2.bitwise_and(mv, notgreen), bright)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
        return BallTracker._blobs(m, min_circ=0.4)

    @staticmethod
    def _blobs(mask, min_circ=0.45):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 120 or a > 11000:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            if r < 7 or r > 48:
                continue
            circ = a / (np.pi * r * r)
            if circ < min_circ:
                continue
            out.append((float(x), float(y), float(r), float(circ)))
        return out

    def candidates(self, frames):
        """Retourne, par frame, une liste de candidats (x,y,r,circ)."""
        merged, prev = [], None
        for bgr in frames:
            cs = self._color_candidates(bgr) + self._motion_candidates(bgr, prev)
            keep = []
            for c in sorted(cs, key=lambda z: -z[3]):            # dedup spatial
                if all((c[0]-k[0])**2 + (c[1]-k[1])**2 > 400 for k in keep):
                    keep.append(c)
            merged.append(keep)
            prev = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.int16)
        return merged

    def track(self, frames):
        """Association globale (Viterbi) -> trajectoire lissee + flags mesure."""
        cand = self.candidates(frames)
        N = len(cand)
        MISS = ("miss",)
        states = [[tuple(c) for c in cand[i]] + [MISS] for i in range(N)]

        def emit(s):  return 12.0 if s == MISS else 8.0 * (1.0 - s[3])
        def trans(a, b):
            if a == MISS or b == MISS:
                return 35.0
            d = ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** .5
            return d if d < 70 else d * 3

        dp = [dict() for _ in range(N)]
        bp = [dict() for _ in range(N)]
        for k, s in enumerate(states[0]):
            dp[0][k] = emit(s)
        for i in range(1, N):
            for k, s in enumerate(states[i]):
                ce = emit(s); best, bk = 1e18, -1
                for pk, ps in enumerate(states[i-1]):
                    c = dp[i-1][pk] + trans(ps, s) + ce
                    if c < best:
                        best, bk = c, pk
                dp[i][k], bp[i][k] = best, bk
        k = min(dp[N-1], key=dp[N-1].get)
        path = [k]
        for i in range(N-1, 0, -1):
            k = bp[i][k]; path.append(k)
        path = path[::-1]

        xs = np.full(N, np.nan); ys = np.full(N, np.nan); meas = np.zeros(N, bool)
        for i, k in enumerate(path):
            s = states[i][k]
            if s != MISS:
                xs[i], ys[i], meas[i] = s[0], s[1], True
        idx = np.arange(N); g = ~np.isnan(xs)
        xs[~g] = np.interp(idx[~g], idx[g], xs[g])
        ys[~g] = np.interp(idx[~g], idx[g], ys[g])
        ys_s = np.convolve(ys, np.ones(3)/3, mode='same')
        return dict(x=xs, y=ys, ys=ys_s, meas=meas, N=N)


# --------------------------------------------------------------------------- #
#  2. DETECTION DES CONTACTS  (= jongles)                                      #
# --------------------------------------------------------------------------- #
class ContactDetector:
    """Un contact = le ballon au plus bas avant de repartir (minimum de hauteur)."""
    def __init__(self, fps): self.fps = fps

    def detect(self, traj):
        y, meas = traj['ys'], traj['meas']
        # prominence 28 px : un vrai contact fait nettement remonter le ballon ;
        # en-dessous = bruit de trajectoire (souvent en zone interpolee). distance
        # 0.18 s : ecarte les doubles-comptages d'un meme contact.
        peaks, _ = find_peaks(y, prominence=28, distance=int(0.18 * self.fps))
        return [int(p) for p in peaks if meas[max(0, p-2):p+3].any()]


# --------------------------------------------------------------------------- #
#  3. ZONE DU CORPS (proxy hauteur — a remplacer par MediaPipe Pose en prod)   #
# --------------------------------------------------------------------------- #
class BodyZone:
    ZONES = ["tete", "poitrine", "cuisse/genou", "pied"]
    def __init__(self, traj):
        y = traj['y']
        self.feet  = np.percentile(y, 95)      # ballon au plus bas ~ niveau pieds
        self.head  = np.percentile(y, 3)        # ballon au plus haut ~ niveau tete
        self.cx    = float(np.median(traj['x']))

    def classify(self, p, traj):
        by = traj['y'][p]; bx = traj['x'][p]
        span = max(self.feet - self.head, 1)
        f = (by - self.head) / span             # 0 (tete) -> 1 (pieds)
        if   f > 0.78: zone = "pied"
        elif f > 0.50: zone = "cuisse/genou"
        elif f > 0.22: zone = "poitrine"
        else:          zone = "tete"
        side = "gauche" if bx < self.cx else "droite"   # cote ecran (proxy)
        return zone, side


# --------------------------------------------------------------------------- #
#  4 & 5. METRIQUES + SCORE                                                    #
# --------------------------------------------------------------------------- #
DIFFICULTY = {"pied": 1.0, "cuisse/genou": 0.8, "poitrine": 1.2, "tete": 1.6}

def compute_metrics(traj, contacts, fps):
    N = traj['N']; duration = N / fps; y = traj['ys']
    bz = BodyZone(traj)
    touches = []
    for p in contacts:
        zone, side = bz.classify(p, traj)
        touches.append(dict(frame=p, t=round(p/fps, 2), zone=zone, side=side,
                            height_px=int(bz.feet - traj['y'][p])))
    n = len(touches)
    times = np.array([c['frame']/fps for c in touches])
    intervals = np.diff(times) if n > 1 else np.array([])

    # repartition
    by_zone = {z: sum(t['zone'] == z for t in touches) for z in BodyZone.ZONES}
    left  = sum(t['side'] == 'gauche' for t in touches)
    right = n - left

    # rythme : coefficient de variation des intervalles (faible = metronomique)
    cv = float(np.std(intervals)/np.mean(intervals)) if len(intervals) else 0.0
    regularity = max(0.0, 1.0 - cv)                 # 1 = parfaitement regulier
    tempo = n / duration if duration else 0.0

    # controle : variance relative de la hauteur d'apex (faible = maitrise)
    apex_y = -y; apex = find_peaks(apex_y, prominence=15)[0]
    apex_h = (bz.feet - y[apex]) if len(apex) else np.array([0])
    control = float(max(0.0, 1.0 - (np.std(apex_h)/(np.mean(apex_h)+1e-6))))

    # plus longue serie (ici : tout le clip est une serie continue, 0 chute visible)
    drops = 0
    longest_streak = n

    # equilibre G/D : 1 = parfait 50/50
    balance = 1.0 - abs(left - right)/n if n else 0.0
    # variete : entropie normalisee des zones utilisees
    ps = np.array([v for v in by_zone.values() if v]) / max(n, 1)
    variety = float(-(ps*np.log(ps)).sum()/np.log(len(BodyZone.ZONES))) if n else 0.0

    # --- Dynamique du ballon : hauteurs (bas/haut) + vitesse ------------------
    # Echelle = diametre median du ballon (px) -> valeurs en "diametres de ballon"
    # (independant de la resolution). Estimation km/h via Ø reel ~22 cm (monoculaire,
    # profondeur ignoree -> ordre de grandeur, pas une mesure exacte).
    diam = traj.get('ball_diam')
    x = traj['x']
    contact_h = np.array([bz.feet - y[p] for p in contacts], float) if contacts else np.array([0.0])
    speed_px_s = (np.hypot(np.diff(x), np.diff(y)) * fps) if N > 1 else np.array([0.0])
    KMH = 0.22 * 3.6                       # (diam/s) -> km/h si Ø = 22 cm
    pd = lambda v: round(float(v)/diam, 1) if diam else None
    kmh = lambda v: round(float(v)/diam*KMH, 1) if diam else None
    ball_dynamics = dict(
        ball_diam_px          = round(diam, 1) if diam else None,
        contact_height_diam   = pd(np.mean(contact_h)),     # hauteur au toucher (bas)
        apex_height_mean_diam = pd(np.mean(apex_h)),        # hauteur moyenne en l'air
        apex_height_max_diam  = pd(np.max(apex_h)),         # hauteur max (haut)
        speed_mean_diam_s     = pd(np.mean(speed_px_s)),
        speed_max_diam_s      = pd(np.percentile(speed_px_s, 95)),
        speed_mean_kmh_est    = kmh(np.mean(speed_px_s)),
        speed_max_kmh_est     = kmh(np.percentile(speed_px_s, 95)),
    )

    return dict(
        duration_s=round(duration, 2),
        total_juggles=n,
        tempo_touches_per_s=round(tempo, 2),
        left_foot_side=left, right_foot_side=right,
        balance_LR=round(balance, 2),
        by_body_part=by_zone,
        rhythm_regularity=round(regularity, 2),
        rhythm_cv=round(cv, 2),
        control_score=round(control, 2),
        variety=round(variety, 2),
        drops=drops, longest_streak=longest_streak,
        mean_interval_s=round(float(np.mean(intervals)), 2) if len(intervals) else None,
        ball_dynamics=ball_dynamics,
        touches=touches,
    )

def compute_score(m):
    base = sum(DIFFICULTY[t['zone']] for t in m['touches']) * 10        # points/touche ponderes
    bonus = (m['balance_LR']*40 + m['rhythm_regularity']*40 +
             m['control_score']*40 + m['variety']*30 +
             min(m['duration_s']/15, 1)*20)
    penalty = m['drops'] * 25
    raw = base + bonus - penalty
    score100 = int(max(0, min(100, raw / (m['total_juggles']*12 + 170) * 100))) if m['total_juggles'] else 0
    grade = ("S" if score100>=85 else "A" if score100>=72 else "B" if score100>=58
             else "C" if score100>=42 else "D")
    return dict(score_0_100=score100, grade=grade,
                breakdown=dict(base=round(base,1), bonus=round(bonus,1), penalty=penalty))


# --------------------------------------------------------------------------- #
#  6. RENDU : video annotee                                                    #
# --------------------------------------------------------------------------- #
def render_video(frames, traj, contacts, metrics, fps, out_path):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    cset = set(contacts); count = 0; flash = 0; trail = []
    cmap = {"pied":(80,220,90),"cuisse/genou":(0,200,255),
            "poitrine":(255,140,0),"tete":(0,80,255)}
    tinfo = {c['frame']: c for c in metrics['touches']}
    for i, bgr in enumerate(frames):
        f = bgr.copy()
        bx, by = int(traj['x'][i]), int(traj['y'][i])
        trail.append((bx, by)); trail = trail[-14:]
        for j, (tx, ty) in enumerate(trail):                     # trainee
            cv2.circle(f, (tx, ty), 2, (255, 255, 255), -1)
        col = (0, 0, 255) if traj['meas'][i] else (160, 160, 160)
        cv2.circle(f, (bx, by), 13, col, 2)
        if i in cset:
            count += 1; flash = 6
            z = tinfo[i]['zone']
        if flash > 0:
            cv2.circle(f, (bx, by), 20, (0, 255, 255), 3); flash -= 1
        # bandeau
        cv2.rectangle(f, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.putText(f, f"JONGLES: {count}", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(f, f"{i/fps:4.1f}s", (w-90, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        vw.write(f)
    vw.release()


# --------------------------------------------------------------------------- #
#  ORCHESTRATION                                                               #
# --------------------------------------------------------------------------- #
def analyze(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        frames.append(fr)
    cap.release()
    print(f"[+] {len(frames)} frames @ {fps:.0f} fps")

    tracker = BallTracker(fps)
    traj = tracker.track(frames)
    print("[+] tracking termine")
    contacts = ContactDetector(fps).detect(traj)
    print(f"[+] {len(contacts)} contacts detectes")
    metrics = compute_metrics(traj, contacts, fps)
    score = compute_score(metrics)
    report = dict(metrics=metrics, score=score)
    json.dump(report, open(os.path.join(out_dir, "metrics.json"), "w"),
              indent=2, ensure_ascii=False)
    render_video(frames, traj, contacts, metrics, fps,
                 os.path.join(out_dir, "annotated.mp4"))
    print(f"[+] score = {score['score_0_100']}/100 (grade {score['grade']})")
    # expose pour le tableau de bord
    np.save(os.path.join(out_dir, "_traj_y.npy"), traj['ys'])
    np.save(os.path.join(out_dir, "_traj_meas.npy"), traj['meas'])
    json.dump([int(c) for c in contacts],
              open(os.path.join(out_dir, "_contacts.json"), "w"))
    return report


if __name__ == "__main__":
    vid = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
    out = sys.argv[2] if len(sys.argv) > 2 else "out"
    analyze(vid, out)
