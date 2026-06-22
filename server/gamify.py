#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gamify.py — Profil joueur facon FC : OVR, attributs, niveau/XP, palier, deblocages.
===================================================================================
Transforme l'historique des sessions en une carte joueur progressive (gamification)
tout en gardant des attributs derives des vraies metriques (serieux des stats).
"""

DIFF = {"pied": 1.0, "cuisse/genou": 0.8, "poitrine": 1.2, "tete": 1.6}

# 6 attributs facon FC (code court -> libelle)
ATTRS = [("CTR", "Contrôle"), ("RYT", "Rythme"), ("END", "Endurance"),
         ("EQU", "Équilibre"), ("VAR", "Variété"), ("TEC", "Technique")]

# poids OVR (somme = 1)
OVR_W = dict(CTR=0.24, RYT=0.20, END=0.18, EQU=0.14, VAR=0.14, TEC=0.10)


def _clamp(v, a=0, b=99):
    return max(a, min(b, v))


def _session_attrs(m):
    """Attributs 0-99 d'UNE session a partir de ses metriques."""
    streak = m.get("longest_streak", 0) or 0
    bp = m.get("by_body_part", {}) or {}
    tot = sum(bp.values()) or 1
    avg_diff = sum(DIFF.get(z, 1.0) * n for z, n in bp.items()) / tot
    return dict(
        CTR=_clamp(round((m.get("control_score", 0) or 0) * 99)),
        RYT=_clamp(round((m.get("rhythm_regularity", 0) or 0) * 99)),
        EQU=_clamp(round((m.get("balance_LR", 0) or 0) * 99)),
        VAR=_clamp(round((m.get("variety", 0) or 0) * 99)),
        END=_clamp(round(min(1.0, streak / 50) * 99)),
        TEC=_clamp(round((avg_diff - 0.8) / (1.6 - 0.8) * 99)),
    )


def level_from_xp(xp):
    """Niveau a partir de l'XP cumulee (paliers croissants)."""
    lvl, need, base = 1, 300, 0
    while xp >= base + need:
        base += need; lvl += 1; need = int(need * 1.35)
    return lvl, xp - base, need        # niveau, xp dans le niveau, xp pour le suivant


def _tier(ovr):
    if ovr >= 85: return "elite"
    if ovr >= 73: return "gold"
    if ovr >= 60: return "silver"
    return "bronze"


TIER_LABEL = {"bronze": "Bronze", "silver": "Argent", "gold": "Or", "elite": "Élite"}

# style joueur = attribut dominant
STYLE = dict(CTR="Technicien", RYT="Métronome", END="Marathonien",
             EQU="Ambidextre", VAR="Showman", TEC="Freestyler")


def _unlocks(t):
    """Trophees/capacites a debloquer, avec progression (current/target)."""
    def U(key, icon, label, desc, current, target):
        return dict(key=key, icon=icon, label=label, desc=desc,
                    current=round(current, 1), target=target,
                    unlocked=current >= target,
                    progress=round(min(1.0, current / target) if target else 1.0, 3))
    return [
        U("debut",     "🎬", "Premier pas",   "Réaliser 1 session",            t["sessions"], 1),
        U("regulier",  "📅", "Régulier",       "5 sessions",                    t["sessions"], 5),
        U("assidu",    "🔁", "Assidu",         "20 sessions",                   t["sessions"], 20),
        U("cent",      "💯", "Centurion",      "100 jongles cumulés",           t["total_juggles"], 100),
        U("mille",     "🔥", "Millier",        "1 000 jongles cumulés",         t["total_juggles"], 1000),
        U("serie20",   "⚡", "Série de 20",     "20 jongles d'affilée",          t["best_streak"], 20),
        U("serie50",   "🌟", "Série de 50",     "50 jongles d'affilée",          t["best_streak"], 50),
        U("maestro",   "🎯", "Maestro",        "Contrôle ≥ 85",                 t["best_control"], 85),
        U("metronome", "🥁", "Métronome",      "Rythme ≥ 85",                   t["best_rhythm"], 85),
        U("ambi",      "🦶", "Ambidextre",     "Équilibre G/D ≥ 90",            t["best_balance"], 90),
        U("artiste",   "🎨", "Artiste",        "Variété ≥ 70",                  t["best_variety"], 70),
        U("tete",      "🧠", "Jeu de tête",    "10 touches de tête cumulées",   t["head_touches"], 10),
        U("noteA",     "🏅", "Niveau A",       "Score ≥ 72 sur une session",    t["best_score"], 72),
        U("noteS",     "👑", "Niveau S",       "Score ≥ 85 sur une session",    t["best_score"], 85),
    ]


def build_profile(user, sessions):
    """sessions : liste {id, created_at, result:{metrics, score}} triee par date.
    Retourne le profil joueur complet (carte FC + progression)."""
    avatar = f"/assets/{user['username']}.png"
    metrics = [s["result"]["metrics"] for s in sessions if s.get("result")]
    scores  = [s["result"]["score"]["score_0_100"] for s in sessions if s.get("result")]
    n = len(metrics)

    if n == 0:
        return dict(
            user=user, avatar=avatar, has_data=False,
            ovr=None, tier="bronze", tier_label="Bronze", style="Recrue",
            level=1, xp=0, xp_in=0, xp_need=300,
            attrs=[dict(code=c, label=l, value=None) for c, l in ATTRS],
            totals=dict(sessions=0, total_juggles=0, total_time_s=0,
                        best_score=0, best_streak=0, head_touches=0),
            unlocks=_unlocks(dict(sessions=0, total_juggles=0, best_streak=0,
                                  best_control=0, best_rhythm=0, best_balance=0,
                                  best_variety=0, head_touches=0, best_score=0)),
        )

    # attributs carriere = moyenne des sessions
    per = [_session_attrs(m) for m in metrics]
    car = {c: round(sum(p[c] for p in per) / n) for c, _ in ATTRS}
    ovr = _clamp(round(sum(car[c] * w for c, w in OVR_W.items())))
    tier = _tier(ovr)
    style = STYLE[max(car, key=car.get)]

    # XP cumulee -> niveau
    xp = sum(int(m.get("total_juggles", 0)) + 2 * sc for m, sc in zip(metrics, scores))
    level, xp_in, xp_need = level_from_xp(xp)

    totals = dict(
        sessions=n,
        total_juggles=sum(int(m.get("total_juggles", 0)) for m in metrics),
        total_time_s=round(sum(float(m.get("duration_s", 0)) for m in metrics), 1),
        best_score=max(scores),
        best_streak=max(int(m.get("longest_streak", 0)) for m in metrics),
        head_touches=sum((m.get("by_body_part", {}) or {}).get("tete", 0) for m in metrics),
        best_control=max(p["CTR"] for p in per),
        best_rhythm=max(p["RYT"] for p in per),
        best_balance=max(p["EQU"] for p in per),
        best_variety=max(p["VAR"] for p in per),
    )

    return dict(
        user=user, avatar=avatar, has_data=True,
        ovr=ovr, tier=tier, tier_label=TIER_LABEL[tier], style=style,
        level=level, xp=xp, xp_in=xp_in, xp_need=xp_need,
        attrs=[dict(code=c, label=l, value=car[c]) for c, l in ATTRS],
        totals=totals,
        unlocks=_unlocks(totals),
    )
