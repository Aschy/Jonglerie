#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
db.py — Persistance SQLite : utilisateurs + sessions de jonglage.
=================================================================
Stdlib uniquement (sqlite3, hashlib, hmac). Mots de passe hashes (pbkdf2-sha256).
Seed deux utilisateurs PoC : Steph (adulte, 168 cm) et Liam (enfant, 142 cm).
Ballon taille 5 = 22 cm de diametre pour les deux.
"""
import os, sqlite3, hashlib, hmac, secrets, json
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.environ.get("JONGLE_DB",
                         str(Path(os.environ.get("JONGLE_DATA", "data")) / "jonglerie.db"))

BALL_SIZE5_CM = 22.0   # diametre d'un ballon de foot taille 5

# Utilisateurs de depart (mots de passe simples pour le PoC)
SEED_USERS = [
    dict(username="steph", password="steph123", display_name="Steph",
         role="adulte", height_cm=168, ball_diam_cm=BALL_SIZE5_CM),
    dict(username="liam",  password="liam123",  display_name="Liam",
         role="enfant", height_cm=142, ball_diam_cm=BALL_SIZE5_CM),
]


def _conn():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# --- mots de passe -------------------------------------------------------- #
def hash_password(password: str, salt: str = None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100_000)
    return salt, h.hex()


def verify_password(password: str, salt: str, expected_hex: str) -> bool:
    _, got = hash_password(password, salt)
    return hmac.compare_digest(got, expected_hex)


# --- schema + seed -------------------------------------------------------- #
def init_db():
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            pass_salt TEXT NOT NULL,
            pass_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL,
            height_cm REAL NOT NULL,
            ball_diam_cm REAL NOT NULL,
            created_at TEXT NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS sessions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            filename TEXT,
            juggles INTEGER,
            score INTEGER,
            grade TEXT,
            duration_s REAL,
            tempo REAL,
            metrics_json TEXT,
            UNIQUE(user_id, id))""")
    for u in SEED_USERS:
        if not get_user_by_username(u["username"]):
            create_user(**u)


def create_user(username, password, display_name, role, height_cm, ball_diam_cm):
    salt, h = hash_password(password)
    with _conn() as c:
        cur = c.execute("""INSERT INTO users(username,pass_salt,pass_hash,display_name,
            role,height_cm,ball_diam_cm,created_at) VALUES(?,?,?,?,?,?,?,?)""",
            (username, salt, h, display_name, role, height_cm, ball_diam_cm, now_iso()))
        return cur.lastrowid


def _user_dict(r):
    if not r:
        return None
    d = dict(r)
    d.pop("pass_salt", None); d.pop("pass_hash", None)
    return d


def get_user_by_username(username):
    with _conn() as c:
        return c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()


def get_user_public(user_id):
    with _conn() as c:
        return _user_dict(c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())


def authenticate(username, password):
    r = get_user_by_username(username)
    if r and verify_password(password, r["pass_salt"], r["pass_hash"]):
        return _user_dict(r)
    return None


# --- sessions ------------------------------------------------------------- #
def add_session(user_id, result, filename=None):
    m = result["metrics"]; s = result["score"]
    with _conn() as c:
        cur = c.execute("""INSERT INTO sessions(user_id,created_at,filename,juggles,score,
            grade,duration_s,tempo,metrics_json) VALUES(?,?,?,?,?,?,?,?,?)""",
            (user_id, now_iso(), filename, m["total_juggles"], s["score_0_100"],
             s["grade"], m["duration_s"], m["tempo_touches_per_s"],
             json.dumps(result, ensure_ascii=False)))
        return cur.lastrowid


def list_sessions(user_id):
    with _conn() as c:
        rows = c.execute("""SELECT id,created_at,filename,juggles,score,grade,duration_s,tempo
            FROM sessions WHERE user_id=? ORDER BY created_at ASC""", (user_id,)).fetchall()
        return [dict(r) for r in rows]


def list_sessions_full(user_id):
    """Toutes les sessions avec leurs metriques completes (pour le profil joueur)."""
    with _conn() as c:
        rows = c.execute("""SELECT id,created_at,metrics_json FROM sessions
            WHERE user_id=? ORDER BY created_at ASC""", (user_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(id=r["id"], created_at=r["created_at"])
        d["result"] = json.loads(r["metrics_json"]) if r["metrics_json"] else None
        out.append(d)
    return out


def get_session(user_id, session_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM sessions WHERE user_id=? AND id=?",
                      (user_id, session_id)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["result"] = json.loads(d.pop("metrics_json")) if d.get("metrics_json") else None
        return d


if __name__ == "__main__":
    init_db()
    print("DB:", DB_PATH)
    for u in SEED_USERS:
        print(" -", u["username"], "/", u["password"], "->", u["display_name"])
