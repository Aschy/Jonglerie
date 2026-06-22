#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auth.py — Tokens de session signes (HMAC, stdlib, sans etat serveur).
=====================================================================
token = base64url("{user_id}.{exp}.{sig}"), sig = HMAC-SHA256(secret, "uid.exp").
Le secret est genere une fois et persiste dans DATA/secret.key.
"""
import os, hmac, hashlib, base64, time
from pathlib import Path

TTL = 30 * 24 * 3600   # 30 jours
_SECRET = None


def _secret() -> bytes:
    global _SECRET
    if _SECRET is None:
        p = Path(os.environ.get("JONGLE_DATA", "data")) / "secret.key"
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_text(os.urandom(32).hex())
            try: os.chmod(p, 0o600)
            except Exception: pass
        _SECRET = bytes.fromhex(p.read_text().strip())
    return _SECRET


def _sign(msg: str) -> str:
    return hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()


def make_token(user_id: int) -> str:
    exp = int(time.time()) + TTL
    msg = f"{user_id}.{exp}"
    raw = f"{msg}.{_sign(msg)}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def verify_token(token: str):
    """Retourne user_id (int) si valide et non expire, sinon None."""
    try:
        raw = base64.urlsafe_b64decode(token.encode()).decode()
        uid, exp, sig = raw.rsplit(".", 2)
        if not hmac.compare_digest(sig, _sign(f"{uid}.{exp}")):
            return None
        if int(exp) < time.time():
            return None
        return int(uid)
    except Exception:
        return None
