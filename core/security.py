"""Authentication, roles and sensor keys for the ChronoFlux console.

Roles
-----
* admin   - everything: retrain models, clear history/devices, manage users
* analyst - monitor, run demos and offline (CSV/PCAP) analysis, view evidence

Sensors authenticate with the X-API-Key header (CHRONOFLUX_SENSOR_KEY).

Implementation uses only the Python standard library:
* passwords: scrypt (hashlib) with a per-user random salt
* sessions : HMAC-SHA256 signed token in an HttpOnly, SameSite=Lax cookie
* CSRF     : per-session token required in X-CSRF-Token on state-changing requests
* lockout  : 5 failed logins for a username lock it for 5 minutes
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ROLES = ("admin", "analyst")
COOKIE = "cf_session"
SESSION_SECONDS = 8 * 3600
MAX_FAILURES, LOCKOUT_SECONDS, MIN_PASSWORD = 5, 300, 8

ADMIN_ONLY = {
    ("POST", "/api/temporal/train"),
    ("DELETE", "/api/history"),
    ("DELETE", "/api/devices"),
}


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def auth_enabled() -> bool:
    return _env("CHRONOFLUX_AUTH", "1").lower() not in ("0", "false", "no")


def sensor_key() -> str:
    return _env("CHRONOFLUX_SENSOR_KEY")


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_b64, hash_b64 = stored.split("$")
        digest = hashlib.scrypt((password or "").encode(), salt=base64.b64decode(salt_b64),
                                n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(digest, base64.b64decode(hash_b64))
    except Exception:
        return False


_DUMMY = hash_password("timing-equaliser")


class UserStore:
    def __init__(self, path: Optional[Path] = None):
        path = path or Path(_env("CHRONOFLUX_USERS_DB", str(DATA / "users.db")))
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.execute("""CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            role TEXT NOT NULL, created_at REAL NOT NULL, last_login REAL)""")
        self._failures: Dict[str, List[float]] = {}

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def create(self, username: str, password: str, role: str = "analyst") -> None:
        username = (username or "").strip().lower()
        if not username or len(username) > 32 or not username.replace("_", "").replace(".", "").isalnum():
            raise ValueError("Username must be 1-32 letters/digits (._ allowed)")
        if role not in ROLES:
            raise ValueError(f"Role must be one of {ROLES}")
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
        with self._lock:
            try:
                self._conn.execute("INSERT INTO users VALUES (?, ?, ?, ?, NULL)",
                                   (username, hash_password(password), role, time.time()))
                self._conn.commit()
            except sqlite3.IntegrityError:
                raise ValueError(f"User '{username}' already exists")

    def set_password(self, username: str, password: str) -> None:
        if len(password or "") < MIN_PASSWORD:
            raise ValueError(f"Password must be at least {MIN_PASSWORD} characters")
        with self._lock:
            self._conn.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                               (hash_password(password), username))
            self._conn.commit()

    def delete(self, username: str) -> None:
        with self._lock:
            row = self._conn.execute("SELECT role FROM users WHERE username = ?", (username,)).fetchone()
            if not row:
                raise KeyError(username)
            admins = self._conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
            if row["role"] == "admin" and admins <= 1:
                raise ValueError("Cannot delete the last admin")
            self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
            self._conn.commit()

    def list(self) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT username, role, created_at, last_login FROM users ORDER BY username")]

    def get(self, username: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT username, role FROM users WHERE username = ?",
                                   (username,)).fetchone()
        return dict(r) if r else None

    def locked(self, username: str) -> bool:
        now = time.time()
        recent = [t for t in self._failures.get(username, []) if now - t < LOCKOUT_SECONDS]
        self._failures[username] = recent
        return len(recent) >= MAX_FAILURES

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        username = (username or "").strip().lower()
        if self.locked(username):
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        ok = verify_password(password, row["password_hash"] if row else _DUMMY)
        if not (row and ok):
            self._failures.setdefault(username, []).append(time.time())
            return None
        self._failures.pop(username, None)
        with self._lock:
            self._conn.execute("UPDATE users SET last_login = ? WHERE username = ?", (time.time(), username))
            self._conn.commit()
        return {"username": row["username"], "role": row["role"]}

    def ensure_admin(self) -> Optional[str]:
        if self.count():
            return None
        pw = _env("CHRONOFLUX_ADMIN_PASSWORD")
        generated = None
        if not pw:
            pw = generated = secrets.token_urlsafe(12)
        self.create("admin", pw, "admin")
        return generated


# ------------------------------------------------------------------ sessions
def _secret() -> bytes:
    env = _env("CHRONOFLUX_SECRET_KEY")
    if env:
        return env.encode()
    path = Path(_env("CHRONOFLUX_SECRET_KEY_PATH", str(DATA / "secret_key")))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path.read_text().strip().encode()


def make_token(user: dict) -> tuple[str, str]:
    csrf = secrets.token_urlsafe(24)
    payload = {"u": user["username"], "r": user["role"], "exp": int(time.time()) + SESSION_SECONDS,
               "csrf": csrf}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}", csrf


def read_token(token: Optional[str]) -> Optional[dict]:
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(_secret(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception:
        return None
    if payload.get("exp", 0) < time.time():
        return None
    return payload
