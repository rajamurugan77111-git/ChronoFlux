"""Login, roles, CSRF, sensor key and alerts for the ChronoFlux console."""
import pytest
from fastapi.testclient import TestClient

from backend.api import ALERTS, USERS, app
from core.security import hash_password, make_token, read_token, verify_password


@pytest.fixture
def auth(monkeypatch):
    monkeypatch.setenv("CHRONOFLUX_AUTH", "1")
    monkeypatch.setenv("CHRONOFLUX_SENSOR_KEY", "sensor-key-1")
    return TestClient(app)


def login(c, user="admin", pw="test-admin-pass"):
    r = c.post("/api/auth/login", json={"username": user, "password": pw})
    return r, (r.json() if r.status_code == 200 else {}).get("csrf")


def test_password_hash_and_token_roundtrip():
    h = hash_password("s3cret-pass")
    assert verify_password("s3cret-pass", h) and not verify_password("nope", h)
    tok, csrf = make_token({"username": "a", "role": "admin"})
    assert read_token(tok)["csrf"] == csrf
    assert read_token(tok[:-2] + "00") is None             # tampered signature rejected


def test_pages_and_api_require_login(auth):
    r = auth.get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/login"
    assert auth.get("/api/latest").status_code == 401
    assert auth.get("/login").status_code == 200
    assert auth.get("/api/health").status_code == 200


def test_admin_session_and_csrf(auth):
    r, csrf = login(auth)
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert auth.get("/").status_code == 200
    assert auth.get("/api/latest").status_code == 200
    assert auth.post("/api/demo/seed").status_code == 403                       # no CSRF
    assert auth.post("/api/demo/seed", headers={"X-CSRF-Token": csrf}).status_code == 200


def test_analyst_restrictions(auth):
    _, csrf = login(auth)
    try:
        USERS.delete("meena")
    except Exception:
        pass
    assert auth.post("/api/users", json={"username": "meena", "password": "analyst-pw-1"},
                     headers={"X-CSRF-Token": csrf}).status_code == 201
    a = TestClient(app)
    _, acsrf = login(a, "meena", "analyst-pw-1")
    h = {"X-CSRF-Token": acsrf}
    assert a.get("/api/latest").status_code == 200
    assert a.post("/api/demo/seed", headers=h).status_code == 200              # analysts can run demos
    assert a.post("/api/temporal/train", headers=h).status_code == 403
    assert a.delete("/api/history", headers=h).status_code == 403
    assert a.get("/api/users").status_code == 403


def test_sensor_requires_key(auth):
    payload = {"sensor_id": "t", "features": {"proto": "tcp", "spkts": 5}, "telemetry": {}}
    assert auth.post("/api/sensor/live", json=payload).status_code == 401
    r = auth.post("/api/sensor/live", json=payload, headers={"X-API-Key": "sensor-key-1"})
    assert r.status_code == 200 and "prediction" in r.json()


def test_lockout(auth):
    for _ in range(5):
        assert login(auth, "ghost", "wrong-pass")[0].status_code == 401
    assert login(auth, "ghost", "wrong-pass")[0].status_code == 429


def test_live_alert_dispatch(monkeypatch):
    sent = []
    monkeypatch.setattr(ALERTS.cfg, "telegram_token", "1:x")
    monkeypatch.setattr(ALERTS.cfg, "telegram_chat_id", "9")
    monkeypatch.setattr(ALERTS, "_send", lambda ev: sent.append(ev) or {})
    ALERTS._last.clear()
    c = TestClient(app)
    hot = {"sensor_id": "gw", "features": {"proto": "tcp", "spkts": 900, "sbytes": 900000,
                                           "rate": 5000, "sttl": 254, "state": "INT"},
           "telemetry": {"source_ip": "198.51.100.77", "packet_count": 900, "syn_count": 800,
                         "unique_dst_ports": 600, "port_scan_score": 0.95}}
    for _ in range(3):
        r = c.post("/api/sensor/live", json=hot)
        assert r.status_code == 200
    ALERTS._pool.shutdown(wait=True)
    ALERTS._pool = __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=2)
    pred = r.json()["prediction"]
    if float(pred.get("risk_score") or 0) >= ALERTS.cfg.min_score:
        assert len(sent) == 1 and sent[0]["ip_address"] == "198.51.100.77"   # cool-down: one alert
    else:
        assert sent == []
