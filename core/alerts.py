"""Alert notifications: email (SMTP) and Telegram.

Configured with environment variables (see .env.example). An alert is sent when an
event's final risk score reaches BG_ALERT_MIN_SCORE (default 60 = High Risk).
A per-IP cool-down (BG_ALERT_COOLDOWN, default 10 min) prevents alert storms:
a brute-force attack produces one alert, not hundreds. A later alert for the same
IP is still sent if the severity escalates (e.g. High -> Critical).

Sending happens on a background thread so ingestion is never slowed down.
"""
from __future__ import annotations

import logging
import smtplib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Callable, Dict, List, Optional, Tuple

import requests

log = logging.getLogger("chronoflux.alerts")
ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


@dataclass
class AlertConfig:
    min_score: int = 60
    cooldown_seconds: int = 600
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_tls: bool = True
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    telegram_token: str = ""
    telegram_chat_id: str = ""
    dashboard_url: str = ""

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.email_to)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_token and self.telegram_chat_id)

    @classmethod
    def from_env(cls, env) -> "AlertConfig":
        return cls(
            min_score=int(env.get("CHRONOFLUX_ALERT_MIN_SCORE", "60")),
            cooldown_seconds=int(env.get("CHRONOFLUX_ALERT_COOLDOWN", "600")),
            smtp_host=env.get("CHRONOFLUX_SMTP_HOST", ""),
            smtp_port=int(env.get("CHRONOFLUX_SMTP_PORT", "587")),
            smtp_user=env.get("CHRONOFLUX_SMTP_USER", ""),
            smtp_password=env.get("CHRONOFLUX_SMTP_PASSWORD", ""),
            smtp_tls=env.get("CHRONOFLUX_SMTP_TLS", "true").lower() == "true",
            email_from=env.get("CHRONOFLUX_ALERT_EMAIL_FROM", env.get("CHRONOFLUX_SMTP_USER", "")),
            email_to=[a.strip() for a in env.get("CHRONOFLUX_ALERT_EMAIL_TO", "").split(",") if a.strip()],
            telegram_token=env.get("CHRONOFLUX_TELEGRAM_BOT_TOKEN", ""),
            telegram_chat_id=env.get("CHRONOFLUX_TELEGRAM_CHAT_ID", ""),
            dashboard_url=env.get("CHRONOFLUX_PUBLIC_URL", ""),
        )


def format_alert(event: dict, dashboard_url: str = "") -> Tuple[str, str]:
    subject = (f"[ChronoFlux] {event['severity']} ({event['risk_score']}/100): "
               f"{event.get('stage', 'Attack')} from {event['ip_address']}")
    lines = [
        f"Risk     : {event['severity']} - {event['risk_score']}/100",
        f"Stage    : {event.get('stage', '-')}  (MITRE: {', '.join(event.get('mitre', [])[:3]) or '-'})",
        f"Source IP: {event['ip_address']}",
        f"Sensor   : {event.get('sensor_id', '-')}",
        f"Forecast : next stage {event.get('forecast_next') or '-'}, "
        f"infiltration probability {event.get('infiltration', '-')}",
        f"Time     : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.get('created_at', time.time())))}",
        "",
        "Why:",
        *[f"  - {r}" for r in event.get("reasons", [])[:8]],
    ]
    if dashboard_url:
        lines += ["", f"Investigate: {dashboard_url.rstrip('/')}/"]
    return subject, "\n".join(lines)


class AlertManager:
    def __init__(self, cfg: AlertConfig, record: Optional[Callable[[dict], None]] = None):
        self.cfg = cfg
        self.record = record
        self._last: Dict[str, Tuple[float, int]] = {}   # ip -> (time, severity rank)
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cf-alert")

    @property
    def enabled(self) -> bool:
        return self.cfg.email_enabled or self.cfg.telegram_enabled

    def channels(self) -> List[str]:
        return [c for c, on in (("email", self.cfg.email_enabled),
                                ("telegram", self.cfg.telegram_enabled)) if on]

    def should_alert(self, event: dict) -> bool:
        if event["risk_score"] < self.cfg.min_score:
            return False
        rank = ORDER.index(event["severity"]) if event["severity"] in ORDER else 0
        now = time.time()
        with self._lock:
            last = self._last.get(event["ip_address"])
            if last and now - last[0] < self.cfg.cooldown_seconds and rank <= last[1]:
                return False   # same or lower severity within cool-down
            self._last[event["ip_address"]] = (now, rank)
        return True

    def handle(self, event: dict, wait: bool = False):
        """Called for every scored event. Returns the future (or None)."""
        if not self.enabled or not self.should_alert(event):
            return None
        fut = self._pool.submit(self._send, dict(event))
        if wait:
            fut.result(timeout=30)
        return fut

    def send_test(self) -> dict:
        event = {"ip_address": "203.0.113.1", "stage": "Alert test", "risk_score": 99,
                 "severity": "CRITICAL", "created_at": time.time(),
                 "reasons": ["This is a test alert from ChronoFlux."]}
        return self._send(event)

    # ------------------------------------------------------------------ send
    def _send(self, event: dict) -> dict:
        subject, body = format_alert(event, self.cfg.dashboard_url)
        results = {}
        if self.cfg.email_enabled:
            results["email"] = self._safe(self._send_email, subject, body)
        if self.cfg.telegram_enabled:
            results["telegram"] = self._safe(self._send_telegram, f"🚨 {subject}\n\n{body}")
        record = {"created_at": time.time(), "ip_address": event["ip_address"],
                  "severity": event["severity"], "risk_score": event["risk_score"],
                  "stage": event.get("stage"), "results": results}
        if self.record:
            try:
                self.record(record)
            except Exception:  # never let bookkeeping break alerting
                log.exception("could not record alert")
        return record

    @staticmethod
    def _safe(fn, *args) -> str:
        try:
            fn(*args)
            return "sent"
        except Exception as exc:
            log.warning("alert channel failed: %s", exc)
            return f"failed: {exc}"

    def _send_email(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.cfg.email_from or "chronoflux@localhost", ", ".join(self.cfg.email_to)
        msg.set_content(body)
        with smtplib.SMTP(self.cfg.smtp_host, self.cfg.smtp_port, timeout=15) as s:
            if self.cfg.smtp_tls:
                s.starttls()
            if self.cfg.smtp_user:
                s.login(self.cfg.smtp_user, self.cfg.smtp_password)
            s.send_message(msg)

    def _send_telegram(self, text: str) -> None:
        r = requests.post(f"https://api.telegram.org/bot{self.cfg.telegram_token}/sendMessage",
                          json={"chat_id": self.cfg.telegram_chat_id, "text": text[:4000]},
                          timeout=15)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram API {r.status_code}: {r.text[:200]}")
