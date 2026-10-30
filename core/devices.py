"""Small persistent device-intelligence registry for live gateway observations."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]
DEVICE_DIR = ROOT / "data" / "live"
DEVICE_PATH = DEVICE_DIR / "devices.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if not DEVICE_PATH.exists():
        return {}
    try:
        data = json.loads(DEVICE_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    DEVICE_DIR.mkdir(parents=True, exist_ok=True)
    DEVICE_PATH.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def update(telemetry: dict | None, timestamp: str | None, prediction: dict | None) -> dict | None:
    telemetry = telemetry or {}
    prediction = prediction or {}
    ip = telemetry.get("source_ip")
    if not ip:
        return None

    key = str(telemetry.get("source_mac") or ip).lower()
    data = _load()
    now = timestamp or _now()
    item = data.get(key, {
        "device_id": key,
        "ip": ip,
        "mac": telemetry.get("source_mac"),
        "first_seen": now,
        "last_seen": now,
        "observations": 0,
        "ports": [],
        "destinations": [],
        "protocols": [],
        "services": [],
        "risk_level": "LOW",
        "current_stage": "Normal",
    })

    item["ip"] = ip
    item["mac"] = telemetry.get("source_mac") or item.get("mac")
    item["last_seen"] = now
    item["observations"] = int(item.get("observations", 0)) + 1

    def merge(field, values, limit=40):
        old = list(item.get(field, []))
        for value in values:
            if value is not None and value != "" and value not in old:
                old.append(value)
        item[field] = old[-limit:]

    ports = list(telemetry.get("dst_ports_sample") or [])
    if telemetry.get("destination_port") is not None:
        ports.append(telemetry.get("destination_port"))
    merge("ports", ports)

    dests = list(telemetry.get("destination_ips_sample") or [])
    if telemetry.get("primary_destination"):
        dests.append(telemetry.get("primary_destination"))
    merge("destinations", dests)

    merge("protocols", [telemetry.get("protocol")])
    services = list(telemetry.get("services_seen") or [])
    if telemetry.get("service"):
        services.append(telemetry.get("service"))
    merge("services", services)

    item["risk_level"] = prediction.get("risk_level", item.get("risk_level", "LOW"))
    item["current_stage"] = prediction.get("current_stage", item.get("current_stage", "Normal"))
    item["status"] = "NEW" if item["observations"] <= 2 else "KNOWN"

    data[key] = item
    _save(data)
    return item


def read() -> list[dict]:
    rows = list(_load().values())
    rows.sort(key=lambda x: x.get("last_seen", ""), reverse=True)
    return rows


def clear() -> None:
    if DEVICE_PATH.exists():
        DEVICE_PATH.unlink()
