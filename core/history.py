"""Small persistent live-history store used by the backend and dashboard."""
from pathlib import Path
from datetime import datetime, timezone
import json

ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = ROOT / "data" / "live"
HISTORY_PATH = LIVE_DIR / "history.jsonl"
LATEST_PATH = LIVE_DIR / "latest.json"


def _ensure():
    LIVE_DIR.mkdir(parents=True, exist_ok=True)


def append(record: dict) -> None:
    _ensure()
    item = dict(record)
    item.setdefault("received_at", datetime.now(timezone.utc).isoformat())
    with HISTORY_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    LATEST_PATH.write_text(json.dumps(item, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def latest() -> dict | None:
    if not LATEST_PATH.exists():
        return None
    try:
        return json.loads(LATEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def read(limit: int = 100) -> list[dict]:
    if not HISTORY_PATH.exists():
        return []
    try:
        lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-max(1, limit):]:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def clear() -> None:
    for path in [HISTORY_PATH, LATEST_PATH]:
        if path.exists():
            path.unlink()
