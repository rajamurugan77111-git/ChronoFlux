"""ChronoFlux private security-evidence chain.

This is a local hash-chained, append-only ledger for tamper-evident security
evidence. It provides local tamper-evident security evidence without claiming distributed consensus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "data" / "evidence"
LEDGER_PATH = EVIDENCE_DIR / "ledger.jsonl"
GENESIS_HASH = "0" * 64


def _ensure() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def read(limit: int | None = None) -> list[dict]:
    if not LEDGER_PATH.exists():
        return []

    rows: list[dict] = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

    if limit is not None:
        return rows[-max(1, int(limit)):]
    return rows


def append_event(event: dict) -> dict:
    """Append one compact security event to the evidence chain."""
    _ensure()
    chain = read()
    index = len(chain)
    previous_hash = chain[-1]["block_hash"] if chain else GENESIS_HASH

    compact_event = {
        "event_id": f"CF-EV-{index + 1:06d}",
        "event_time": event.get("event_time")
        or datetime.now(timezone.utc).isoformat(),
        "sensor_id": event.get("sensor_id", "unknown-sensor"),
        "stage": event.get("stage", "Unknown"),
        "risk_level": event.get("risk_level", "UNKNOWN"),
        "risk_score": event.get("risk_score", 0),
        "confidence": event.get("confidence"),
        "source_ip": event.get("source_ip"),
        "source_port": event.get("source_port"),
        "destination_ip": event.get("destination_ip"),
        "destination_port": event.get("destination_port"),
        "protocol": event.get("protocol"),
        "service": event.get("service"),
        "signature": event.get("signature"),
        "forecast_next": event.get("forecast_next"),
        "evidence_summary": event.get("evidence_summary", []),
    }

    event_hash = _sha256(compact_event)
    block_without_hash = {
        "index": index,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "previous_hash": previous_hash,
        "event_hash": event_hash,
        "event": compact_event,
    }
    block_hash = _sha256(block_without_hash)
    block = {**block_without_hash, "block_hash": block_hash}

    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(block, ensure_ascii=False, default=str) + "\n")

    return block


def verify_chain() -> dict:
    chain = read()
    previous_hash = GENESIS_HASH

    for expected_index, block in enumerate(chain):
        if block.get("index") != expected_index:
            return {
                "valid": False,
                "blocks": len(chain),
                "failed_index": expected_index,
                "reason": "Block index mismatch",
            }

        if block.get("previous_hash") != previous_hash:
            return {
                "valid": False,
                "blocks": len(chain),
                "failed_index": expected_index,
                "reason": "Previous-hash link mismatch",
            }

        event = block.get("event", {})
        if _sha256(event) != block.get("event_hash"):
            return {
                "valid": False,
                "blocks": len(chain),
                "failed_index": expected_index,
                "reason": "Event hash mismatch",
            }

        block_without_hash = {
            "index": block.get("index"),
            "created_at": block.get("created_at"),
            "previous_hash": block.get("previous_hash"),
            "event_hash": block.get("event_hash"),
            "event": event,
        }
        if _sha256(block_without_hash) != block.get("block_hash"):
            return {
                "valid": False,
                "blocks": len(chain),
                "failed_index": expected_index,
                "reason": "Block hash mismatch",
            }

        previous_hash = block.get("block_hash")

    latest = chain[-1] if chain else None
    return {
        "valid": True,
        "blocks": len(chain),
        "failed_index": None,
        "reason": "All event hashes and block links verified",
        "latest_block_hash": latest.get("block_hash") if latest else None,
        "latest_event_id": latest.get("event", {}).get("event_id") if latest else None,
    }


def clear() -> None:
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()
