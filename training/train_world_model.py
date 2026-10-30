"""Train LSTM world model (+ logistic regression baseline) from lab + live sessions."""
from __future__ import annotations

from pathlib import Path
import json

from core.history import read as read_history
from core.world_model import train_world_model
from data_tools.generate_attack_sessions import generate, OUT as LAB_PATH

ROOT = Path(__file__).resolve().parents[1]


def _load_lab() -> list[dict]:
    rows = generate(sessions_per_pattern=4)
    LAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LAB_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main():
    lab = _load_lab()
    live = read_history(limit=2500)
    # Prefer combining lab progression prior with any genuine live sessions.
    combined = lab + live
    print(f"Training world model on {len(combined)} windows ({len(lab)} lab + {len(live)} live)")
    result = train_world_model(combined, force=True, epochs=30)
    print(json.dumps({k: result.get(k) for k in ("trained", "ready", "samples", "sessions", "comparison_vs_logistic_regression", "model_name")}, indent=2))
    metrics = result.get("metrics") or {}
    if metrics.get("test"):
        print("World-model test:", json.dumps(metrics["test"], indent=2))
    baseline = (result.get("baseline_metrics") or {}).get("test")
    if baseline:
        print("Logistic regression test:", json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
