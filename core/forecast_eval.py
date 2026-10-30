"""Online verification of issued T5 forecasts against later observations.

A transition forecast is counted separately from a same-state forecast. This
lets ChronoFlux prove the stronger case:

Current = Reconnaissance -> forecast Initial Access -> later observed Initial Access.
"""
from pathlib import Path
from datetime import datetime
import json

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "live"
EVAL_PATH = EVAL_DIR / "forecast_evaluation.jsonl"


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def evaluate_previous(previous_records: list[dict], current_record: dict) -> dict | None:
    if not previous_records:
        return None

    previous = previous_records[-1]
    previous_prediction = previous.get("prediction") or {}
    forecast = previous_prediction.get("forecast") or {}
    predicted = forecast.get("next_stage")
    if not predicted or forecast.get("status") != "active":
        return None

    actual = (current_record.get("prediction") or {}).get("current_stage")
    prior_stage = previous_prediction.get("current_stage")
    if not actual or not prior_stage:
        return None

    issued = _parse_time(previous.get("timestamp") or previous.get("received_at"))
    observed = _parse_time(current_record.get("timestamp") or current_record.get("received_at"))
    lead = None
    if issued and observed:
        lead = max(0.0, (observed - issued).total_seconds())

    predicted_transition = predicted != prior_stage
    observed_transition = actual != prior_stage
    correct = predicted == actual
    correct_transition = bool(correct and predicted_transition and observed_transition)

    event = {
        "issued_at": previous.get("timestamp") or previous.get("received_at"),
        "observed_at": current_record.get("timestamp") or current_record.get("received_at"),
        "current_stage_at_issue": prior_stage,
        "predicted_stage": predicted,
        "actual_stage": actual,
        "correct": correct,
        "predicted_transition": predicted_transition,
        "observed_transition": observed_transition,
        "correct_transition": correct_transition,
        "confidence": forecast.get("confidence"),
        "lead_time_seconds": lead,
        "model": forecast.get("model_name"),
        "session_id": current_record.get("session_id") or previous.get("session_id"),
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with EVAL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")
    return event


def read(limit: int = 500) -> list[dict]:
    if not EVAL_PATH.exists():
        return []
    lines = EVAL_PATH.read_text(encoding="utf-8").splitlines()[-max(1, limit) :]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _matrix_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "labels": [],
            "confusion_matrix": [],
            "macro_f1": None,
            "macro_precision": None,
            "macro_recall": None,
            "normal_to_attack_false_positive_rate": None,
        }
    y_true = [str(r.get("actual_stage")) for r in rows]
    y_pred = [str(r.get("predicted_stage")) for r in rows]
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    normal_count = sum(1 for y in y_true if y == "Normal")
    normal_fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == "Normal" and pred != "Normal")
    return {
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "normal_to_attack_false_positive_rate": round(float(normal_fp / normal_count), 4) if normal_count else None,
    }


def summary(limit: int = 500) -> dict:
    rows = read(limit)
    total = len(rows)
    correct = sum(1 for x in rows if x.get("correct"))
    transition_rows = [x for x in rows if x.get("predicted_transition")]
    correct_transitions = [x for x in transition_rows if x.get("correct_transition")]
    transition_leads = [
        float(x["lead_time_seconds"])
        for x in correct_transitions
        if x.get("lead_time_seconds") is not None
    ]

    matrix = _matrix_metrics(rows)
    return {
        "verified_forecasts": total,
        "correct_forecasts": correct,
        "accuracy": (correct / total) if total else None,
        "next_stage_accuracy": (correct / total) if total else None,
        "verified_transition_forecasts": len(transition_rows),
        "correct_transition_forecasts": len(correct_transitions),
        "transition_accuracy": (len(correct_transitions) / len(transition_rows)) if transition_rows else None,
        "average_correct_transition_lead_time_seconds": (sum(transition_leads) / len(transition_leads)) if transition_leads else None,
        "last": rows[-1] if rows else None,
        **matrix,
    }


def clear() -> None:
    if EVAL_PATH.exists():
        EVAL_PATH.unlink()
