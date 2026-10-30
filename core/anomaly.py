"""Normal-behaviour anomaly scoring for live traffic."""
from pathlib import Path
import json
import joblib
import pandas as pd

from core.schema import NUMERIC_FEATURES, DEFAULT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "anomaly" / "anomaly_model.joblib"
META_PATH = ROOT / "models" / "anomaly" / "anomaly_meta.json"

_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    if _meta is None and META_PATH.exists():
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))


def ready() -> bool:
    _load()
    return _model is not None


def score_anomaly(features: dict) -> dict:
    _load()
    if _model is None:
        return {
            "available": False,
            "score": 0.0,
            "label": "Unavailable",
            "raw_score": None,
        }

    row = {}
    for name in NUMERIC_FEATURES:
        value = features.get(name, DEFAULT_FEATURES[name])
        try:
            row[name] = float(value)
        except (TypeError, ValueError):
            row[name] = 0.0

    X = pd.DataFrame([row], columns=NUMERIC_FEATURES)
    raw = float(_model.decision_function(X)[0])

    # Calibration is learned from normal training traffic. A lower decision
    # score means more anomalous. Convert it to an intuitive 0..100 score.
    q_low = float((_meta or {}).get("normal_q01", -0.1))
    q_high = float((_meta or {}).get("normal_q95", 0.1))
    denom = max(q_high - q_low, 1e-9)
    pct = (q_high - raw) / denom * 100.0
    pct = max(0.0, min(100.0, pct))

    if pct >= 80:
        label = "High anomaly"
    elif pct >= 60:
        label = "Suspicious"
    elif pct >= 35:
        label = "Elevated"
    else:
        label = "Normal baseline"

    return {
        "available": True,
        "score": round(pct, 1),
        "label": label,
        "raw_score": round(raw, 6),
    }


def score_batch(frame: pd.DataFrame) -> list[dict]:
    """Vectorized anomaly scoring for offline CSV analysis."""
    _load()
    if _model is None:
        return [{"available": False, "score": 0.0, "label": "Unavailable", "raw_score": None} for _ in range(len(frame))]

    X = frame[NUMERIC_FEATURES].copy()
    for name in NUMERIC_FEATURES:
        X[name] = pd.to_numeric(X[name], errors="coerce")
    raw_scores = _model.decision_function(X)
    q_low = float((_meta or {}).get("normal_q01", -0.1))
    q_high = float((_meta or {}).get("normal_q95", 0.1))
    denom = max(q_high - q_low, 1e-9)
    out = []
    for raw in raw_scores:
        pct = (q_high - float(raw)) / denom * 100.0
        pct = max(0.0, min(100.0, pct))
        if pct >= 80:
            label = "High anomaly"
        elif pct >= 60:
            label = "Suspicious"
        elif pct >= 35:
            label = "Elevated"
        else:
            label = "Normal baseline"
        out.append({"available": True, "score": round(pct, 1), "label": label, "raw_score": round(float(raw), 6)})
    return out
