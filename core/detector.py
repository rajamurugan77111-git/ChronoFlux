"""Current-state ML detector used by both live and offline modes."""
from pathlib import Path
import joblib
import pandas as pd

from core.schema import FEATURE_COLUMNS, DEFAULT_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DETECTION_MODEL_PATH = ROOT / "models" / "current_stage" / "detection_model.joblib"
STAGE_MODEL_PATH = ROOT / "models" / "current_stage" / "stage_model.joblib"

_detection_model = None
_stage_model = None

RISK_MAP = {
    "Normal": "LOW",
    "Suspicious": "MEDIUM",
    "Reconnaissance": "MEDIUM",
    "Fuzzers": "MEDIUM",
    "Analysis": "MEDIUM",
    "Exploits": "HIGH",
    "DoS": "HIGH",
    "Generic": "HIGH",
    "Initial Access": "HIGH",
    "Backdoor": "CRITICAL",
    "Shellcode": "CRITICAL",
    "Worms": "CRITICAL",
    "Command & Control": "CRITICAL",
    "Exfiltration": "CRITICAL",
}


def _load_models():
    global _detection_model, _stage_model
    if _detection_model is None and DETECTION_MODEL_PATH.exists():
        _detection_model = joblib.load(DETECTION_MODEL_PATH)
    if _stage_model is None and STAGE_MODEL_PATH.exists():
        _stage_model = joblib.load(STAGE_MODEL_PATH)


def models_ready() -> dict:
    _load_models()
    return {
        "detection": _detection_model is not None,
        "stage": _stage_model is not None,
    }


def normalize_features(features: dict) -> dict:
    row = dict(DEFAULT_FEATURES)
    for key in FEATURE_COLUMNS:
        if key in features and features[key] is not None:
            row[key] = features[key]
    return row


def _make_frame(features: dict) -> pd.DataFrame:
    row = normalize_features(features)
    return pd.DataFrame([{column: row[column] for column in FEATURE_COLUMNS}])


def predict_current(features: dict) -> dict:
    _load_models()
    if _detection_model is None or _stage_model is None:
        return {
            "ml_available": False,
            "attack_detected": False,
            "attack_probability": None,
            "ml_stage": "Unavailable",
            "ml_confidence": None,
            "ml_risk": "UNKNOWN",
        }

    X = _make_frame(features)

    attack_pred = int(_detection_model.predict(X)[0])
    attack_probs = _detection_model.predict_proba(X)[0]
    attack_classes = list(_detection_model.named_steps["classifier"].classes_)

    if 1 in attack_classes:
        attack_prob = float(attack_probs[attack_classes.index(1)])
    else:
        attack_prob = float(max(attack_probs))

    if attack_pred == 0:
        stage = "Normal"
        stage_conf = float(max(attack_probs))
    else:
        stage = str(_stage_model.predict(X)[0])
        stage_probs = _stage_model.predict_proba(X)[0]
        stage_conf = float(max(stage_probs))

    return {
        "ml_available": True,
        "attack_detected": bool(attack_pred),
        "attack_probability": round(attack_prob, 4),
        "ml_stage": stage,
        "ml_confidence": round(stage_conf, 4),
        "ml_risk": RISK_MAP.get(stage, "MEDIUM"),
    }


def predict_batch(frame: pd.DataFrame) -> list[dict]:
    """Vectorized current-stage prediction for offline CSV analysis."""
    _load_models()
    if _detection_model is None or _stage_model is None:
        return [predict_current(row) for row in frame.to_dict(orient="records")]

    X = frame[FEATURE_COLUMNS].copy()
    attack_pred = _detection_model.predict(X)
    attack_prob_matrix = _detection_model.predict_proba(X)
    attack_classes = list(_detection_model.named_steps["classifier"].classes_)
    attack_idx = attack_classes.index(1) if 1 in attack_classes else None

    stage_pred_all = _stage_model.predict(X)
    stage_prob_all = _stage_model.predict_proba(X)

    out = []
    for i in range(len(X)):
        ap = float(attack_prob_matrix[i][attack_idx]) if attack_idx is not None else float(max(attack_prob_matrix[i]))
        if int(attack_pred[i]) == 0:
            stage = "Normal"
            conf = float(max(attack_prob_matrix[i]))
        else:
            stage = str(stage_pred_all[i])
            conf = float(max(stage_prob_all[i]))
        out.append({
            "ml_available": True,
            "attack_detected": bool(int(attack_pred[i])),
            "attack_probability": round(ap, 4),
            "ml_stage": stage,
            "ml_confidence": round(conf, 4),
            "ml_risk": RISK_MAP.get(stage, "MEDIUM"),
        })
    return out
