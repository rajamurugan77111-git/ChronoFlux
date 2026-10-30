"""ChronoFlux forecasting facade.

Prefers the LSTM world model (learned P(S_t+1|history) + K-step rollout).
Falls back to the sequence Random Forest ablation when the world model is unavailable.
Never forces a hard-coded future stage when evidence is insufficient.
"""
from core.world_model import (
    predict_world_model,
    world_model_ready,
    status as world_status,
    MIN_ACCEPTED_CONFIDENCE,
    MIN_SEQUENCE_SAMPLES,
    MIN_SESSIONS,
    MIN_CLASSES,
)
from core.temporal_ml import (
    temporal_model_ready,
    predict as predict_sequence_model,
    status as temporal_status,
)


def forecast(history_records: list[dict], window_seconds: float = 10.0) -> dict:
    wm = predict_world_model(history_records, window_seconds=window_seconds)
    if wm:
        return wm

    ml = predict_sequence_model(history_records, window_seconds=window_seconds)
    if ml:
        ml["fallback_note"] = "Sequence RF ablation active because LSTM world-model weights are not ready."
        return ml

    status = world_status(history_records) if history_records is not None else {}
    temporal = temporal_status(history_records)
    observed = int(status.get("observed_windows") or temporal.get("observed_windows") or len(history_records) or 0)
    sessions = int(status.get("recording_sessions") or temporal.get("recording_sessions") or 0)
    samples = int(status.get("available_transition_samples") or (temporal.get("available_samples") or {}).get("1", 0) or 0)

    ready = world_model_ready()
    if ready and observed < 4:
        evidence = (
            "LSTM world-model weights are loaded. Waiting for 4 chronological windows "
            f"(T1–T4) before issuing a forecast (have {observed}). "
            "Click Run demo or upload CSV/PCAP if no traffic windows are loaded yet."
        )
    elif ready:
        evidence = (
            "LSTM world-model weights are loaded, but this sequence is not confident enough "
            f"to issue a forced next stage yet ({observed} windows, {sessions} sessions)."
        )
    else:
        evidence = (
            "World-model weights are not available yet. ChronoFlux needs chronological "
            f"transition evidence (currently {observed} windows, {sessions} sessions, "
            f"{samples} transition samples; target ≥{MIN_SEQUENCE_SAMPLES} samples, "
            f"≥{MIN_SESSIONS} sessions, ≥{MIN_CLASSES} behaviour classes). "
            "Use Forecasting → Train / refresh world model."
        )
    return {
        "status": "insufficient_evidence",
        "mode": "lstm_world_model",
        "model_name": "LSTM World Model",
        "next_stage": None,
        "confidence": None,
        "infiltration_probability": None,
        "progression": [],
        "horizons": [],
        "timeline": [],
        "driving_features": [],
        "evidence": evidence,
        "window_count": min(4, len(history_records)),
        "minimum_confidence": MIN_ACCEPTED_CONFIDENCE,
        "training_status": status or temporal,
        "world_model_ready": ready,
        "sequence_rf_ready": temporal_model_ready(),
    }
