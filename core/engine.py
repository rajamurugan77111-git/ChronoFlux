"""Single entry point for ChronoFlux detection + intelligence + forecast."""
from core.detector import predict_current
from core.anomaly import score_anomaly
from core.signatures import evaluate_signatures
from core.fusion import fuse
from core.forecast import forecast


def analyse(features: dict, telemetry: dict | None = None, history: list[dict] | None = None) -> dict:
    telemetry = telemetry or {}
    ml = predict_current(features)
    anomaly = score_anomaly(features)
    signatures = evaluate_signatures(features, telemetry)
    result = fuse(ml, anomaly, signatures)

    history_rows = list(history or [])
    current_record = {
        "features": features,
        "telemetry": telemetry,
        "prediction": result,
    }
    history_rows.append(current_record)
    window_seconds = float(telemetry.get("window_seconds", 10) or 10)
    result["forecast"] = forecast(history_rows, window_seconds=window_seconds)
    return result
