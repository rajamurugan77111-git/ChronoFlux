"""Hybrid fusion of supervised ML, anomaly score and transparent signatures.

The risk formula is intentionally explicit and auditable:

    Hybrid Risk = 0.50 * ML Risk
                + 0.30 * Anomaly Risk
                + 0.20 * Signature Risk

All component signals are normalized to 0..100 before fusion.
"""
from core.intelligence import category_to_stage, context_for

ML_WEIGHT = 0.50
ANOMALY_WEIGHT = 0.30
SIGNATURE_WEIGHT = 0.20

RISK_BASE = {
    "Normal": 0,
    "Suspicious": 40,
    "Reconnaissance": 50,
    "Initial Access": 70,
    "Execution": 76,
    "Persistence": 82,
    "Lateral Movement": 84,
    "Command & Control": 90,
    "Exfiltration": 94,
    "Impact": 88,
}


def _risk_level(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def _actions(stage: str, risk: str) -> list[str]:
    common = ["Continue packet capture and retain the latest traffic windows."]
    if stage in {"Reconnaissance", "Suspicious"}:
        return [
            "Review the source IP, destination hosts and destination-port pattern.",
            "Restrict unnecessary exposed services if the activity is unauthorized.",
            *common,
        ]
    if stage == "Impact":
        return [
            "Rate-limit or isolate the traffic source in the controlled environment.",
            "Check service availability and packet-rate trends.",
            *common,
        ]
    if stage in {"Initial Access", "Execution"}:
        return [
            "Inspect the targeted service and recent authentication/application events.",
            "Preserve the relevant PCAP or flow evidence.",
            *common,
        ]
    if stage in {"Persistence", "Lateral Movement", "Command & Control", "Exfiltration"}:
        return [
            "Escalate the event for immediate incident review.",
            "Isolate the affected lab host if the activity is not expected.",
            "Preserve PCAP, process and authentication evidence.",
        ]
    if risk == "MEDIUM":
        return ["Increase monitoring sensitivity for the next windows.", *common]
    return ["Network behaviour is within the current baseline.", *common]


def fuse(ml: dict, anomaly: dict, signatures: list[dict]) -> dict:
    ml_category = ml.get("ml_stage", "Unavailable")
    ml_conf = ml.get("ml_confidence")
    attack_probability = float(ml.get("attack_probability") or 0.0)
    anomaly_score = float(anomaly.get("score", 0.0) or 0.0)

    stage = category_to_stage(ml_category) if ml_category != "Unavailable" else "Normal"
    evidence = []

    strongest = signatures[0] if signatures else None
    if strongest:
        mapped = strongest.get("mapped_stage")
        if mapped and (stage == "Normal" or strongest.get("severity") in {"HIGH", "CRITICAL"}):
            stage = mapped
        evidence.extend(sig["reason"] for sig in signatures[:3])

    if anomaly_score >= 72 and stage == "Normal":
        stage = "Suspicious"
        evidence.append(f"Normal-baseline anomaly score increased to {anomaly_score:.0f}%.")
    elif anomaly_score >= 50:
        evidence.append(f"Traffic is elevated relative to the learned normal baseline ({anomaly_score:.0f}%).")

    if not evidence:
        if stage == "Normal":
            evidence.append("No high-confidence behavioural signature is active in the latest window.")
        else:
            evidence.append(
                f"The supervised detector classified the flow as benchmark category {ml_category}, "
                f"interpreted as {stage} behaviour."
            )

    # 0..100 normalized component signals.
    if ml.get("attack_detected"):
        ml_signal = max(attack_probability * 100.0, float(RISK_BASE.get(stage, 50)))
    else:
        ml_signal = min(attack_probability * 100.0, 25.0)
    anomaly_signal = max(0.0, min(100.0, anomaly_score))
    signature_signal = max(0.0, min(100.0, float(strongest.get("score", 0) if strongest else 0)))

    risk_score = round(
        min(
            100.0,
            ML_WEIGHT * ml_signal
            + ANOMALY_WEIGHT * anomaly_signal
            + SIGNATURE_WEIGHT * signature_signal,
        ),
        1,
    )
    risk = _risk_level(risk_score)

    if strongest and stage == strongest.get("mapped_stage"):
        confidence = max(float(ml_conf or 0.0), min(0.96, 0.58 + signature_signal / 250.0))
    elif ml_conf is not None:
        confidence = float(ml_conf)
    elif anomaly_score >= 72:
        confidence = min(0.9, anomaly_score / 100)
    else:
        confidence = 0.6

    intel = context_for(stage, None if ml_category == "Unavailable" else ml_category)

    return {
        "current_stage": stage,
        "current_attack_category": None if ml_category == "Unavailable" else ml_category,
        "current_stage_confidence": round(confidence, 4),
        "risk_level": risk,
        "risk_score": risk_score,
        "risk_formula": "0.50*ML + 0.30*Anomaly + 0.20*Signature",
        "risk_components": {
            "ml": round(ml_signal, 1),
            "anomaly": round(anomaly_signal, 1),
            "signature": round(signature_signal, 1),
            "weights": {"ml": ML_WEIGHT, "anomaly": ANOMALY_WEIGHT, "signature": SIGNATURE_WEIGHT},
        },
        "anomaly_score": anomaly_score,
        "anomaly_label": anomaly.get("label", "Unavailable"),
        "ml_stage": ml_category,
        "ml_confidence": ml_conf,
        "attack_probability": ml.get("attack_probability"),
        "attack_detected": bool(stage != "Normal"),
        "signatures": signatures,
        "explanations": evidence[:4],
        "recommended_actions": _actions(stage, risk),
        **intel,
    }
