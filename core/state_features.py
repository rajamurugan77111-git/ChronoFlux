"""Shared network-state feature schema for ChronoFlux world-model forecasting.

State S_t is a structured feature vector covering both flow-level aggregates
and packet-level timing / flag / TTL / window signals required by the NTRO brief.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

STAGE_ORDER = [
    "Normal",
    "Suspicious",
    "Reconnaissance",
    "Initial Access",
    "Execution",
    "Persistence",
    "Lateral Movement",
    "Command & Control",
    "Exfiltration",
    "Impact",
]
STAGE_CODE = {name: idx for idx, name in enumerate(STAGE_ORDER)}
ATTACK_STAGES = {s for s in STAGE_ORDER if s not in {"Normal", "Suspicious"}}

# Human-readable labels used by attention / SHAP-style explanations.
STATE_FEATURES = [
    "log_packet_count",
    "log_packet_rate",
    "log_outbound_bytes",
    "log_inbound_bytes",
    "bidirectional_byte_ratio",
    "unique_dst_ports",
    "unique_dst_ips",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "icmp_count",
    "iat_mean",
    "iat_variance",
    "iat_max",
    "ttl_mean",
    "ttl_variance",
    "tcp_window_mean",
    "retransmit_count",
    "fragment_flag_count",
    "payload_size_mean",
    "port_scan_score",
    "anomaly",
    "risk",
    "attack_probability",
    "stage_code",
    "signature_count",
]

FEATURE_HELP = {
    "log_packet_count": "packets per window",
    "log_packet_rate": "packet rate",
    "log_outbound_bytes": "outbound bytes",
    "log_inbound_bytes": "inbound bytes",
    "bidirectional_byte_ratio": "bidirectional flow byte ratio",
    "unique_dst_ports": "destination port diversity",
    "unique_dst_ips": "destination host diversity",
    "syn_count": "TCP SYN flags",
    "ack_count": "TCP ACK flags",
    "rst_count": "TCP RST flags",
    "fin_count": "TCP FIN flags",
    "icmp_count": "ICMP activity",
    "iat_mean": "inter-arrival time mean",
    "iat_variance": "inter-arrival time variance",
    "iat_max": "inter-arrival time max",
    "ttl_mean": "TTL mean",
    "ttl_variance": "TTL variance",
    "tcp_window_mean": "TCP window size",
    "retransmit_count": "retransmission count",
    "fragment_flag_count": "IP fragment flags",
    "payload_size_mean": "payload size distribution",
    "port_scan_score": "port-scan signature score",
    "anomaly": "anomaly baseline score",
    "risk": "hybrid risk score",
    "attack_probability": "current attack probability",
    "stage_code": "current behaviour stage",
    "signature_count": "active behavioural signatures",
}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _log(value: Any) -> float:
    return math.log1p(max(0.0, _num(value)))


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def port_scan_score(telemetry: dict, features: dict) -> float:
    """Heuristic score for sequential / randomised port-access patterns."""
    unique_ports = _num(telemetry.get("unique_dst_ports", features.get("ct_dst_ltm", 0)))
    unique_ips = _num(telemetry.get("unique_dst_ips", features.get("ct_dst_src_ltm", 0)))
    syn = _num(telemetry.get("syn_count", 0))
    sample = telemetry.get("dst_ports_sample") or []
    sequential = 0.0
    if len(sample) >= 3:
        diffs = [sample[i] - sample[i - 1] for i in range(1, len(sample))]
        sequential = 1.0 if sum(1 for d in diffs if 0 < d <= 3) >= max(2, len(diffs) // 2) else 0.0
    randomised = 1.0 if unique_ports >= 15 and unique_ips <= 3 else 0.0
    return min(1.0, (unique_ports / 40.0) + (syn / 100.0) * 0.4 + 0.25 * sequential + 0.25 * randomised)


def state_vector(record: dict) -> np.ndarray:
    """Encode one traffic window as network state S_t."""
    tel = record.get("telemetry") or {}
    feat = record.get("features") or {}
    pred = record.get("prediction") or {}

    packet_count = tel.get("packet_count")
    if packet_count is None:
        packet_count = _num(feat.get("spkts")) + _num(feat.get("dpkts"))

    outbound = _num(tel.get("outbound_bytes", feat.get("sbytes", 0)))
    inbound = _num(tel.get("inbound_bytes", feat.get("dbytes", 0)))
    bidirectional = outbound / max(inbound, 1.0)

    iat_mean = _num(tel.get("iat_mean", feat.get("sinpkt", 0)))
    iat_var = _num(tel.get("iat_variance", feat.get("sjit", 0)))
    iat_max = _num(tel.get("iat_max", max(iat_mean, _num(feat.get("dinpkt", 0)))))

    ttl_mean = _num(tel.get("ttl_mean", feat.get("sttl", 0)))
    ttl_var = _num(tel.get("ttl_variance", abs(_num(feat.get("sttl", 0)) - _num(feat.get("dttl", 0)))))

    stage = str(pred.get("current_stage", "Normal"))

    return np.asarray(
        [
            _log(packet_count),
            _log(tel.get("packet_rate", feat.get("rate", 0))),
            _log(outbound),
            _log(inbound),
            min(bidirectional, 50.0) / 50.0,
            _num(tel.get("unique_dst_ports", feat.get("ct_dst_ltm", 0))),
            _num(tel.get("unique_dst_ips", feat.get("ct_dst_src_ltm", 0))),
            _num(tel.get("syn_count", 0)),
            _num(tel.get("ack_count", 0)),
            _num(tel.get("rst_count", 0)),
            _num(tel.get("fin_count", 0)),
            _num(tel.get("icmp_count", 0)),
            _log(iat_mean),
            _log(iat_var),
            _log(iat_max),
            ttl_mean / 255.0,
            _log(ttl_var),
            _num(tel.get("tcp_window_mean", feat.get("swin", 0))) / 65535.0,
            _num(tel.get("retransmit_count", feat.get("sloss", 0))),
            _num(tel.get("fragment_flag_count", 0)),
            _log(tel.get("payload_size_mean", feat.get("smean", 0))),
            port_scan_score(tel, feat),
            _num(pred.get("anomaly_score", 0)) / 100.0,
            _num(pred.get("risk_score", 0)) / 100.0,
            _num(pred.get("attack_probability", 0)),
            float(STAGE_CODE.get(stage, STAGE_CODE["Suspicious"])) / max(1, len(STAGE_ORDER) - 1),
            float(len(pred.get("signatures") or [])),
        ],
        dtype=np.float64,
    )


def stage_from_code(code: float) -> str:
    idx = int(round(code * (len(STAGE_ORDER) - 1)))
    idx = max(0, min(len(STAGE_ORDER) - 1, idx))
    return STAGE_ORDER[idx]


def infiltration_score_from_stages(stages: list[str], probs: list[float] | None = None) -> float:
    """Likelihood that the rolled-out trajectory converges to infiltration."""
    if not stages:
        return 0.0
    weights = probs or [1.0] * len(stages)
    total = 0.0
    mass = 0.0
    for stage, weight in zip(stages, weights):
        w = max(0.0, float(weight))
        mass += w
        if stage in ATTACK_STAGES:
            # Later kill-chain stages contribute more.
            depth = STAGE_CODE.get(stage, 2) / max(1, len(STAGE_ORDER) - 1)
            total += w * (0.55 + 0.45 * depth)
        elif stage == "Suspicious":
            total += w * 0.25
    return float(min(1.0, total / max(mass, 1e-9)))
