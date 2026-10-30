"""Generate curated chronological attack-progression sessions for world-model training.

These are labelled lab-style trajectories (Normal → Recon → Access → …), not
UNSW row reshuffles. They teach P(S_t+1|S_t) transition dynamics for offline demos.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import random

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "world_model_sessions.jsonl"

PROGRESSIONS = [
    ["Normal", "Normal", "Suspicious", "Reconnaissance", "Reconnaissance", "Initial Access", "Execution"],
    ["Normal", "Suspicious", "Reconnaissance", "Initial Access", "Persistence", "Command & Control"],
    ["Normal", "Normal", "Reconnaissance", "Reconnaissance", "Lateral Movement", "Exfiltration"],
    ["Normal", "Suspicious", "Reconnaissance", "Impact", "Impact"],
    ["Normal", "Normal", "Normal", "Suspicious", "Reconnaissance", "Initial Access", "Command & Control", "Exfiltration"],
    ["Normal", "Reconnaissance", "Initial Access", "Execution", "Persistence", "Lateral Movement"],
    ["Normal", "Normal", "Suspicious", "Suspicious", "Reconnaissance", "Reconnaissance"],
    ["Normal", "Suspicious", "Initial Access", "Execution", "Impact"],
]


def _stage_profile(stage: str, rng: random.Random) -> dict:
    base = {
        "Normal": dict(packets=(40, 120), rate=(8, 30), ports=(1, 4), ips=(1, 3), syn=(0, 4), ack=(10, 40), rst=(0, 2), fin=(0, 2), icmp=(0, 2), out=(800, 8000), inn=(800, 8000), anomaly=(5, 20), risk=(0, 18), attack_p=(0.0, 0.08), sigs=0, scan=0.05),
        "Suspicious": dict(packets=(80, 200), rate=(20, 60), ports=(4, 10), ips=(2, 6), syn=(5, 20), ack=(10, 40), rst=(3, 12), fin=(0, 4), icmp=(0, 8), out=(2_000, 20_000), inn=(1_000, 10_000), anomaly=(35, 55), risk=(25, 45), attack_p=(0.15, 0.35), sigs=1, scan=0.25),
        "Reconnaissance": dict(packets=(120, 400), rate=(40, 120), ports=(12, 40), ips=(1, 5), syn=(20, 90), ack=(5, 25), rst=(5, 20), fin=(0, 5), icmp=(0, 15), out=(3_000, 40_000), inn=(500, 8_000), anomaly=(50, 70), risk=(45, 65), attack_p=(0.4, 0.7), sigs=2, scan=0.75),
        "Initial Access": dict(packets=(100, 280), rate=(30, 90), ports=(3, 12), ips=(1, 3), syn=(8, 30), ack=(20, 60), rst=(2, 10), fin=(1, 6), icmp=(0, 4), out=(8_000, 80_000), inn=(2_000, 20_000), anomaly=(55, 75), risk=(60, 78), attack_p=(0.55, 0.85), sigs=2, scan=0.35),
        "Execution": dict(packets=(90, 250), rate=(25, 80), ports=(2, 8), ips=(1, 3), syn=(4, 18), ack=(25, 70), rst=(1, 8), fin=(1, 5), icmp=(0, 3), out=(10_000, 120_000), inn=(3_000, 40_000), anomaly=(60, 80), risk=(70, 85), attack_p=(0.6, 0.9), sigs=2, scan=0.2),
        "Persistence": dict(packets=(70, 180), rate=(15, 50), ports=(2, 6), ips=(1, 2), syn=(2, 10), ack=(20, 50), rst=(0, 4), fin=(0, 3), icmp=(0, 2), out=(5_000, 60_000), inn=(2_000, 25_000), anomaly=(55, 75), risk=(75, 88), attack_p=(0.55, 0.85), sigs=1, scan=0.1),
        "Lateral Movement": dict(packets=(110, 320), rate=(35, 100), ports=(6, 20), ips=(4, 15), syn=(10, 40), ack=(20, 60), rst=(3, 12), fin=(1, 5), icmp=(0, 6), out=(12_000, 150_000), inn=(4_000, 40_000), anomaly=(60, 82), risk=(78, 90), attack_p=(0.65, 0.92), sigs=2, scan=0.4),
        "Command & Control": dict(packets=(60, 160), rate=(10, 40), ports=(1, 4), ips=(1, 2), syn=(1, 6), ack=(15, 45), rst=(0, 3), fin=(0, 2), icmp=(0, 2), out=(4_000, 50_000), inn=(1_000, 20_000), anomaly=(50, 72), risk=(82, 94), attack_p=(0.6, 0.9), sigs=1, scan=0.05),
        "Exfiltration": dict(packets=(80, 220), rate=(20, 70), ports=(1, 5), ips=(1, 2), syn=(1, 8), ack=(20, 60), rst=(0, 3), fin=(0, 3), icmp=(0, 2), out=(500_000, 3_000_000), inn=(1_000, 20_000), anomaly=(55, 78), risk=(88, 98), attack_p=(0.7, 0.95), sigs=2, scan=0.05),
        "Impact": dict(packets=(300, 900), rate=(120, 500), ports=(1, 8), ips=(1, 6), syn=(80, 300), ack=(10, 40), rst=(10, 40), fin=(0, 5), icmp=(50, 250), out=(20_000, 200_000), inn=(5_000, 50_000), anomaly=(70, 95), risk=(80, 100), attack_p=(0.75, 0.98), sigs=3, scan=0.3),
    }
    p = base[stage]

    def pick(bounds):
        return rng.uniform(*bounds)

    packets = int(pick(p["packets"]))
    syn = int(pick(p["syn"]))
    ports = int(pick(p["ports"]))
    iat_mean = rng.uniform(1.0, 40.0) if stage != "Impact" else rng.uniform(0.2, 5.0)
    return {
        "packet_count": packets,
        "packet_rate": pick(p["rate"]),
        "outbound_bytes": pick(p["out"]),
        "inbound_bytes": pick(p["inn"]),
        "unique_dst_ports": ports,
        "unique_dst_ips": int(pick(p["ips"])),
        "syn_count": syn,
        "ack_count": int(pick(p["ack"])),
        "rst_count": int(pick(p["rst"])),
        "fin_count": int(pick(p["fin"])),
        "icmp_count": int(pick(p["icmp"])),
        "iat_mean": iat_mean,
        "iat_variance": iat_mean * rng.uniform(0.2, 2.5),
        "iat_max": iat_mean * rng.uniform(2.0, 8.0),
        "ttl_mean": rng.choice([64, 128, 255]),
        "ttl_variance": rng.uniform(0, 20),
        "tcp_window_mean": rng.uniform(1024, 65535),
        "retransmit_count": int(rng.uniform(0, 8 if stage in {"Impact", "Initial Access"} else 3)),
        "fragment_flag_count": int(rng.uniform(0, 4 if stage == "Impact" else 1)),
        "payload_size_mean": rng.uniform(40, 900),
        "dst_ports_sample": list(range(20, 20 + ports)),
        "anomaly_score": pick(p["anomaly"]),
        "risk_score": pick(p["risk"]),
        "attack_probability": pick(p["attack_p"]),
        "signature_count": p["sigs"],
        "port_scan": p["scan"],
        "source_ip": f"10.0.{rng.randint(0,5)}.{rng.randint(2,200)}",
        "window_seconds": 5.0,
    }


def _record(stage: str, ts: datetime, session_id: str, rng: random.Random) -> dict:
    tel = _stage_profile(stage, rng)
    features = {
        "dur": 5.0,
        "proto": "tcp",
        "service": "http" if stage != "Impact" else "-",
        "state": "CON",
        "spkts": tel["packet_count"],
        "dpkts": max(1, tel["packet_count"] // 3),
        "sbytes": tel["outbound_bytes"],
        "dbytes": tel["inbound_bytes"],
        "rate": tel["packet_rate"],
        "sttl": tel["ttl_mean"],
        "dttl": tel["ttl_mean"],
        "sload": tel["outbound_bytes"],
        "dload": tel["inbound_bytes"],
        "sloss": tel["retransmit_count"],
        "dloss": 0,
        "sinpkt": tel["iat_mean"],
        "dinpkt": tel["iat_mean"],
        "sjit": tel["iat_variance"],
        "djit": tel["iat_variance"],
        "swin": tel["tcp_window_mean"],
        "dwin": tel["tcp_window_mean"],
        "tcprtt": 0.01,
        "synack": 0.005,
        "ackdat": 0.005,
        "smean": tel["payload_size_mean"],
        "dmean": tel["payload_size_mean"] / 2,
        "ct_srv_src": tel["unique_dst_ports"],
        "ct_state_ttl": 1,
        "ct_dst_ltm": tel["unique_dst_ports"],
        "ct_src_dport_ltm": tel["unique_dst_ports"],
        "ct_dst_sport_ltm": 2,
        "ct_dst_src_ltm": tel["unique_dst_ips"],
        "ct_src_ltm": 1,
        "ct_srv_dst": tel["unique_dst_ports"],
    }
    sigs = [{"name": "PORT_FANOUT", "reason": "Port diversity elevated"}] * int(tel["signature_count"])
    prediction = {
        "current_stage": stage,
        "anomaly_score": tel["anomaly_score"],
        "risk_score": tel["risk_score"],
        "attack_probability": tel["attack_probability"],
        "signatures": sigs,
        "mitre_tactics": [],
    }
    telemetry = {k: v for k, v in tel.items() if k not in {"anomaly_score", "risk_score", "attack_probability", "signature_count", "port_scan"}}
    telemetry["protocol"] = "tcp"
    telemetry["service"] = features["service"]
    return {
        "sensor_id": "lab-generator",
        "session_id": session_id,
        "timestamp": ts.isoformat(),
        "features": features,
        "telemetry": telemetry,
        "prediction": prediction,
    }


def generate(sessions_per_pattern: int = 4, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = []
    sid = 1
    for pattern in PROGRESSIONS:
        for _ in range(sessions_per_pattern):
            session_id = f"lab-session-{sid:03d}"
            t = start + timedelta(days=sid)
            # pad with extra normals so each session has enough windows
            stages = ["Normal", "Normal"] + pattern + [pattern[-1]]
            for stage in stages:
                rows.append(_record(stage, t, session_id, rng))
                t += timedelta(seconds=5)
            sid += 1
    return rows


def generate_normal(windows: int = 8, seed: int = 21) -> list[dict]:
    """All-Normal chronological session for a no-attack / baseline demo."""
    rng = random.Random(seed)
    start = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    session_id = "lab-session-normal"
    rows = []
    t = start
    for _ in range(max(4, int(windows))):
        row = _record("Normal", t, session_id, rng)
        # Keep normal demo on a stable internal host talking to the gateway.
        row["telemetry"]["source_ip"] = "10.0.2.20"
        row["telemetry"]["primary_destination"] = "10.0.2.1"
        row["telemetry"]["destination_port"] = 443
        row["telemetry"]["source_port"] = 50000 + (_ % 40)
        rows.append(row)
        t += timedelta(seconds=5)
    return rows


def main():
    rows = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} windows across curated lab sessions → {OUT}")


if __name__ == "__main__":
    main()
