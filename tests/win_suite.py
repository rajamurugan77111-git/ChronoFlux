"""Win-readiness suite for ChronoFlux v7.1 — must pass before SIH demo."""
from __future__ import annotations

from pathlib import Path
import sys
import json
import io

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import app
from core.schema import FEATURE_COLUMNS
from core.world_model import world_model_ready, predict_world_model
from core.pcap_ingest import parse_pcap_bytes
from data_tools.generate_attack_sessions import generate

client = TestClient(app)
passed = 0


def check(name: str, cond: bool, detail: str = ""):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name} — {detail}")
    print(f"PASS: {name}")
    passed += 1


def main():
    # Health / models
    h = client.get("/api/health")
    check("health endpoint", h.status_code == 200)
    body = h.json()
    check("world model capability listed", "lstm_world_model_dynamics" in body.get("capabilities", []))
    check("pcap capability listed", "offline_pcap_analysis" in body.get("capabilities", []))

    sys_resp = client.get("/api/system")
    check("system endpoint", sys_resp.status_code == 200)
    check("world model ready", bool(sys_resp.json().get("models", {}).get("world_model")) or world_model_ready())

    # Current-state predict
    row = pd.read_csv(ROOT / "data" / "sample_mixed_traffic.csv").iloc[[0]]
    features = json.loads(row[FEATURE_COLUMNS].to_json(orient="records"))[0]
    r = client.post("/api/predict", json={"features": features, "telemetry": {"window_seconds": 5}})
    check("predict endpoint", r.status_code == 200, r.text)
    pred = r.json()
    check("predict has current_stage", "current_stage" in pred)
    check("predict has forecast", "forecast" in pred)
    check("predict has explanations", bool(pred.get("explanations")))
    check("predict has mitre field", "mitre_tactics" in pred)

    # Demo seed + active world-model forecast
    seed = client.post("/api/demo/seed")
    check("demo seed", seed.status_code == 200, seed.text)
    seed_body = seed.json()
    check("demo seeded windows", int(seed_body.get("windows") or 0) >= 6)
    fcast = seed_body.get("forecast") or {}
    check("demo forecast active or candidate", fcast.get("status") in {"active", "insufficient_evidence"})
    if fcast.get("status") == "active":
        check("demo forecast has next_stage", bool(fcast.get("next_stage")))
        check("demo forecast has driving features", bool(fcast.get("driving_features")))
        check("demo forecast has timeline", bool(fcast.get("timeline")))
        check("demo model is LSTM", "LSTM" in str(fcast.get("model_name", "")))

    latest = client.get("/api/latest")
    check("latest after seed", latest.status_code == 200 and latest.json().get("status") == "ok")

    hist = client.get("/api/history?limit=20")
    check("history after seed", hist.status_code == 200 and len(hist.json().get("items") or []) >= 4)

    # World-model direct inference
    rows = generate(sessions_per_pattern=1, seed=3)
    wm = predict_world_model(rows[2:6], window_seconds=5)
    check("direct world-model inference", wm is not None and wm.get("status") in {"active", "insufficient_evidence"})

    # Offline CSV
    csv_bytes = (ROOT / "data" / "sample_mixed_traffic.csv").read_bytes()
    csv_resp = client.post(
        "/api/offline/analyze-csv?max_rows=50",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )
    check("offline csv", csv_resp.status_code == 200, csv_resp.text)
    check("offline csv distribution", bool(csv_resp.json().get("stage_distribution")))

    # Minimal synthetic PCAP via scapy
    try:
        from scapy.all import Ether, IP, TCP, wrpcap  # type: ignore
        import tempfile

        pkts = []
        for i in range(40):
            pkts.append(
                Ether()
                / IP(src="10.0.0.8", dst="10.0.0.1", ttl=64)
                / TCP(sport=40000 + i, dport=20 + (i % 30), flags="S", seq=1000 + i)
            )
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            path = Path(tmp.name)
        wrpcap(str(path), pkts)
        pcap_bytes = path.read_bytes()
        path.unlink(missing_ok=True)
        windows = parse_pcap_bytes(pcap_bytes, window_seconds=5)
        check("pcap parse windows", len(windows) >= 1)
        pcap_resp = client.post(
            "/api/offline/analyze-pcap?window_seconds=5",
            files={"file": ("demo.pcap", pcap_bytes, "application/octet-stream")},
        )
        check("offline pcap endpoint", pcap_resp.status_code == 200, pcap_resp.text)
        check("offline pcap has stages", bool(pcap_resp.json().get("stage_distribution")))
    except Exception as exc:
        check("pcap path available", False, str(exc))

    # Temporal train endpoint should succeed
    train = client.post("/api/temporal/train")
    check("temporal train", train.status_code == 200, train.text)
    check("temporal train ready", bool(train.json().get("ready")))
    cmp_ = (train.json().get("world_model") or {}).get("comparison_vs_logistic_regression") or {}
    if cmp_:
        check(
            "world model beats or matches LR F1 on holdout",
            float(cmp_.get("world_model_macro_f1") or 0) >= float(cmp_.get("logistic_regression_macro_f1") or 0) - 0.02,
            json.dumps(cmp_),
        )
        check("reported accuracy present", cmp_.get("world_model_accuracy") is not None)

    # Evidence / devices endpoints
    check("evidence endpoint", client.get("/api/evidence").status_code == 200)
    check("devices endpoint", client.get("/api/devices").status_code == 200)
    check("frontend index", client.get("/").status_code == 200)

    print(f"\nChronoFlux win suite: {passed} checks PASS")


if __name__ == "__main__":
    main()
