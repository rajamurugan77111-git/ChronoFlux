"""Pytest suite for ChronoFlux (runs offline, no server needed)."""
import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.api import app
from core.schema import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health_models_ready(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    models = r.json()["models"]
    assert models["detection"] and models["stage"] and models["anomaly"] and models["world_model"]


def test_predict_single_flow(client):
    row = pd.read_csv(ROOT / "data" / "sample_mixed_traffic.csv").iloc[[0]]
    features = json.loads(row[FEATURE_COLUMNS].to_json(orient="records"))[0]
    r = client.post("/api/predict", json={"features": features, "telemetry": {}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "current_stage" in body and "forecast" in body


def test_model_arena_leaderboard(client):
    arena = client.get("/api/models/arena").json()
    assert arena["available"] is True
    for task in ("detection", "stage"):
        t = arena["tasks"][task]
        board = t["leaderboard"]
        metric = t["ranking_metric"]
        assert board[0]["key"] == t["winner"]
        assert [e[metric] for e in board] == sorted((e[metric] for e in board), reverse=True)
        assert t["feature_importance"], "explainability missing"
    # The deployed stage model must beat the v7 Random Forest reference.
    assert arena["tasks"]["stage"]["improvement_vs_reference"] > 0


def test_demo_seed_produces_forecast(client):
    r = client.post("/api/demo/seed")
    assert r.status_code == 200
    body = r.json()
    assert body.get("windows", 0) >= 4
    assert client.get("/api/latest").json()["prediction"]["current_stage"]


@pytest.mark.parametrize("csv", ["sample_mixed_traffic.csv", "samples/cicids2018_sample.csv",
                                 "samples/ctu13_sample.csv"])
def test_offline_csv_analysis(client, csv):
    data = (ROOT / "data" / csv).read_bytes()
    r = client.post("/api/offline/analyze-csv", files={"file": (Path(csv).name, data, "text/csv")})
    assert r.status_code == 200, r.text
    assert r.json()["analyzed_rows"] > 0


def test_evidence_chain_valid(client):
    assert client.get("/api/evidence/verify").json()["valid"] is True


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "AI Model Arena" in r.text
