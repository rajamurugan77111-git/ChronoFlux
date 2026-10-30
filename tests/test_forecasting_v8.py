"""Tests for the v8 hybrid forecaster and benchmark."""
import numpy as np
from fastapi.testclient import TestClient

from backend.api import app
from core.killchain import KILL_CHAIN, KillChainPrior, blend, gated_blend, load_prior


def test_prior_learns_transitions_and_smooths():
    kp = KillChainPrior().fit([("Reconnaissance", "Initial Access")] * 5)
    p = kp.proba("Reconnaissance", KILL_CHAIN)
    assert KILL_CHAIN[int(np.argmax(p))] == "Initial Access"
    assert np.all(p > 0) and abs(p.sum() - 1) < 1e-9          # unseen transitions still possible
    # unseen current stage falls back to kill-chain ordering (advance or stay)
    q = kp.proba("Execution", KILL_CHAIN)
    assert q[KILL_CHAIN.index("Persistence")] > q[KILL_CHAIN.index("Normal")]


def test_blend_extremes_and_gate():
    a, b = np.array([0.9, 0.1]), np.array([0.1, 0.9])
    assert np.allclose(blend(a, b, 1.0), a, atol=1e-4)
    assert np.allclose(blend(a, b, 0.0), b, atol=1e-4)
    confident = gated_blend(a, b, 1.0, 0.0, 0.8)     # max(a)=0.9 >= tau -> model
    unsure = gated_blend(np.array([0.55, 0.45]), b, 1.0, 0.0, 0.8)
    assert np.argmax(confident) == 0 and np.argmax(unsure) == 1


def test_prior_roundtrip():
    loaded = load_prior()
    assert loaded is not None, "models/temporal/killchain_prior.json missing"
    prior, gate = loaded
    assert len(gate) == 3 and prior.counts


def test_benchmark_endpoint_and_hybrid_in_demo():
    c = TestClient(app)
    b = c.get("/api/forecast/benchmark").json()
    assert b["available"]
    for proto in ("A_unseen_sessions", "B_unseen_progressions"):
        res = b["protocols"][proto]["results"]
        assert set(res) == {"lstm_world_model_v7", "logistic_regression", "kill_chain_prior", "hybrid_v8"}
    # hybrid beats the plain LSTM on new attack campaigns (top-2)
    B = b["protocols"]["B_unseen_progressions"]["results"]
    assert B["hybrid_v8"]["top2_accuracy"] > B["lstm_world_model_v7"]["top2_accuracy"]
    fc = c.post("/api/demo/seed").json().get("forecast") or {}
    assert fc.get("forecaster") == "hybrid_killchain_v8"


def test_arena_reports_data_ceiling():
    a = TestClient(app).get("/api/models/arena").json()
    amb = a["tasks"]["stage"]["label_ambiguity"]
    assert amb["share_of_rows_with_conflicting_duplicates"]["Analysis"] > 0.9
    assert 0 < amb["oracle_macro_f1_upper_bound"] < 1
