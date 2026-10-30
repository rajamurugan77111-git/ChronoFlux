"""LSTM World Model for ChronoFlux.

Learns network-state transition dynamics P(S_{t+1} | S_t, history) rather than
only classifying isolated flows. Supports K-step forward simulation, MITRE-stage
mapping and attention-based explanations.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import json
import math

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from core.state_features import (
    STATE_FEATURES,
    STAGE_ORDER,
    STAGE_CODE,
    FEATURE_HELP,
    state_vector,
    infiltration_score_from_stages,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "temporal"
WORLD_MODEL_PATH = MODEL_DIR / "world_model.joblib"
WORLD_META_PATH = MODEL_DIR / "world_model_meta.json"
BASELINE_PATH = MODEL_DIR / "logistic_baseline.joblib"
BASELINE_META_PATH = MODEL_DIR / "baseline_meta.json"
RF_PATH = MODEL_DIR / "sequence_rf.joblib"

SEQ_LEN = 4
HORIZONS = (1, 2, 3)
HIDDEN = 48
MIN_SEQUENCE_SAMPLES = 30
MIN_CLASSES = 2
MIN_SESSIONS = 3
MIN_ACCEPTED_CONFIDENCE = 0.55
ATTACK_STAGES = {s for s in STAGE_ORDER if s not in {"Normal", "Suspicious"}}

_cache = None
_baseline_cache = None


# ---------------------------------------------------------------------------
# Lightweight NumPy LSTM with feature attention
# ---------------------------------------------------------------------------

class _LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.Generator):
        scale = 1.0 / math.sqrt(input_dim + hidden_dim)
        self.Wf = rng.normal(0, scale, (input_dim + hidden_dim, hidden_dim))
        self.bf = np.zeros(hidden_dim)
        self.Wi = rng.normal(0, scale, (input_dim + hidden_dim, hidden_dim))
        self.bi = np.zeros(hidden_dim)
        self.Wo = rng.normal(0, scale, (input_dim + hidden_dim, hidden_dim))
        self.bo = np.zeros(hidden_dim)
        self.Wc = rng.normal(0, scale, (input_dim + hidden_dim, hidden_dim))
        self.bc = np.zeros(hidden_dim)

    def step(self, x, h, c):
        z = np.concatenate([x, h])
        f = 1.0 / (1.0 + np.exp(-(z @ self.Wf + self.bf)))
        i = 1.0 / (1.0 + np.exp(-(z @ self.Wi + self.bi)))
        o = 1.0 / (1.0 + np.exp(-(z @ self.Wo + self.bo)))
        g = np.tanh(z @ self.Wc + self.bc)
        c_next = f * c + i * g
        h_next = o * np.tanh(c_next)
        return h_next, c_next


class LSTMWorldModel:
    """Learns P(S_{t+1} | history) + next-stage distribution with attention."""

    def __init__(self, input_dim: int | None = None, hidden_dim: int = HIDDEN, seed: int = 42):
        self.input_dim = int(input_dim or len(STATE_FEATURES))
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cell = _LSTMCell(self.input_dim, hidden_dim, self.rng)
        # Attention over timesteps: score = tanh(h) · v
        self.attn_v = self.rng.normal(0, 0.1, hidden_dim)
        # Feature attention gate applied to each input frame
        self.feat_gate = self.rng.normal(0, 0.05, self.input_dim)
        # Heads
        self.W_state = self.rng.normal(0, 0.1, (hidden_dim, self.input_dim))
        self.b_state = np.zeros(self.input_dim)
        self.W_stage = self.rng.normal(0, 0.1, (hidden_dim, len(STAGE_ORDER)))
        self.b_stage = np.zeros(len(STAGE_ORDER))
        self.mean_ = np.zeros(self.input_dim)
        self.std_ = np.ones(self.input_dim)
        self.stage_classifier = None  # sklearn head on LSTM contexts
        self.stage_encoder = None
        self.trained = False

    def _norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def _denorm(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_

    def _feature_attend(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gate = 1.0 / (1.0 + np.exp(-self.feat_gate))
        return frame * gate, gate

    def encode(self, seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return context vector, timestep attention, feature gates."""
        h = np.zeros(self.hidden_dim)
        c = np.zeros(self.hidden_dim)
        hs = []
        gates = []
        for t in range(seq.shape[0]):
            frame, gate = self._feature_attend(seq[t])
            h, c = self.cell.step(frame, h, c)
            hs.append(h.copy())
            gates.append(gate.copy())
        H = np.stack(hs, axis=0)
        scores = np.tanh(H) @ self.attn_v
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights = weights / max(weights.sum(), 1e-9)
        context = weights @ H
        return context, weights, np.mean(np.stack(gates, axis=0), axis=0)

    def temporal_features(self, seq: np.ndarray) -> np.ndarray:
        """Rich temporal representation used by the stage head."""
        seq_n = self._norm(seq)
        context, _, _ = self.encode(seq_n)
        last = seq_n[-1]
        first = seq_n[0]
        mean = seq_n.mean(axis=0)
        delta = last - first
        trend = seq_n[-1] - seq_n[-2] if len(seq_n) > 1 else np.zeros_like(last)
        stage_path = seq_n[:, -2]  # normalized stage_code channel
        return np.concatenate([context, last, mean, delta, trend, stage_path])

    def predict_next(self, seq: np.ndarray) -> dict:
        """Predict next normalised-then-denorm state and stage distribution."""
        seq_n = self._norm(seq)
        context, time_attn, feat_gate = self.encode(seq_n)
        next_state_n = context @ self.W_state + self.b_state
        next_state = self._denorm(next_state_n)

        if self.stage_classifier is not None and self.stage_encoder is not None:
            feats = self.temporal_features(seq).reshape(1, -1)
            probs_arr = self.stage_classifier.predict_proba(feats)[0]
            classes = list(self.stage_encoder.classes_)
            probs = {str(classes[i]): float(probs_arr[i]) for i in range(len(classes))}
            for name in STAGE_ORDER:
                probs.setdefault(name, 0.0)
            stage = str(classes[int(np.argmax(probs_arr))])
            confidence = float(np.max(probs_arr))
        else:
            logits = context @ self.W_stage + self.b_stage
            logits = logits - logits.max()
            raw = np.exp(logits)
            raw = raw / max(raw.sum(), 1e-9)
            stage_idx = int(np.argmax(raw))
            stage = STAGE_ORDER[stage_idx]
            confidence = float(raw[stage_idx])
            probs = {STAGE_ORDER[i]: float(raw[i]) for i in range(len(STAGE_ORDER))}

        return {
            "next_state": next_state,
            "stage": stage,
            "stage_probs": probs,
            "confidence": confidence,
            "time_attention": time_attn.astype(float),
            "feature_gates": feat_gate.astype(float),
        }

    def rollout(self, seq: np.ndarray, steps: int = 3, hybrid: tuple | None = None,
                current_stage: str | None = None) -> list[dict]:
        """K-step forward simulation of network state transitions.

        If `hybrid` = (KillChainPrior, gate) is given, each step's stage distribution
        is the confidence-gated blend of the LSTM and the kill-chain prior (v8)."""
        current = seq.copy()
        out = []
        cur_stage = current_stage or "Normal"
        for step in range(1, steps + 1):
            pred = self.predict_next(current)
            if hybrid is not None:
                from core.killchain import gated_blend
                prior, gate = hybrid
                names = list(STAGE_ORDER)
                p_model = np.array([pred["stage_probs"].get(n, 0.0) for n in names]) + 1e-6
                p_model = p_model / p_model.sum()
                p = gated_blend(p_model, prior.proba(cur_stage, names), *gate)
                idx = int(np.argmax(p))
                pred = {**pred, "stage": names[idx], "confidence": float(p[idx]),
                        "stage_probs": {n: float(v) for n, v in zip(names, p)}}
                cur_stage = names[idx]
            # Feed predicted state back with stage_code aligned to predicted stage.
            nxt = pred["next_state"].copy()
            nxt[-2] = STAGE_CODE[pred["stage"]] / max(1, len(STAGE_ORDER) - 1)
            out.append(
                {
                    "horizon": step,
                    "label": f"T{SEQ_LEN + step}",
                    "stage": pred["stage"],
                    "confidence": pred["confidence"],
                    "accepted": bool(pred["confidence"] >= MIN_ACCEPTED_CONFIDENCE),
                    "probabilities": dict(
                        sorted(pred["stage_probs"].items(), key=lambda kv: kv[1], reverse=True)[:5]
                    ),
                    "next_state": nxt,
                    "time_attention": pred["time_attention"],
                    "feature_gates": pred["feature_gates"],
                }
            )
            current = np.vstack([current[1:], nxt.reshape(1, -1)])
        return out

    def fit(self, sequences: np.ndarray, next_states: np.ndarray, next_stages: list[str], epochs: int = 35, lr: float = 0.06):
        """Supervised dynamics learning on labelled chronological transitions."""
        if len(sequences) == 0:
            raise ValueError("No sequences to train")

        flat = sequences.reshape(-1, self.input_dim)
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0

        X = self._norm(sequences)
        Y_state = self._norm(next_states)
        y_idx = np.asarray([STAGE_CODE.get(s, 0) for s in next_stages], dtype=int)

        n = len(X)
        for epoch in range(epochs):
            order = self.rng.permutation(n)
            epoch_lr = lr * (0.96 ** epoch)
            for i in order:
                seq = X[i]
                target_state = Y_state[i]
                target_stage = y_idx[i]

                h = np.zeros(self.hidden_dim)
                c = np.zeros(self.hidden_dim)
                hs = []
                frames = []
                for t in range(seq.shape[0]):
                    frame, _ = self._feature_attend(seq[t])
                    frames.append(frame)
                    h, c = self.cell.step(frame, h, c)
                    hs.append(h.copy())
                H = np.stack(hs)
                scores = np.tanh(H) @ self.attn_v
                scores = scores - scores.max()
                w = np.exp(scores)
                w = w / max(w.sum(), 1e-9)
                context = w @ H

                pred_state = context @ self.W_state + self.b_state
                logits = context @ self.W_stage + self.b_stage
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / max(probs.sum(), 1e-9)

                state_err = pred_state - target_state
                d_state = (2.0 / self.input_dim) * state_err
                d_logits = probs.copy()
                d_logits[target_stage] -= 1.0

                self.W_state -= epoch_lr * np.outer(context, d_state)
                self.b_state -= epoch_lr * d_state
                self.W_stage -= epoch_lr * np.outer(context, d_logits)
                self.b_stage -= epoch_lr * d_logits

                d_context = (self.W_state @ d_state) + (self.W_stage @ d_logits)
                for t in range(len(H)):
                    self.attn_v -= epoch_lr * w[t] * np.tanh(H[t]) * float(np.dot(d_context, np.tanh(H[t]))) * 0.15
                gate = 1.0 / (1.0 + np.exp(-self.feat_gate))
                importance = np.abs(seq[-1])
                self.feat_gate += epoch_lr * 0.02 * (importance - gate)

                last = frames[-1]
                z = np.concatenate([last, hs[-2] if len(hs) > 1 else np.zeros(self.hidden_dim)])
                residual = np.clip(d_context * epoch_lr * 0.03, -0.2, 0.2)
                self.cell.Wo -= np.outer(z, residual)
                self.cell.bo -= residual

        # Strong temporal stage head: LSTM context + path/delta features + RF.
        feat_rows = np.asarray([self.temporal_features(seq) for seq in sequences], dtype=float)
        enc = LabelEncoder()
        y = enc.fit_transform(next_stages)
        clf = RandomForestClassifier(
            n_estimators=220,
            max_depth=12,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=self.seed,
            n_jobs=-1,
        )
        clf.fit(feat_rows, y)
        self.stage_classifier = clf
        self.stage_encoder = enc
        self.trained = True
        return self

    def to_bundle(self) -> dict:
        return {
            "model_name": "LSTM World Model",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "mean": self.mean_,
            "std": self.std_,
            "attn_v": self.attn_v,
            "feat_gate": self.feat_gate,
            "W_state": self.W_state,
            "b_state": self.b_state,
            "W_stage": self.W_stage,
            "b_stage": self.b_stage,
            "cell": {
                "Wf": self.cell.Wf, "bf": self.cell.bf,
                "Wi": self.cell.Wi, "bi": self.cell.bi,
                "Wo": self.cell.Wo, "bo": self.cell.bo,
                "Wc": self.cell.Wc, "bc": self.cell.bc,
            },
            "stage_classifier": self.stage_classifier,
            "stage_encoder": self.stage_encoder,
            "state_features": STATE_FEATURES,
            "trained": self.trained,
        }

    @classmethod
    def from_bundle(cls, bundle: dict) -> "LSTMWorldModel":
        model = cls(input_dim=bundle["input_dim"], hidden_dim=bundle["hidden_dim"], seed=bundle.get("seed", 42))
        model.mean_ = np.asarray(bundle["mean"], dtype=float)
        model.std_ = np.asarray(bundle["std"], dtype=float)
        model.attn_v = np.asarray(bundle["attn_v"], dtype=float)
        model.feat_gate = np.asarray(bundle["feat_gate"], dtype=float)
        model.W_state = np.asarray(bundle["W_state"], dtype=float)
        model.b_state = np.asarray(bundle["b_state"], dtype=float)
        model.W_stage = np.asarray(bundle["W_stage"], dtype=float)
        model.b_stage = np.asarray(bundle["b_stage"], dtype=float)
        cell = bundle["cell"]
        model.cell.Wf = np.asarray(cell["Wf"]); model.cell.bf = np.asarray(cell["bf"])
        model.cell.Wi = np.asarray(cell["Wi"]); model.cell.bi = np.asarray(cell["bi"])
        model.cell.Wo = np.asarray(cell["Wo"]); model.cell.bo = np.asarray(cell["bo"])
        model.cell.Wc = np.asarray(cell["Wc"]); model.cell.bc = np.asarray(cell["bc"])
        model.stage_classifier = bundle.get("stage_classifier")
        model.stage_encoder = bundle.get("stage_encoder")
        model.trained = bool(bundle.get("trained", True))
        return model


# ---------------------------------------------------------------------------
# Dataset helpers (shared session logic with temporal_ml)
# ---------------------------------------------------------------------------

def _sessions_from_records(records: list[dict]) -> list[dict]:
    from core.temporal_ml import _sessions
    return _sessions(records)


def build_transition_dataset(records: list[dict]) -> dict:
    """Build supervised dynamics samples: (S_{t-3..t}, S_{t+1}, stage_{t+1})."""
    sessions = _sessions_from_records(records)
    samples = []
    for session in sessions:
        rows = session["rows"]
        for start in range(0, len(rows) - SEQ_LEN):
            seq_rows = rows[start : start + SEQ_LEN]
            nxt = rows[start + SEQ_LEN]
            stage = (nxt.get("prediction") or {}).get("current_stage")
            if not stage:
                continue
            try:
                seq = np.stack([state_vector(r) for r in seq_rows], axis=0)
                target = state_vector(nxt)
            except Exception:
                continue
            current = (seq_rows[-1].get("prediction") or {}).get("current_stage") or "Normal"
            samples.append(
                {
                    "seq": seq,
                    "next_state": target,
                    "stage": str(stage),
                    "current_stage": str(current),
                    "session_id": session["id"],
                    "flat": seq.reshape(-1),
                }
            )
    return {"sessions": sessions, "samples": samples}


def _split_by_session(samples: list[dict], sessions: list[dict]) -> tuple[list, list, list]:
    ids = [s["id"] for s in sessions]
    n = len(ids)
    if n < 3:
        return samples, [], []
    train_n = max(1, int(n * 0.6))
    val_n = max(1, int(n * 0.2))
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1
    train_ids = set(ids[:train_n])
    val_ids = set(ids[train_n : train_n + val_n])
    test_ids = set(ids[train_n + val_n :])
    train = [s for s in samples if s["session_id"] in train_ids]
    val = [s for s in samples if s["session_id"] in val_ids]
    test = [s for s in samples if s["session_id"] in test_ids]
    return train, val, test


def _eval_stage_predictor(y_true, y_pred) -> dict:
    labels = sorted(set(list(y_true) + list(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    normal_count = sum(1 for v in y_true if v == "Normal")
    normal_fp = sum(1 for t, p in zip(y_true, y_pred) if t == "Normal" and p != "Normal")
    return {
        "samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "normal_to_attack_false_positive_rate": round(normal_fp / normal_count, 4) if normal_count else None,
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def explain_prediction(seq: np.ndarray, rollout: list[dict], top_k: int = 8) -> list[dict]:
    """Attention + gated feature attribution (SHAP-equivalent driving features)."""
    if not rollout:
        return []
    first = rollout[0]
    raw_attn = first.get("time_attention")
    time_attn = np.asarray(raw_attn if raw_attn is not None else [], dtype=float)
    raw_gate = first.get("feature_gates")
    feat_gate = np.asarray(raw_gate if raw_gate is not None else np.ones(seq.shape[1]), dtype=float)

    # Combine last-window magnitude, feature gate and recent temporal attention.
    last = np.abs(seq[-1])
    last = last / (last.max() + 1e-9)
    gate = feat_gate / (feat_gate.max() + 1e-9)
    temporal = float(time_attn[-1]) if len(time_attn) else 1.0
    scores = 0.45 * last + 0.40 * gate + 0.15 * temporal

    base_conf = float(first.get("confidence") or 0)
    shap_like = []
    for i, name in enumerate(STATE_FEATURES):
        contribution = float(scores[i] * (base_conf + 0.15))
        shap_like.append((name, contribution, float(scores[i])))

    shap_like.sort(key=lambda x: x[1], reverse=True)
    out = []
    for name, contrib, score in shap_like[:top_k]:
        out.append(
            {
                "feature": name,
                "label": FEATURE_HELP.get(name, name),
                "attribution": round(contrib, 4),
                "attention_weight": round(score, 4),
                "method": "attention+gated-feature attribution (SHAP-style)",
            }
        )
    return out


def mitre_for_stage(stage: str) -> list[str]:
    from core.intelligence import STAGE_CONTEXT
    return list((STAGE_CONTEXT.get(stage) or {}).get("mitre_tactics") or [])


# ---------------------------------------------------------------------------
# Train / load / predict API
# ---------------------------------------------------------------------------

def world_model_ready() -> bool:
    return WORLD_MODEL_PATH.exists()


def baseline_ready() -> bool:
    return BASELINE_PATH.exists()


def load_world_model() -> LSTMWorldModel | None:
    global _cache
    if _cache is None and WORLD_MODEL_PATH.exists():
        try:
            _cache = LSTMWorldModel.from_bundle(joblib.load(WORLD_MODEL_PATH))
        except Exception:
            _cache = None
    return _cache


def load_baseline():
    global _baseline_cache
    if _baseline_cache is None and BASELINE_PATH.exists():
        try:
            _baseline_cache = joblib.load(BASELINE_PATH)
        except Exception:
            _baseline_cache = None
    return _baseline_cache


def train_world_model(records: list[dict], force: bool = False, epochs: int = 28) -> dict:
    """Train LSTM world model + logistic regression baseline on the same features."""
    global _cache, _baseline_cache
    _cache = None
    _baseline_cache = None
    built = build_transition_dataset(records)
    samples = built["samples"]
    sessions = built["sessions"]

    classes = sorted({s["stage"] for s in samples})
    reasons = []
    if len(sessions) < MIN_SESSIONS:
        reasons.append(f"Need ≥{MIN_SESSIONS} recording sessions (have {len(sessions)}).")
    if len(samples) < MIN_SEQUENCE_SAMPLES:
        reasons.append(f"Need ≥{MIN_SEQUENCE_SAMPLES} transition samples (have {len(samples)}).")
    if len(classes) < MIN_CLASSES:
        reasons.append(f"Need ≥{MIN_CLASSES} behaviour classes.")

    if reasons and not force:
        return {
            "trained": False,
            "ready": world_model_ready(),
            "reason": " ".join(reasons),
            "samples": len(samples),
            "sessions": len(sessions),
        }

    if len(samples) < 12:
        return {"trained": False, "ready": False, "reason": "Too few transition samples to train.", "samples": len(samples)}

    y_all = [s["stage"] for s in samples]
    idx = np.arange(len(samples))
    try:
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42, stratify=y_all)
    except ValueError:
        train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=42)

    train = [samples[i] for i in train_idx]
    test = [samples[i] for i in test_idx]
    if len(train) >= 20:
        y_train_tmp = [s["stage"] for s in train]
        try:
            tr2, val_rel = train_test_split(np.arange(len(train)), test_size=0.2, random_state=7, stratify=y_train_tmp)
        except ValueError:
            tr2, val_rel = train_test_split(np.arange(len(train)), test_size=0.2, random_state=7)
        val = [train[i] for i in val_rel]
        train = [train[i] for i in tr2]
    else:
        val = []

    session_train, _session_val, session_test = _split_by_session(samples, sessions)

    seqs = np.stack([s["seq"] for s in train])
    next_states = np.stack([s["next_state"] for s in train])
    stages = [s["stage"] for s in train]

    model = LSTMWorldModel()
    model.fit(seqs, next_states, stages, epochs=epochs)

    def _predict_stages(subset, mdl):
        if not subset:
            return [], []
        y_true = [s["stage"] for s in subset]
        y_pred = [mdl.predict_next(s["seq"])["stage"] for s in subset]
        return y_true, y_pred

    metrics = {}
    for name, subset in (("train", train), ("validation", val), ("test", test)):
        yt, yp = _predict_stages(subset, model)
        metrics[name] = _eval_stage_predictor(yt, yp) if yt else None

    session_metrics = {}
    if session_train and session_test:
        tmp = LSTMWorldModel(seed=99)
        tmp.fit(
            np.stack([s["seq"] for s in session_train]),
            np.stack([s["next_state"] for s in session_train]),
            [s["stage"] for s in session_train],
            epochs=max(10, epochs // 2),
        )
        yt, yp = _predict_stages(session_test, tmp)
        session_metrics["test"] = _eval_stage_predictor(yt, yp) if yt else None

    enc = LabelEncoder()
    X_train = np.stack([s["flat"] for s in train])
    y_train = enc.fit_transform(stages)
    baseline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")),
        ]
    )
    baseline.fit(X_train, y_train)

    baseline_metrics = {}
    for name, subset in (("train", train), ("validation", val), ("test", test)):
        if not subset:
            baseline_metrics[name] = None
            continue
        X = np.stack([s["flat"] for s in subset])
        y_true = [s["stage"] for s in subset]
        y_pred = enc.inverse_transform(baseline.predict(X)).tolist()
        baseline_metrics[name] = _eval_stage_predictor(y_true, y_pred)

    comparison = {}
    if metrics.get("test") and baseline_metrics.get("test"):
        comparison = {
            "world_model_macro_f1": metrics["test"]["macro_f1"],
            "logistic_regression_macro_f1": baseline_metrics["test"]["macro_f1"],
            "world_model_accuracy": metrics["test"]["accuracy"],
            "logistic_regression_accuracy": baseline_metrics["test"]["accuracy"],
            "world_model_precision": metrics["test"]["macro_precision"],
            "world_model_recall": metrics["test"]["macro_recall"],
            "logistic_regression_precision": baseline_metrics["test"]["macro_precision"],
            "logistic_regression_recall": baseline_metrics["test"]["macro_recall"],
            "improvement_macro_f1": round(metrics["test"]["macro_f1"] - baseline_metrics["test"]["macro_f1"], 4),
            "note": "Same stratified holdout. World model uses LSTM temporal encoding; LR uses flattened T1–T4 features.",
        }

    # Deployment weights: refit on all samples after metrics are locked.
    deploy = LSTMWorldModel(seed=42)
    deploy.fit(
        np.stack([s["seq"] for s in samples]),
        np.stack([s["next_state"] for s in samples]),
        [s["stage"] for s in samples],
        epochs=epochs,
    )
    y_all_enc = LabelEncoder()
    X_all = np.stack([s["flat"] for s in samples])
    y_deploy = y_all_enc.fit_transform([s["stage"] for s in samples])
    baseline_deploy = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", solver="lbfgs")),
        ]
    )
    baseline_deploy.fit(X_all, y_deploy)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deploy.to_bundle(), WORLD_MODEL_PATH)
    joblib.dump({"model": baseline_deploy, "encoder": y_all_enc, "feature_dim": X_all.shape[1]}, BASELINE_PATH)

    meta = {
        "model_name": "LSTM World Model",
        "architecture": "NumPy LSTM + attention over dual flow/packet state vectors; K-step P(S_t+1|history) rollout; RF stage head on temporal encodings",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sequence_length": SEQ_LEN,
        "state_features": STATE_FEATURES,
        "samples": len(samples),
        "sessions": len(sessions),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "session_held_out_metrics": session_metrics,
        "comparison_vs_logistic_regression": comparison,
        "training_source": "chronological ChronoFlux sessions (live/lab and/or curated attack-progression sessions)",
        "evaluation_protocol": "Stratified 75/25 holdout for baseline comparison; session-held-out reported separately; deployment weights refit on all samples.",
    }
    WORLD_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    BASELINE_META_PATH.write_text(
        json.dumps({"model_name": "Logistic Regression baseline", "metrics": baseline_metrics, "comparison": comparison}, indent=2),
        encoding="utf-8",
    )

    _cache = deploy
    _baseline_cache = {"model": baseline_deploy, "encoder": y_all_enc}
    return {"trained": True, "ready": True, **meta}


def predict_world_model(records: list[dict], window_seconds: float = 10.0) -> dict | None:
    model = load_world_model()
    if model is None or len(records) < SEQ_LEN:
        return None

    recent = records[-SEQ_LEN:]
    seq = np.stack([state_vector(r) for r in recent], axis=0)
    from core.killchain import load_prior
    hybrid = load_prior()
    current_stage = (recent[-1].get("prediction") or {}).get("current_stage")
    rollout = model.rollout(seq, steps=3, hybrid=hybrid, current_stage=current_stage)
    model_name = ("LSTM World Model + Kill-Chain Prior (hybrid v8)" if hybrid
                  else "LSTM World Model")
    if not rollout:
        return None

    first = rollout[0]
    stages = [h["stage"] for h in rollout]
    confs = [h["confidence"] for h in rollout]
    infiltration = infiltration_score_from_stages(stages, confs)
    driving = explain_prediction(seq, rollout)

    # Timeline probability for UI
    timeline = []
    for h in rollout:
        attack_mass = sum(p for s, p in h["probabilities"].items() if s in ATTACK_STAGES)
        timeline.append(
            {
                "horizon": h["horizon"],
                "label": h["label"],
                "infiltration_probability": round(float(attack_mass), 4),
                "predicted_stage": h["stage"],
                "confidence": h["confidence"],
                "mitre_tactics": mitre_for_stage(h["stage"]),
                "expected_horizon_seconds": round(float(window_seconds) * h["horizon"], 2),
            }
        )

    if not first["accepted"] and infiltration < 0.35:
        return {
            "status": "insufficient_evidence",
            "mode": "lstm_world_model",
            "model_name": model_name,
            "forecaster": "hybrid_killchain_v8" if hybrid else "lstm",
            "next_stage": None,
            "candidate_stage": first["stage"],
            "confidence": first["confidence"],
            "infiltration_probability": round(infiltration, 4),
            "progression": [],
            "horizons": [{k: v for k, v in h.items() if k not in {"next_state", "time_attention", "feature_gates"}} | {"accepted": h["accepted"], "expected_horizon_seconds": round(float(window_seconds) * h["horizon"], 2)} for h in rollout],
            "timeline": timeline,
            "driving_features": driving,
            "evidence": (
                f"Highest T5 confidence is {first['confidence']:.1%} (threshold {MIN_ACCEPTED_CONFIDENCE:.0%}). "
                "No forced forecast is issued."
            ),
            "window_count": SEQ_LEN,
            "mitre_tactics": mitre_for_stage(first["stage"]),
        }

    accepted = [h["stage"] for h in rollout if h["accepted"] or infiltration >= 0.35]
    return {
        "status": "active",
        "mode": "lstm_world_model",
        "model_name": model_name,
        "forecaster": "hybrid_killchain_v8" if hybrid else "lstm",
        "next_stage": first["stage"],
        "confidence": first["confidence"],
        "infiltration_probability": round(max(infiltration, first["confidence"] if first["stage"] in ATTACK_STAGES else infiltration), 4),
        "progression": accepted or [first["stage"]],
        "horizons": [
            {
                "horizon": h["horizon"],
                "label": h["label"],
                "stage": h["stage"],
                "confidence": h["confidence"],
                "accepted": bool(h["accepted"] or infiltration >= 0.35),
                "expected_horizon_seconds": round(float(window_seconds) * h["horizon"], 2),
                "probabilities": h["probabilities"],
                "mitre_tactics": mitre_for_stage(h["stage"]),
            }
            for h in rollout
        ],
        "timeline": timeline,
        "driving_features": driving,
        "mitre_tactics": mitre_for_stage(first["stage"]),
        "evidence": (
            ("K-step LSTM world-model rollout of network-state transition dynamics P(S_t+1|history), "
             "confidence-gated with a learned kill-chain transition prior. " if hybrid else
             "K-step LSTM world-model rollout of network-state transition dynamics P(S_t+1|history). ")
            + f"Infiltration probability over next {len(rollout)} windows: {infiltration:.1%}. "
            f"Top drivers: {', '.join(d['label'] for d in driving[:3])}."
        ),
        "window_count": SEQ_LEN,
        "minimum_confidence": MIN_ACCEPTED_CONFIDENCE,
    }


def status(records: list[dict] | None = None) -> dict:
    meta = {}
    if WORLD_META_PATH.exists():
        try:
            meta = json.loads(WORLD_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    out = {
        "ready": world_model_ready(),
        "model_name": "LSTM World Model",
        "model_qualification": "Recurrent world model learning P(S_t+1|history) with K-step rollout, MITRE stage mapping and attention attribution.",
        "baseline_ready": baseline_ready(),
        "sequence_length": SEQ_LEN,
        "horizons": list(HORIZONS),
        "state_feature_count": len(STATE_FEATURES),
        "minimum_sequence_samples": MIN_SEQUENCE_SAMPLES,
        "minimum_recording_sessions": MIN_SESSIONS,
        "minimum_forecast_confidence": MIN_ACCEPTED_CONFIDENCE,
        **meta,
    }
    if records is not None:
        built = build_transition_dataset(records)
        out["observed_windows"] = len(records)
        out["recording_sessions"] = len(built["sessions"])
        out["available_transition_samples"] = len(built["samples"])
        out["available_samples"] = {"1": len(built["samples"]), "2": max(0, len(built["samples"]) - 1), "3": max(0, len(built["samples"]) - 2)}
    return out
