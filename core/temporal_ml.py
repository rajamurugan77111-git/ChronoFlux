"""Sequence Random Forest ablation for ChronoFlux forecasting.

Kept as a non-recurrent baseline/ablation beside the LSTM world model.
Random Forest itself is not time-aware; temporal context comes from ordered
T1..T4 state vectors predicting T5/T6/T7.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import math

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.state_features import STATE_FEATURES, STAGE_ORDER, STAGE_CODE, state_vector

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "temporal"
MODEL_PATH = MODEL_DIR / "sequence_rf.joblib"
META_PATH = MODEL_DIR / "temporal_meta.json"

SEQ_LEN = 4
HORIZONS = (1, 2, 3)
MAX_HISTORY = 5000
MIN_SEQUENCE_SAMPLES = 30
MIN_CLASSES = 2
MIN_SESSIONS = 3
MIN_ACCEPTED_CONFIDENCE = 0.60

_cache = None


def _num(value, default=0.0):
    try:
        value = float(value)
        if not math.isfinite(value):
            return default
        return value
    except Exception:
        return default


def _parse_time(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def sequence_vector(records: list[dict]) -> np.ndarray:
    recent = records[-SEQ_LEN:]
    if len(recent) != SEQ_LEN:
        raise ValueError(f"Need exactly {SEQ_LEN} ordered windows")
    return np.concatenate([state_vector(r) for r in recent])


def _session_id(record: dict, fallback: str) -> str:
    return str(record.get("session_id") or record.get("recording_id") or fallback)


def _sessions(records: list[dict]) -> list[dict]:
    """Return chronological recording sessions.

    New sensors submit an explicit session_id. Legacy records are segmented by
    large time gaps so they are not silently treated as one uninterrupted run.
    """
    rows = records[-MAX_HISTORY:]
    explicit = any(r.get("session_id") or r.get("recording_id") for r in rows)

    if explicit:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[_session_id(r, "legacy")].append(r)
        sessions = []
        for sid, group in grouped.items():
            group = sorted(group, key=lambda r: str(r.get("timestamp") or r.get("received_at") or ""))
            sessions.append({"id": sid, "rows": group})
        sessions.sort(key=lambda s: str((s["rows"][0] if s["rows"] else {}).get("timestamp") or ""))
        return sessions

    # Legacy fallback: split when inter-window gap is much larger than expected.
    sessions = []
    current = []
    prev_time = None
    legacy_index = 1
    for r in sorted(rows, key=lambda x: str(x.get("timestamp") or x.get("received_at") or "")):
        t = _parse_time(r.get("timestamp") or r.get("received_at"))
        window = _num((r.get("telemetry") or {}).get("window_seconds", 10), 10)
        if current and prev_time and t and (t - prev_time).total_seconds() > max(30.0, window * 4):
            sessions.append({"id": f"legacy-session-{legacy_index}", "rows": current})
            legacy_index += 1
            current = []
        current.append(r)
        if t:
            prev_time = t
    if current:
        sessions.append({"id": f"legacy-session-{legacy_index}", "rows": current})
    return sessions


def _session_samples(session: dict, horizon: int) -> list[dict]:
    rows = session["rows"]
    out = []
    last_start = len(rows) - SEQ_LEN - horizon + 1
    for start in range(max(0, last_start)):
        seq = rows[start : start + SEQ_LEN]
        target = rows[start + SEQ_LEN + horizon - 1]
        stage = (target.get("prediction") or {}).get("current_stage")
        if not stage:
            continue
        try:
            x = sequence_vector(seq)
        except Exception:
            continue
        out.append(
            {
                "x": x,
                "y": str(stage),
                "session_id": session["id"],
                "input_start": seq[0].get("timestamp") or seq[0].get("received_at"),
                "input_end": seq[-1].get("timestamp") or seq[-1].get("received_at"),
                "target_time": target.get("timestamp") or target.get("received_at"),
            }
        )
    return out


def _split_sessions(sessions: list[dict]) -> tuple[list[str], list[str], list[str]]:
    ids = [s["id"] for s in sessions]
    n = len(ids)
    if n < MIN_SESSIONS:
        return ids, [], []

    train_n = max(1, int(n * 0.60))
    val_n = max(1, int(n * 0.20))
    if train_n + val_n >= n:
        train_n = max(1, n - 2)
        val_n = 1

    train_ids = ids[:train_n]
    val_ids = ids[train_n : train_n + val_n]
    test_ids = ids[train_n + val_n :]
    return train_ids, val_ids, test_ids


def build_training_sets(records: list[dict]) -> dict:
    sessions = _sessions(records)
    train_ids, val_ids, test_ids = _split_sessions(sessions)
    split_map = {sid: "train" for sid in train_ids}
    split_map.update({sid: "validation" for sid in val_ids})
    split_map.update({sid: "test" for sid in test_ids})

    by_horizon = {}
    for horizon in HORIZONS:
        buckets = {"train": [], "validation": [], "test": [], "all": []}
        for session in sessions:
            samples = _session_samples(session, horizon)
            buckets["all"].extend(samples)
            split = split_map.get(session["id"], "train")
            buckets[split].extend(samples)
        by_horizon[horizon] = buckets

    return {
        "sessions": sessions,
        "train_session_ids": train_ids,
        "validation_session_ids": val_ids,
        "test_session_ids": test_ids,
        "horizons": by_horizon,
    }


def _xy(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        return (
            np.empty((0, SEQ_LEN * len(STATE_FEATURES))),
            np.empty((0,), dtype=object),
        )
    return np.asarray([s["x"] for s in samples], dtype=float), np.asarray(
        [s["y"] for s in samples], dtype=object
    )


def _macro_attack_fpr(y_true, y_pred) -> float | None:
    labels = sorted(set(list(y_true) + list(y_pred)))
    if not labels:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    rates = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        if denom > 0:
            rates.append(float(fp / denom))
    return float(np.mean(rates)) if rates else None


def _eval(model, samples: list[dict]) -> dict | None:
    X, y = _xy(samples)
    if not len(y):
        return None
    pred = model.predict(X)
    labels = sorted(set(y.tolist()) | set(pred.tolist()))
    cm = confusion_matrix(y, pred, labels=labels)
    normal_count = int(sum(1 for value in y if value == "Normal"))
    normal_fp = int(sum(1 for truth, p in zip(y, pred) if truth == "Normal" and p != "Normal"))
    return {
        "samples": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "macro_precision": round(float(precision_score(y, pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y, pred, average="macro", zero_division=0)), 4),
        "macro_false_positive_rate": round(float(_macro_attack_fpr(y, pred) or 0.0), 4),
        "normal_to_attack_false_positive_rate": round(float(normal_fp / normal_count), 4) if normal_count else None,
        "labels": labels,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def _load_bundle():
    global _cache
    if _cache is None and MODEL_PATH.exists():
        try:
            _cache = joblib.load(MODEL_PATH)
        except Exception:
            _cache = None
    return _cache


def temporal_model_ready() -> bool:
    bundle = _load_bundle()
    return bool(bundle and bundle.get("models", {}).get(1))


def status(records: list[dict] | None = None) -> dict:
    bundle = _load_bundle() or {}
    meta = {}
    if META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    out = {
        "ready": temporal_model_ready(),
        "model_name": "Sequence-based Random Forest (ablation)",
        "model_qualification": "Non-recurrent ablation: Random Forest over ordered T1–T4 feature representations. Prefer the LSTM world model for P(S_t+1|history) dynamics.",
        "role": "ablation_baseline",
        "sequence_length": SEQ_LEN,
        "horizons": list(HORIZONS),
        "state_features": STATE_FEATURES,
        "minimum_sequence_samples": MIN_SEQUENCE_SAMPLES,
        "minimum_classes": MIN_CLASSES,
        "minimum_recording_sessions": MIN_SESSIONS,
        "minimum_forecast_confidence": MIN_ACCEPTED_CONFIDENCE,
        "training_source": "genuine chronological ChronoFlux live/lab recording sessions",
        "split_policy": "recording/session-time split; entire sessions are kept in train/validation/test partitions",
        "models_ready": sorted(int(h) for h in (bundle.get("models") or {}).keys()),
        **meta,
    }

    if records is not None:
        built = build_training_sets(records)
        out["observed_windows"] = len(records)
        out["recording_sessions"] = len(built["sessions"])
        out["train_sessions"] = len(built["train_session_ids"])
        out["validation_sessions"] = len(built["validation_session_ids"])
        out["test_sessions"] = len(built["test_session_ids"])
        out["available_samples"] = {
            str(h): int(len(built["horizons"][h]["all"])) for h in HORIZONS
        }
        out["split_samples"] = {
            str(h): {
                split: int(len(built["horizons"][h][split]))
                for split in ("train", "validation", "test")
            }
            for h in HORIZONS
        }
        devices = {
            str((r.get("telemetry") or {}).get("source_ip"))
            for r in records
            if (r.get("telemetry") or {}).get("source_ip")
        }
        out["observed_devices"] = len(devices)

    return out


def train(records: list[dict], force: bool = False) -> dict:
    global _cache
    built = build_training_sets(records)
    sessions = built["sessions"]
    models = {}
    metrics = {}

    for horizon in HORIZONS:
        buckets = built["horizons"][horizon]
        train_samples = buckets["train"]
        val_samples = buckets["validation"]
        test_samples = buckets["test"]
        all_samples = buckets["all"]
        classes = sorted({s["y"] for s in train_samples})

        reasons = []
        if len(sessions) < MIN_SESSIONS:
            reasons.append(f"Need at least {MIN_SESSIONS} separate chronological recording sessions.")
        if len(all_samples) < MIN_SEQUENCE_SAMPLES:
            reasons.append(f"Need at least {MIN_SEQUENCE_SAMPLES} chronological sequence samples.")
        if len(classes) < MIN_CLASSES:
            reasons.append(f"Training partition needs at least {MIN_CLASSES} observed behaviour classes.")
        if not val_samples or not test_samples:
            reasons.append("Independent validation and test sessions are required.")

        if reasons:
            metrics[str(horizon)] = {
                "trained": False,
                "samples": int(len(all_samples)),
                "classes": sorted({s["y"] for s in all_samples}),
                "reason": " ".join(reasons),
                "split_samples": {
                    "train": len(train_samples),
                    "validation": len(val_samples),
                    "test": len(test_samples),
                },
            }
            continue

        X_train, y_train = _xy(train_samples)
        model = RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42 + horizon,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        validation_eval = _eval(model, val_samples)
        test_eval = _eval(model, test_samples)

        # Deployment model may learn from train+validation only after the test
        # result has been measured with the train-only model. Test stays unseen.
        deploy_samples = train_samples + val_samples
        X_deploy, y_deploy = _xy(deploy_samples)
        deploy_model = RandomForestClassifier(
            n_estimators=240,
            max_depth=14,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=42 + horizon,
            n_jobs=-1,
        )
        deploy_model.fit(X_deploy, y_deploy)
        models[horizon] = deploy_model

        metrics[str(horizon)] = {
            "trained": True,
            "samples": int(len(all_samples)),
            "classes": sorted({s["y"] for s in all_samples}),
            "train_samples": len(train_samples),
            "validation_samples": len(val_samples),
            "test_samples": len(test_samples),
            "validation": validation_eval,
            "test": test_eval,
        }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "sequence_length": SEQ_LEN,
        "state_features": STATE_FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_windows": len(records),
        "recording_sessions": len(sessions),
    }
    joblib.dump(bundle, MODEL_PATH)

    meta = {
        "model_name": "Sequence-based Random Forest",
        "model_qualification": "Ordered T1–T4 feature representation + separate Random Forest horizon classifiers.",
        "trained_at": bundle["trained_at"],
        "training_windows": len(records),
        "recording_sessions": len(sessions),
        "train_session_ids": built["train_session_ids"],
        "validation_session_ids": built["validation_session_ids"],
        "test_session_ids": built["test_session_ids"],
        "split_policy": "Entire recording sessions are split chronologically into train/validation/test to prevent overlapping-window leakage across partitions.",
        "metrics": metrics,
        "note": "Targets are future ChronoFlux behaviour states observed in genuine chronological live/lab sessions; they are not UNSW stage ground truth.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _cache = bundle
    return status(records)


def auto_train(records: list[dict]) -> dict:
    current = status(records)
    h1_samples = int((current.get("available_samples") or {}).get("1", 0))
    sessions = int(current.get("recording_sessions", 0) or 0)
    behaviours = {
        str((r.get("prediction") or {}).get("current_stage"))
        for r in records
        if (r.get("prediction") or {}).get("current_stage")
    }
    last_windows = int(current.get("training_windows", 0) or 0)
    should_train = (
        h1_samples >= MIN_SEQUENCE_SAMPLES
        and sessions >= MIN_SESSIONS
        and len(behaviours) >= MIN_CLASSES
        and (not temporal_model_ready() or len(records) >= last_windows + 12)
    )
    if should_train:
        return train(records)
    return current


def predict(records: list[dict], window_seconds: float = 10.0) -> dict | None:
    if len(records) < SEQ_LEN:
        return None
    bundle = _load_bundle()
    if not bundle:
        return None
    models = bundle.get("models") or {}
    if not models:
        return None

    x = sequence_vector(records).reshape(1, -1)
    horizons = []
    for horizon in HORIZONS:
        model = models.get(horizon)
        if model is None:
            continue
        probs = model.predict_proba(x)[0]
        classes = list(model.classes_)
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        horizons.append(
            {
                "horizon": horizon,
                "label": f"T{SEQ_LEN + horizon}",
                "stage": str(classes[idx]),
                "confidence": round(confidence, 4),
                "accepted": bool(confidence >= MIN_ACCEPTED_CONFIDENCE),
                "expected_horizon_seconds": round(float(window_seconds) * horizon, 2),
                "probabilities": {
                    str(classes[i]): round(float(probs[i]), 4)
                    for i in np.argsort(probs)[::-1][:4]
                },
            }
        )

    if not horizons:
        return None

    first = horizons[0]
    if not first["accepted"]:
        return {
            "mode": "sequence_rf",
            "model_name": "Sequence-based Random Forest",
            "status": "insufficient_evidence",
            "next_stage": None,
            "candidate_stage": first["stage"],
            "confidence": first["confidence"],
            "progression": [],
            "horizons": horizons,
            "evidence": (
                f"The highest T5 probability is {first['confidence']:.1%}, below the "
                f"{MIN_ACCEPTED_CONFIDENCE:.0%} reliability threshold. No forecast is issued."
            ),
            "window_count": SEQ_LEN,
        }

    accepted_progression = [h["stage"] for h in horizons if h["accepted"]]
    return {
        "mode": "sequence_rf",
        "model_name": "Sequence-based Random Forest",
        "status": "active",
        "next_stage": first["stage"],
        "confidence": first["confidence"],
        "progression": accepted_progression,
        "horizons": horizons,
        "evidence": "Forecast generated from ordered T1–T4 gateway state vectors. Random Forest receives the constructed temporal representation; it is not itself a time-aware model.",
        "window_count": SEQ_LEN,
    }
