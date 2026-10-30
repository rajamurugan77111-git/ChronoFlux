from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import io

from core.schema import FEATURE_COLUMNS
from core.engine import analyse
from core.forecast import forecast
from core.history import append as append_history, latest as latest_record, read as read_history, clear as clear_history
from core.detector import models_ready, predict_batch
from core.anomaly import ready as anomaly_ready, score_batch
from core.temporal_ml import auto_train as auto_train_temporal, train as train_temporal, status as temporal_status, temporal_model_ready
from core.world_model import (
    train_world_model,
    status as world_status,
    world_model_ready,
)
from core.pcap_ingest import parse_pcap_bytes
from core.forecast_eval import evaluate_previous, summary as forecast_eval_summary, clear as clear_forecast_eval
from core.devices import update as update_device, read as read_devices, clear as clear_devices
from core.evidence_chain import append_event as append_evidence, read as read_evidence, verify_chain

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
MODELS_DIR = ROOT / "models"


def _ensure_world_model():
    if world_model_ready():
        return
    try:
        from data_tools.generate_attack_sessions import generate
        train_world_model(generate(sessions_per_pattern=4), force=True, epochs=22)
    except Exception as exc:
        print(f"[ChronoFlux] world-model bootstrap skipped: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _ensure_world_model()
    yield


app = FastAPI(
    title="ChronoFlux",
    version="7.1.0",
    description="LSTM world-model network attack forecasting, hybrid detection and tamper-evident evidence intelligence",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    telemetry: Dict[str, Any] = Field(default_factory=dict)


class LiveSensorRequest(BaseModel):
    sensor_id: str = "ubuntu-gateway"
    session_id: str | None = None
    timestamp: str | None = None
    features: Dict[str, Any]
    telemetry: Dict[str, Any] = Field(default_factory=dict)


def _compact_history(limit: int = 80) -> list[dict]:
    rows = read_history(limit=limit)
    out = []
    for row in rows:
        pred = row.get("prediction", {})
        tel = row.get("telemetry", {})
        feat = row.get("features", {})
        out.append({
            "timestamp": row.get("timestamp") or row.get("received_at"),
            "sensor_id": row.get("sensor_id"),
            "session_id": row.get("session_id"),
            "current_stage": pred.get("current_stage", "Unknown"),
            "attack_category": pred.get("current_attack_category"),
            "risk_level": pred.get("risk_level", "UNKNOWN"),
            "risk_score": pred.get("risk_score", 0),
            "anomaly_score": pred.get("anomaly_score", 0),
            "confidence": pred.get("current_stage_confidence"),
            "packet_count": tel.get("packet_count", (feat.get("spkts", 0) or 0) + (feat.get("dpkts", 0) or 0)),
            "packet_rate": tel.get("packet_rate", feat.get("rate", 0)),
            "source_ip": tel.get("source_ip"),
            "source_mac": tel.get("source_mac"),
            "source_port": tel.get("source_port"),
            "destination_ip": tel.get("primary_destination"),
            "destination_port": tel.get("destination_port"),
            "protocol": tel.get("protocol", feat.get("proto")),
            "service": tel.get("service", feat.get("service")),
            "mitre_tactics": pred.get("mitre_tactics", []),
            "forecast": pred.get("forecast", {}),
            "signatures": pred.get("signatures", []),
        })
    return out


@app.get("/api/health")
@app.get("/health")
def health():
    current_models = models_ready()
    world = world_status(read_history(limit=2500))
    return {
        "status": "ChronoFlux backend running",
        "version": "8.0.0",
        "mode": "live-ready",
        "models": {
            **current_models,
            "anomaly": anomaly_ready(),
            "world_model": world_model_ready(),
            "temporal_ml": temporal_model_ready(),
            "temporal_engine": world.get("model_name"),
        },
        "capabilities": [
            "live_gateway_ingestion",
            "multi_device_visibility",
            "normal_vs_attack_detection",
            "attack_category_classification",
            "normal_baseline_anomaly_scoring",
            "behavioural_signature_detection",
            "hybrid_risk_fusion",
            "attack_category_to_behaviour_mapping",
            "MITRE_interpretation",
            "flow_and_packet_feature_vectors",
            "lstm_world_model_dynamics",
            "k_step_forward_simulation",
            "infiltration_probability_timeline",
            "attention_shap_style_explainability",
            "logistic_regression_baseline_comparison",
            "sequence_rf_ablation",
            "T5_T6_T7_multi_horizon_prediction",
            "forecast_lead_time_verification",
            "offline_csv_analysis",
            "offline_pcap_analysis",
            "deep_L3_L4_flow_telemetry",
            "tamper_evident_private_evidence_chain",
            "automl_model_arena_selection",
            "permutation_feature_importance",
        ],
    }


@app.get("/api/system")
def system_status():
    current_models = models_ready()
    history_rows = read_history(limit=2500)
    metrics = {}
    for name, path in {
        "detection": MODELS_DIR / "current_stage" / "detection_metrics.json",
        "stage": MODELS_DIR / "current_stage" / "stage_metrics.json",
        "anomaly": MODELS_DIR / "anomaly" / "anomaly_meta.json",
        "temporal": MODELS_DIR / "temporal" / "temporal_meta.json",
        "world_model": MODELS_DIR / "temporal" / "world_model_meta.json",
        "baseline": MODELS_DIR / "temporal" / "baseline_meta.json",
    }.items():
        if path.exists():
            try:
                metrics[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                metrics[name] = {}
    return {
        "models": {
            **current_models,
            "anomaly": anomaly_ready(),
            "world_model": world_model_ready(),
            "temporal_ml": temporal_model_ready(),
        },
        "temporal": world_status(history_rows),
        "sequence_rf_ablation": temporal_status(history_rows),
        "forecast_evaluation": forecast_eval_summary(),
        "metrics": metrics,
        "frontend": "ChronoFlux Professional Web Console",
        "sensor_endpoint": "/api/sensor/live",
        "evidence_chain": verify_chain(),
        "devices": len(read_devices()),
    }


@app.get("/api/models/arena")
def model_arena():
    """Leaderboard from training/model_arena.py (automated model selection)."""
    path = MODELS_DIR / "current_stage" / "model_arena.json"
    if not path.exists():
        return {"available": False,
                "hint": "Run: python -m training.model_arena"}
    try:
        return {"available": True, **json.loads(path.read_text(encoding="utf-8"))}
    except Exception as exc:  # corrupt file
        return {"available": False, "error": str(exc)}


@app.post("/api/predict")
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        history = read_history(limit=4)
        return analyse(request.features, request.telemetry, history)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/sensor/live")
@app.post("/sensor/live")
def sensor_live(request: LiveSensorRequest):
    try:
        previous = read_history(limit=2500)
        prediction = analyse(request.features, request.telemetry, previous[-4:])
        record = {
            "sensor_id": request.sensor_id,
            "session_id": request.session_id,
            "timestamp": request.timestamp,
            "features": request.features,
            "telemetry": request.telemetry,
            "prediction": prediction,
        }

        # Verify the prior T5 forecast before using the current observation to update the model.
        forecast_verification = evaluate_previous(previous, record)

        # Sequence forecasting learns from chronological sessions; also refresh world model when possible.
        combined = (previous + [record])[-2500:]
        temporal_train_status = auto_train_temporal(combined)
        if world_model_ready() or temporal_model_ready():
            prediction["forecast"] = forecast(
                combined,
                window_seconds=float(request.telemetry.get("window_seconds", 10) or 10),
            )
            record["prediction"] = prediction

        update_device(request.telemetry, request.timestamp, prediction)
        append_history(record)

        evidence_block = None
        if (
            prediction.get("risk_level") in {"MEDIUM", "HIGH", "CRITICAL"}
            or prediction.get("current_stage") not in {"Normal", "Monitoring", None}
            or bool(prediction.get("signatures"))
        ):
            strongest = (prediction.get("signatures") or [{}])[0]
            fcast = prediction.get("forecast") or {}
            telemetry = request.telemetry or {}
            evidence_block = append_evidence({
                "event_time": request.timestamp,
                "sensor_id": request.sensor_id,
                "attack_category": prediction.get("current_attack_category"),
                "stage": prediction.get("current_stage"),
                "mitre_tactics": prediction.get("mitre_tactics", []),
                "risk_level": prediction.get("risk_level"),
                "risk_score": prediction.get("risk_score"),
                "confidence": prediction.get("current_stage_confidence"),
                "source_ip": telemetry.get("source_ip"),
                "source_mac": telemetry.get("source_mac"),
                "source_port": telemetry.get("source_port"),
                "destination_ip": telemetry.get("primary_destination"),
                "destination_port": telemetry.get("destination_port"),
                "protocol": telemetry.get("protocol", request.features.get("proto")),
                "service": telemetry.get("service", request.features.get("service")),
                "signature": strongest.get("name"),
                "forecast_next": fcast.get("next_stage"),
                "forecast_model": fcast.get("model_name"),
                "evidence_summary": prediction.get("explanations", [])[:3],
            })

        return {
            "status": "received",
            "sensor_id": request.sensor_id,
            "session_id": request.session_id,
            "prediction": prediction,
            "evidence_block": evidence_block,
            "forecast_verification": forecast_verification,
            "temporal": temporal_train_status,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/latest")
@app.get("/latest")
def latest():
    row = latest_record()
    if not row:
        return {"status": "waiting", "features": None, "telemetry": None, "prediction": None}
    return {
        "status": "ok",
        "sensor_id": row.get("sensor_id"),
        "session_id": row.get("session_id"),
        "timestamp": row.get("timestamp") or row.get("received_at"),
        "features": row.get("features"),
        "telemetry": row.get("telemetry", {}),
        "prediction": row.get("prediction"),
    }


@app.get("/api/history")
def history(limit: int = Query(80, ge=1, le=500)):
    return {"items": _compact_history(limit=limit)}


@app.delete("/api/history")
def reset_history():
    clear_history()
    clear_forecast_eval()
    return {"status": "cleared", "note": "Temporal model and device registry were preserved."}


@app.get("/api/devices")
def devices():
    items = read_devices()
    return {
        "items": items,
        "total": len(items),
        "new": sum(1 for x in items if x.get("status") == "NEW"),
        "elevated": sum(1 for x in items if x.get("risk_level") in {"HIGH", "CRITICAL"}),
    }


@app.delete("/api/devices")
def reset_devices():
    clear_devices()
    return {"status": "cleared"}


@app.get("/api/temporal/status")
def get_temporal_status():
    rows = read_history(limit=2500)
    return {
        "world_model": world_status(rows),
        "sequence_rf_ablation": temporal_status(rows),
    }


@app.post("/api/temporal/train")
def train_temporal_now():
    """Train LSTM world model (primary) and refresh RF ablation when possible."""
    from data_tools.generate_attack_sessions import generate

    rows = read_history(limit=2500)
    lab = generate(sessions_per_pattern=4)
    combined = lab + rows
    world = train_world_model(combined, force=True, epochs=30)
    rf = None
    if len(rows) >= 12:
        try:
            rf = train_temporal(rows, force=True)
        except Exception as exc:
            rf = {"trained": False, "reason": str(exc)}
    return {
        "ready": bool(world.get("ready") or world_model_ready()),
        "world_model": world,
        "sequence_rf_ablation": rf,
    }


@app.get("/api/forecast/evaluation")
def forecast_evaluation():
    return forecast_eval_summary()


@app.post("/api/demo/seed")
def demo_seed(
    clear_existing: bool = Query(True),
    mode: str = Query("attack", pattern="^(attack|normal)$"),
):
    """Load a curated lab session.

    mode=attack  → attack progression (default jury demo / forecast sample)
    mode=normal  → all-Normal baseline (no-attack demo)
    """
    from data_tools.generate_attack_sessions import generate, generate_normal

    _ensure_world_model()
    if clear_existing:
        clear_history()
        clear_forecast_eval()

    if mode == "normal":
        rows = generate_normal(windows=8, seed=21)
        sensor_id = "demo-normal"
        note = "Normal/baseline lab session loaded (no attack). Expect LOW risk and Normal stages."
    else:
        rows = [r for r in generate(sessions_per_pattern=1, seed=11) if r["session_id"] == "lab-session-001"]
        if len(rows) < 6:
            rows = generate(sessions_per_pattern=1, seed=11)[:10]
        sensor_id = "demo-lab"
        note = "Attack-progression lab session loaded for offline jury demonstration."

    history = []
    seeded = 0
    for row in rows:
        prediction = analyse(row["features"], row["telemetry"], history[-4:])
        record = {
            "sensor_id": sensor_id,
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "features": row["features"],
            "telemetry": row["telemetry"],
            "prediction": prediction,
        }
        if history:
            evaluate_previous(history, record)
        history.append(record)
        update_device(row["telemetry"], row["timestamp"], prediction)
        append_history(record)
        seeded += 1

    latest = history[-1] if history else None
    return {
        "status": "seeded",
        "mode": mode,
        "windows": seeded,
        "world_model_ready": world_model_ready(),
        "latest_stage": (latest or {}).get("prediction", {}).get("current_stage"),
        "forecast": (latest or {}).get("prediction", {}).get("forecast"),
        "note": note,
    }


OFFLINE_RISK_MAP = {
    "Normal": "LOW", "Reconnaissance": "MEDIUM", "Fuzzers": "MEDIUM",
    "Analysis": "MEDIUM", "Exploits": "HIGH", "DoS": "HIGH",
    "Generic": "HIGH", "Backdoor": "CRITICAL", "Shellcode": "CRITICAL",
    "Worms": "CRITICAL", "Suspicious": "MEDIUM",
}


def _convert_chunk(df_chunk: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    from core.dataset_adapters import convert_cic_ids2018, convert_ctu13

    if source_kind == "chronoflux":
        return df_chunk.copy()
    if source_kind == "cicids2018":
        return convert_cic_ids2018(df_chunk, max_rows=None)
    if source_kind == "ctu13":
        return convert_ctu13(df_chunk, max_rows=None)
    raise ValueError(f"Unsupported schema: {source_kind}")


def _score_converted_chunk(converted: pd.DataFrame) -> dict:
    from core.dataset_adapters import chronoflux_feature_frame

    sample = chronoflux_feature_frame(converted)
    ml_results = predict_batch(sample)
    anomaly_results = score_batch(sample)

    counts: dict[str, int] = {}
    risks: dict[str, int] = {}
    truth_counts: dict[str, int] = {}
    anomaly_sum = 0.0
    correct = 0
    labelled = 0
    binary_correct = 0
    examples = []

    for idx, (_, row) in enumerate(converted.head(len(sample)).iterrows()):
        ml = ml_results[idx]
        an = anomaly_results[idx]
        stage = ml.get("ml_stage", "Unknown")
        risk = OFFLINE_RISK_MAP.get(stage, "MEDIUM")
        counts[stage] = counts.get(stage, 0) + 1
        risks[risk] = risks.get(risk, 0) + 1
        anomaly_sum += float(an.get("score", 0) or 0)

        if "attack_cat" in converted.columns:
            truth = str(row.get("attack_cat"))
            if truth and truth.lower() != "nan":
                labelled += 1
                truth_counts[truth] = truth_counts.get(truth, 0) + 1
                normalized_truth = "Backdoor" if truth == "Backdoors" else truth
                if normalized_truth == stage:
                    correct += 1
                if (normalized_truth != "Normal") == (stage != "Normal"):
                    binary_correct += 1

        if idx < 8:
            examples.append({
                "row": int(idx),
                "attack_category": stage,
                "truth": str(row.get("attack_cat")) if "attack_cat" in converted.columns else None,
                "risk": risk,
                "confidence": ml.get("ml_confidence"),
                "anomaly_score": an.get("score"),
            })

    n = max(1, len(sample))
    return {
        "analyzed_rows": len(sample),
        "stage_distribution": counts,
        "risk_distribution": risks,
        "truth_distribution": truth_counts,
        "anomaly_sum": anomaly_sum,
        "labelled_rows": labelled,
        "correct": correct,
        "binary_correct": binary_correct,
        "examples": examples,
    }


def _merge_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


@app.post("/api/offline/analyze-csv")
async def offline_csv(
    file: UploadFile = File(...),
    max_rows: int = Query(0, ge=0, le=500000),
    mode: str = Query("all", pattern="^(all|stratified|head)$"),
    batch_size: int = Query(2000, ge=200, le=20000),
):
    """Analyse a CSV in one response (uses internal batching). Prefer /analyze-csv-stream for live UI."""
    events = []
    async for line in _iter_csv_batch_events(file, max_rows=max_rows, mode=mode, batch_size=batch_size):
        events.append(json.loads(line))
    if not events:
        raise HTTPException(status_code=400, detail="No rows analysed.")
    final = events[-1]
    final.pop("event", None)
    final.pop("batch_index", None)
    final.pop("batch_total", None)
    final.pop("done", None)
    return final


@app.post("/api/offline/analyze-csv-stream")
async def offline_csv_stream(
    file: UploadFile = File(...),
    max_rows: int = Query(0, ge=0, le=500000),
    mode: str = Query("all", pattern="^(all|stratified|head)$"),
    batch_size: int = Query(2000, ge=200, le=20000),
):
    """NDJSON stream: one JSON object per batch with cumulative totals."""
    raw = await file.read()
    filename = file.filename or "upload.csv"

    def event_sync():
        for line in _iter_csv_batch_events_bytes(raw, filename=filename, max_rows=max_rows, mode=mode, batch_size=batch_size):
            yield line

    return StreamingResponse(event_sync(), media_type="application/x-ndjson")


async def _iter_csv_batch_events(file: UploadFile, max_rows: int, mode: str, batch_size: int):
    raw = await file.read()
    for line in _iter_csv_batch_events_bytes(
        raw,
        filename=file.filename or "upload.csv",
        max_rows=max_rows,
        mode=mode,
        batch_size=batch_size,
    ):
        yield line


def _iter_csv_batch_events_bytes(
    raw: bytes,
    filename: str,
    max_rows: int,
    mode: str,
    batch_size: int,
):
    from core.dataset_adapters import detect_dataset, select_rows_for_analysis

    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    except Exception as exc:
        yield json.dumps({"event": "error", "detail": f"Unable to read CSV: {exc}", "done": True}) + "\n"
        return

    try:
        source_kind = detect_dataset(df)
        if source_kind == "unknown":
            raise ValueError("Unrecognized CSV schema")
        row_cap = None if max_rows == 0 else max_rows
        selected, sample_meta = select_rows_for_analysis(df, max_rows=row_cap, mode=mode)
    except Exception as exc:
        yield json.dumps({"event": "error", "detail": str(exc), "done": True}) + "\n"
        return

    total_rows = len(df)
    work = selected.reset_index(drop=True)
    n = len(work)
    if n == 0:
        yield json.dumps({"event": "error", "detail": "CSV has no rows.", "done": True}) + "\n"
        return

    batches = max(1, (n + batch_size - 1) // batch_size)
    yield json.dumps({
        "event": "start",
        "filename": filename,
        "source_schema": source_kind,
        "sample_mode": sample_meta.get("mode", mode),
        "total_rows": total_rows,
        "planned_rows": n,
        "batch_size": batch_size,
        "batch_total": batches,
        "label_counts_in_sample": sample_meta.get("label_counts") or {},
        "done": False,
    }) + "\n"

    counts: dict[str, int] = {}
    risks: dict[str, int] = {}
    truth_counts: dict[str, int] = {}
    anomaly_sum = 0.0
    analysed = 0
    labelled = 0
    correct = 0
    binary_correct = 0
    examples: list[dict] = []

    for b in range(batches):
        start = b * batch_size
        end = min(n, start + batch_size)
        chunk = work.iloc[start:end]
        try:
            converted = _convert_chunk(chunk, source_kind)
            part = _score_converted_chunk(converted)
        except Exception as exc:
            yield json.dumps({
                "event": "error",
                "detail": f"Batch {b + 1} failed: {exc}",
                "done": True,
            }) + "\n"
            return

        _merge_counts(counts, part["stage_distribution"])
        _merge_counts(risks, part["risk_distribution"])
        _merge_counts(truth_counts, part["truth_distribution"])
        anomaly_sum += part["anomaly_sum"]
        analysed += part["analyzed_rows"]
        labelled += part["labelled_rows"]
        correct += part["correct"]
        binary_correct += part["binary_correct"]
        if len(examples) < 25:
            examples.extend(part["examples"][: max(0, 25 - len(examples))])

        done = b == batches - 1
        payload = {
            "event": "batch",
            "filename": filename,
            "source_schema": source_kind,
            "sample_mode": sample_meta.get("mode", mode),
            "total_rows": total_rows,
            "planned_rows": n,
            "analyzed_rows": analysed,
            "batch_index": b + 1,
            "batch_total": batches,
            "batch_rows": part["analyzed_rows"],
            "label_counts_in_sample": sample_meta.get("label_counts") or {},
            "truth_distribution": truth_counts,
            "stage_distribution": counts,
            "risk_distribution": risks,
            "average_anomaly": round(anomaly_sum / max(1, analysed), 1),
            "labelled_rows": labelled,
            "sample_accuracy": round(correct / labelled, 4) if labelled else None,
            "binary_accuracy": round(binary_correct / labelled, 4) if labelled else None,
            "examples": examples,
            "done": done,
            "note": (
                f"Batch {b + 1}/{batches} · {analysed}/{n} rows "
                f"(file total {total_rows}) · mode={sample_meta.get('mode', mode)} · schema={source_kind}."
            ),
        }
        yield json.dumps(payload) + "\n"



@app.post("/api/offline/analyze-pcap")
async def offline_pcap(
    file: UploadFile = File(...),
    window_seconds: float = Query(5.0, ge=1.0, le=60.0),
):
    """Ingest a PCAP, extract flow+packet features, run detection and world-model forecast."""
    name = (file.filename or "").lower()
    if not (name.endswith(".pcap") or name.endswith(".pcapng") or name.endswith(".cap")):
        raise HTTPException(status_code=400, detail="Please upload a .pcap / .pcapng file.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty PCAP upload.")

    try:
        windows = parse_pcap_bytes(raw, window_seconds=window_seconds)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not windows:
        raise HTTPException(status_code=400, detail="No IP packets could be parsed from the PCAP.")

    history = []
    stage_counts: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    timeline = []
    last_forecast = {}

    for window in windows:
        prediction = analyse(window["features"], window["telemetry"], history[-4:])
        record = {
            "sensor_id": "pcap-upload",
            "session_id": f"pcap-{file.filename}",
            "timestamp": window["timestamp"],
            "features": window["features"],
            "telemetry": window["telemetry"],
            "prediction": prediction,
        }
        history.append(record)
        stage = prediction.get("current_stage", "Unknown")
        risk = prediction.get("risk_level", "UNKNOWN")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        fcast = prediction.get("forecast") or {}
        last_forecast = fcast
        timeline.append(
            {
                "timestamp": window["timestamp"],
                "current_stage": stage,
                "risk_level": risk,
                "risk_score": prediction.get("risk_score"),
                "infiltration_probability": fcast.get("infiltration_probability"),
                "forecast_stage": fcast.get("next_stage"),
                "driving_features": (fcast.get("driving_features") or [])[:5],
                "mitre_tactics": fcast.get("mitre_tactics") or prediction.get("mitre_tactics") or [],
            }
        )

    return {
        "filename": file.filename,
        "windows": len(windows),
        "stage_distribution": stage_counts,
        "risk_distribution": risk_counts,
        "forecast": last_forecast,
        "timeline": timeline[-40:],
        "world_model_ready": world_model_ready(),
        "note": "PCAP windows use dual flow-level and packet-level features. Forecast uses LSTM world-model K-step rollout when weights are available.",
    }


@app.get("/api/evidence")
def evidence(limit: int = Query(100, ge=1, le=1000)):
    blocks = read_evidence(limit=limit)
    return {"verification": verify_chain(), "items": list(reversed(blocks))}


@app.get("/api/evidence/verify")
def evidence_verify():
    return verify_chain()


@app.get("/")
def home():
    return FileResponse(
        FRONTEND / "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


app.mount("/assets", StaticFiles(directory=FRONTEND), name="assets")
