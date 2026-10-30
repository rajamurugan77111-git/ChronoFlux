"""Quick runtime verification for ChronoFlux APIs + batch CSV stream."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        return json.loads(r.read().decode())


def post_json(path: str):
    req = urllib.request.Request(
        BASE + path,
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read().decode())


def post_csv(path: str, file_path: Path):
    boundary = "----CFBound"
    raw = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + raw + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        BASE + path,
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.status, r.read().decode()


def main():
    ok = []

    h = get("/api/health")
    models = h["models"]
    assert models["detection"] and models["stage"] and models["anomaly"] and models["world_model"]
    ok.append("health: detection/stage/anomaly/world_model ready")

    status, body = post_csv(
        "/api/offline/analyze-csv-stream?mode=stratified&max_rows=2000&batch_size=1000",
        ROOT / "data" / "sample" / "withlabel.csv",
    )
    assert status == 200, status
    msgs = [json.loads(ln) for ln in body.splitlines() if ln.strip()]
    assert msgs[0]["event"] == "start"
    batches = [m for m in msgs if m.get("event") == "batch"]
    assert batches and batches[-1]["done"]
    last = batches[-1]
    assert last["analyzed_rows"] == 2000
    truth = last.get("truth_distribution") or {}
    assert "Normal" in truth and ("DoS" in truth or "Exploits" in truth), truth
    assert last.get("stage_distribution"), last
    assert last.get("binary_accuracy") is not None
    ok.append(
        f"csv-stream: batches={len(batches)} analysed={last['analyzed_rows']} "
        f"truth={truth} pred={last['stage_distribution']} "
        f"bin_acc={last.get('binary_accuracy')} cat_acc={last.get('sample_accuracy')}"
    )

    st, demo = post_json("/api/demo/seed")
    assert st == 200
    assert demo.get("windows", 0) >= 4
    fc = demo.get("forecast") or {}
    ok.append(
        f"demo: windows={demo.get('windows')} next={fc.get('next_stage') or fc.get('status')} "
        f"wm={demo.get('world_model_ready')}"
    )

    latest = get("/api/latest")
    pred = latest.get("prediction") or {}
    assert pred.get("current_stage")
    ok.append(
        f"prediction: stage={pred.get('current_stage')} risk={pred.get('risk_level')} "
        f"anomaly={pred.get('anomaly_score')} "
        f"forecast={(pred.get('forecast') or {}).get('next_stage')}"
    )

    ev = get("/api/evidence/verify")
    ok.append(f"evidence: valid={ev.get('valid')} blocks={ev.get('blocks')}")

    for path in ["/", "/assets/app.js?v=7.1.9", "/assets/styles.css?v=7.1.9"]:
        with urllib.request.urlopen(BASE + path, timeout=30) as r:
            assert r.status == 200
    ok.append("frontend assets ok")

    print("ALL_CHECKS_PASS")
    for line in ok:
        print("OK:", line)


if __name__ == "__main__":
    main()
