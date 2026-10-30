"""Quick live smoke test against a running ChronoFlux server."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=45) as resp:
        return resp.status, resp.read()


def post(path: str):
    req = urllib.request.Request(BASE + path, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.status, json.loads(resp.read().decode())


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # Static files / required IDs
    try:
        st, body = get("/")
        html = body.decode("utf-8", "replace")
        needed = [
            "page-overview",
            "page-simulation",
            "page-live",
            "page-devices",
            "page-offline",
            "page-forecast",
            "page-alerts",
            "page-evidence",
            "page-system",
            "seedDemoButton",
            "simPlayButton",
            "statusBanner",
            "currentState",
            "forecastNext",
            "styles.css?v=7.1.12",
            "app.js?v=7.1.12",
        ]
        missing = [x for x in needed if x not in html]
        checks.append(("UI shell", st == 200 and not missing, missing and f"missing={missing}" or "ok"))
    except Exception as exc:
        checks.append(("UI shell", False, str(exc)))

    try:
        st, body = get("/assets/styles.css?v=7.1.12")
        css = body.decode("utf-8", "replace")
        checks.append(
            (
                "CSS compat",
                st == 200 and "color-mix" not in css and "sim-pipeline" in css,
                f"len={len(css)} color-mix={'yes' if 'color-mix' in css else 'no'}",
            )
        )
    except Exception as exc:
        checks.append(("CSS compat", False, str(exc)))

    try:
        st, body = get("/assets/app.js?v=7.1.12")
        js = body.decode("utf-8", "replace")
        checks.append(
            (
                "JS simulation",
                st == 200 and "simSteps" in js and "playSimulation" in js and "analyze-csv-stream" in js,
                f"len={len(js)}",
            )
        )
    except Exception as exc:
        checks.append(("JS simulation", False, str(exc)))

    # Core API
    for path in ["/api/health", "/api/system", "/api/latest", "/api/history?limit=20", "/api/evidence?limit=10"]:
        try:
            st, body = get(path)
            ok = st == 200
            note = ""
            if path == "/api/system":
                data = json.loads(body)
                ok = ok and bool(data.get("models", {}).get("world_model"))
                note = f"world_model={data.get('models', {}).get('world_model')}"
            checks.append((path, ok, note or f"status={st}"))
        except Exception as exc:
            checks.append((path, False, str(exc)))

    # Demo seed + forecast
    try:
        st, data = post("/api/demo/seed")
        fc = data.get("forecast") or {}
        ok = (
            st == 200
            and data.get("world_model_ready")
            and int(data.get("windows") or 0) >= 4
            and fc.get("model_name")
        )
        checks.append(
            (
                "demo seed",
                ok,
                f"windows={data.get('windows')} ready={data.get('world_model_ready')} "
                f"status={fc.get('status')} next={fc.get('next_stage')} model={fc.get('model_name')}",
            )
        )
    except Exception as exc:
        checks.append(("demo seed", False, str(exc)))

    try:
        st, body = get("/api/latest")
        data = json.loads(body)
        pred = data.get("prediction") or {}
        fc = pred.get("forecast") or {}
        ok = st == 200 and bool(pred.get("current_stage")) and bool(fc)
        checks.append(
            (
                "latest after seed",
                ok,
                f"stage={pred.get('current_stage')} fc={fc.get('status')}/{fc.get('next_stage')}",
            )
        )
    except Exception as exc:
        checks.append(("latest after seed", False, str(exc)))

    # Offline sample CSV if present
    sample = ROOT / "data" / "sample_mixed_traffic.csv"
    if sample.exists():
        try:
            import urllib.request

            boundary = "----ChronoFluxBoundary"
            content = sample.read_bytes()
            body = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="sample_mixed_traffic.csv"\r\n'
                f"Content-Type: text/csv\r\n\r\n"
            ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                BASE + "/api/offline/analyze-csv?max_rows=40",
                data=body,
                method="POST",
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode())
            ok = resp.status == 200 and int(payload.get("analyzed_rows") or 0) > 0
            checks.append(("offline csv", ok, f"analyzed={payload.get('analyzed_rows')}"))
        except Exception as exc:
            checks.append(("offline csv", False, str(exc)))

    # World model load locally
    try:
        import sys

        sys.path.insert(0, str(ROOT))
        from core.world_model import load_world_model, world_model_ready

        model = load_world_model()
        checks.append(("world model load", world_model_ready() and model is not None, "ok"))
    except Exception as exc:
        checks.append(("world model load", False, str(exc)))

    failed = 0
    for name, ok, note in checks:
        print(("PASS" if ok else "FAIL"), "-", name, "-", note)
        if not ok:
            failed += 1
    print("SUMMARY", "ALL_PASS" if failed == 0 else f"{failed}_FAILED")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
