"""ChronoFlux live network sensor (Linux, Windows with Npcap, macOS).

Captures packet/flow METADATA only (no payload contents), builds the dual
flow + packet feature window every N seconds and streams it to the ChronoFlux
server, which detects, forecasts, raises alerts and records evidence.

    sudo python live/sensor.py --server http://127.0.0.1:8000
    sudo python live/sensor.py --server http://10.0.0.5:8000 --interface eth0 --api-key KEY
    python live/sensor.py --list-interfaces

The API key can also come from the CHRONOFLUX_SENSOR_KEY environment variable.
Capturing packets needs root on Linux/macOS, or Administrator + Npcap on Windows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ubuntu_sensor import make_payload, new_window, process  # noqa: E402


def list_interfaces() -> None:
    from scapy.all import conf, get_if_list
    print("Available interfaces (default marked *):")
    for name in get_if_list():
        print(f"  {'*' if str(conf.iface) == name else ' '} {name}")


def capture_window(interface: str | None, seconds: int) -> dict:
    from scapy.all import sniff
    window = new_window()
    sniff(iface=interface, timeout=seconds, store=False, prn=lambda p: process(p, window))
    return window


def run(server: str, interface: str | None, seconds: int, sensor_id: str, api_key: str,
        once: bool = False) -> int:
    from scapy.all import conf
    endpoint = server.rstrip("/") + "/api/sensor/live"
    iface = interface or str(conf.iface)
    session_id = f"{sensor_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    headers = {"X-API-Key": api_key} if api_key else {}
    print("ChronoFlux live sensor")
    print(f"  interface : {iface}")
    print(f"  window    : {seconds}s")
    print(f"  server    : {endpoint}  (api key: {'set' if api_key else 'none'})")
    print(f"  session   : {session_id}\n")

    backoff = 1
    while True:
        window = capture_window(interface, seconds)
        features, telemetry = make_payload(window, seconds)
        payload = {"sensor_id": sensor_id, "session_id": session_id,
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "features": features, "telemetry": telemetry}
        try:
            r = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            if r.status_code == 401:
                print("Server rejected the sensor key (401). Check CHRONOFLUX_SENSOR_KEY.")
                return 2
            r.raise_for_status()
            pred = r.json().get("prediction", {})
            fc = pred.get("forecast") or {}
            print(f"[{time.strftime('%H:%M:%S')}] packets={telemetry.get('packet_count', 0):<5} "
                  f"stage={pred.get('current_stage')} risk={pred.get('risk_level')} "
                  f"next={fc.get('next_stage') or '-'}")
            backoff = 1
        except requests.RequestException as exc:
            print(f"[{time.strftime('%H:%M:%S')}] send failed ({exc}); retrying in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
        if once:
            return 0


def main() -> int:
    p = argparse.ArgumentParser(description="ChronoFlux live network sensor")
    p.add_argument("--server", default=os.environ.get("CHRONOFLUX_SERVER", "http://127.0.0.1:8000"))
    p.add_argument("--interface", help="capture interface (default: system default)")
    p.add_argument("--window", type=int, default=10, help="seconds per feature window")
    p.add_argument("--sensor-id", default=os.environ.get("CHRONOFLUX_SENSOR_ID", "gateway-sensor"))
    p.add_argument("--api-key", default=os.environ.get("CHRONOFLUX_SENSOR_KEY", ""))
    p.add_argument("--list-interfaces", action="store_true")
    p.add_argument("--once", action="store_true", help="send one window and exit (testing)")
    args = p.parse_args()
    if args.list_interfaces:
        list_interfaces()
        return 0
    try:
        return run(args.server, args.interface, args.window, args.sensor_id, args.api_key, args.once)
    except PermissionError:
        print("Packet capture needs root (Linux/macOS) or Administrator + Npcap (Windows).")
        return 1
    except KeyboardInterrupt:
        print("\nSensor stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
