"""Start the ChronoFlux server (production: uvicorn ASGI, no auto-reload).

    python run.py                         # http://127.0.0.1:8000
    CHRONOFLUX_HOST=0.0.0.0 python run.py # listen on the network
"""
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


def _load_dotenv():
    env = Path(__file__).resolve().parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def open_browser(url: str):
    if os.environ.get("CHRONOFLUX_NO_BROWSER") == "1":
        return
    time.sleep(1.4)
    chrome = shutil.which("chrome") or shutil.which("google-chrome")
    for path in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe", chrome):
        if path:
            try:
                subprocess.Popen([path, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
    webbrowser.open(url)


if __name__ == "__main__":
    _load_dotenv()
    host = os.environ.get("CHRONOFLUX_HOST", "0.0.0.0")
    port = int(os.environ.get("CHRONOFLUX_PORT", "8000"))
    print("=" * 60)
    print(" ChronoFlux - AI-Based Network Attack Forecasting")
    print(f" Console : http://127.0.0.1:{port}   (login: {os.environ.get('CHRONOFLUX_AUTH', '1') != '0'})")
    print(f" Sensor  : POST /api/sensor/live  (key: {'set' if os.environ.get('CHRONOFLUX_SENSOR_KEY') else 'NOT SET'})")
    print("=" * 60)
    threading.Thread(target=open_browser, args=(f"http://127.0.0.1:{port}",), daemon=True).start()
    uvicorn.run("backend.api:app", host=host, port=port, reload=False, workers=1,
                proxy_headers=True, forwarded_allow_ips=os.environ.get("CHRONOFLUX_TRUSTED_PROXIES", "127.0.0.1"),
                log_level=os.environ.get("CHRONOFLUX_LOG_LEVEL", "info"))
