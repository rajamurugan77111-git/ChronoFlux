import os
import threading
import time
import shutil
import subprocess
import webbrowser
import uvicorn


def open_browser():
    if os.environ.get("CHRONOFLUX_NO_BROWSER") == "1":
        return
    time.sleep(1.4)
    url = "http://127.0.0.1:8000"
    # Prefer Chrome so the demo matches the jury machine look.
    chrome = shutil.which("chrome") or shutil.which("google-chrome")
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        chrome,
    ]
    for path in candidates:
        if not path:
            continue
        try:
            subprocess.Popen([path, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception:
            continue
    webbrowser.open(url)


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=False)
