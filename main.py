import os
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn


APP_URL = "http://127.0.0.1:8000/dashboard.html"
HEALTH_URL = "http://127.0.0.1:8000/health"


def _wait_for_server(timeout_seconds: int = 15) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.4)
    return False


def _open_browser_when_ready():
    if _wait_for_server():
        webbrowser.open(APP_URL)


if __name__ == "__main__":
    if os.environ.get("RUN_MAIN") != "true":
        threading.Thread(target=_open_browser_when_ready, daemon=True).start()

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
