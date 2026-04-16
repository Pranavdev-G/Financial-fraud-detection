import os
import socket
import threading
import time
import urllib.error
import urllib.request
import webbrowser

import uvicorn


DEFAULT_PORT = int(os.environ.get("PORT", "8000"))


def _reload_enabled() -> bool:
    value = os.environ.get("UVICORN_RELOAD", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _find_available_port(preferred: int = 8000, search_limit: int = 30) -> int:
    for port in range(preferred, preferred + search_limit + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No free port found between {preferred} and {preferred + search_limit}."
    )


def _wait_for_server(health_url: str, timeout_seconds: int = 15) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.5) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.4)
    return False


def _open_browser_when_ready(app_url: str, health_url: str):
    if _wait_for_server(health_url):
        webbrowser.open(app_url)


if __name__ == "__main__":
    run_port = _find_available_port(DEFAULT_PORT)
    app_url = f"http://127.0.0.1:{run_port}/dashboard.html"
    health_url = f"http://127.0.0.1:{run_port}/health"

    if run_port != DEFAULT_PORT:
        print(f"[launcher] Port {DEFAULT_PORT} busy; using {run_port} instead.")

    if os.environ.get("RUN_MAIN") != "true":
        threading.Thread(
            target=_open_browser_when_ready,
            args=(app_url, health_url),
            daemon=True,
        ).start()

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=run_port,
        reload=_reload_enabled(),
    )
