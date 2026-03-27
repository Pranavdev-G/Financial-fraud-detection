import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser


BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_HEALTH_URL = f"{BACKEND_URL}/health"
FRONTEND_URL = "http://127.0.0.1:5500"


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _backend_healthy() -> bool:
    try:
        with urllib.request.urlopen(BACKEND_HEALTH_URL, timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_backend(timeout_seconds: int = 12) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _backend_healthy():
            return True
        time.sleep(0.5)
    return False


def _frontend_healthy(frontend_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{frontend_url}/dashboard.html", timeout=1.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_frontend(frontend_url: str, timeout_seconds: int = 10) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _frontend_healthy(frontend_url):
            return True
        time.sleep(0.5)
    return False


def main():
    print("Starting Fraud Detection System...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_path = os.path.join(base_dir, "backend")
    frontend_path = os.path.join(base_dir, "frontend")

    backend = None
    frontend = None

    if _backend_healthy():
        print(f"Backend already running at: {BACKEND_URL}")
    elif _is_port_open("127.0.0.1", 8000):
        print("Port 8000 is already in use by another process, so the backend could not be started.")
        print("Free port 8000 or stop the conflicting process, then run this launcher again.")
        return
    else:
        print("Starting backend...")
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--reload"],
            cwd=backend_path,
        )
        if not _wait_for_backend():
            print("Backend failed to become healthy on http://127.0.0.1:8000.")
            if backend.poll() is None:
                backend.terminate()
            return
        print(f"Backend running at: {BACKEND_URL}")

    if _frontend_healthy(FRONTEND_URL):
        print(f"Frontend already running at: {FRONTEND_URL}")
    elif _is_port_open("127.0.0.1", 5500):
        print("Port 5500 is already in use by another process, so the frontend could not be started.")
        print("Free port 5500 and run the launcher again.")
        return
    else:
        print("Starting frontend...")
        frontend = subprocess.Popen(
            [sys.executable, "-m", "http.server", "5500"],
            cwd=frontend_path,
            creationflags=_creation_flags(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not _wait_for_frontend(FRONTEND_URL):
            print(f"Frontend failed to become healthy on {FRONTEND_URL}.")
            return
        print(f"Frontend running at: {FRONTEND_URL}")

    webbrowser.open(f"{FRONTEND_URL}/dashboard.html")


if __name__ == "__main__":
    main()
