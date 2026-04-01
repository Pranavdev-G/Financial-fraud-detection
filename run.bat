@echo off
setlocal
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo Virtual environment not found at "%VENV_PY%"
  echo Create it first with: py -m venv .venv
  pause
  exit /b 1
)

echo Installing requirements...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"

echo Starting app...
set "UVICORN_RELOAD=0"
"%VENV_PY%" "%~dp0main.py"
