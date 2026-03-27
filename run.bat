@echo off
echo Installing requirements...
pip install -r requirements.txt
echo Starting backend...
cd backend
start "" uvicorn main:app --reload --port 8000
cd ..
echo Opening dashboard...
timeout /t 3
start "" frontend/index.html
pause
