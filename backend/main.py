import os
import shutil
from threading import Lock, Thread
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio

from .services.fraud_service import load_dataset, get_df, is_loaded, get_dataset_info
from .ai_model import get_model
from .algorithms.divide_conquer import run_divide_conquer
from .algorithms.greedy import detect_suspicious_greedy
from .algorithms.dynamic_programming import run_dynamic_programming
from .algorithms.hashing_implementation import run_hashing_analysis, search_sender_in_hash
from .risk_engine import simulate_transaction, build_dashboard_payload
from .caches import caches

app = FastAPI(title="Fraud Detection ADSA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "dataset", "uploads")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
os.makedirs(UPLOAD_DIR, exist_ok=True)
_warmup_lock = Lock()


def _reload_enabled() -> bool:
    value = os.environ.get("UVICORN_RELOAD", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _warm_model_and_caches(df_snapshot):
    if not _warmup_lock.acquire(blocking=False):
        return

    try:
        caches.preload_all(df_snapshot)
        model = get_model()
        try:
            model.train_model(df_snapshot)
            if caches.pre_features_df is not None:
                caches.ml_preds = model.score_dataframe(caches.pre_features_df)
        except Exception as exc:
            model.training_summary = {
                "status": "training_failed",
                "model_name": model.model_name,
                "error": str(exc),
            }
    finally:
        _warmup_lock.release()

def _require():
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV first.")
    return get_df()

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/dashboard.html", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "dashboard.html")

@app.get("/index.html", include_in_schema=False)
def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    save_path = os.path.join(UPLOAD_DIR, "transactions.csv")
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f, length=1024 * 1024)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    finally:
        await file.close()
    try:
        info = load_dataset(save_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    df_snapshot = get_df()
    model = get_model()
    model.training_summary = {"status": "queued", "model_name": model.model_name}
    Thread(target=_warm_model_and_caches, args=(df_snapshot,), daemon=True).start()

    return {
        "status": "ok",
        "filename": filename,
        "dataset_info": info,
        "model_training": model.training_summary,
    }

@app.get("/dataset-info")
def dataset_info():
    return get_dataset_info()

@app.get("/ai-predict")
async def ai_predict(
    sender: str = Query("ACC001"),
    receiver: str = Query("ACC002"),
    payment_method: str = Query("credit_card"),
    amount: float = Query(5000.0),
    location: str = Query("Mumbai"),
    hour: int = Query(14),
):
    df = _require()
    return await asyncio.to_thread(simulate_transaction, df, sender, receiver, payment_method, amount, location, hour)

@app.get("/simulate-transaction/{batch_id}")
async def batch_simulate(batch_id: str, transactions: list = Query([])):
    df = _require()
    results = []
    for txn in transactions[:10]:  # Batch limit 10
        result = await asyncio.to_thread(simulate_transaction, df, **txn)
        results.append(result)
    return {"batch_id": batch_id, "results": results}

@app.get("/simulate-transaction")
async def simulate(
    sender: str = Query("ACC001"),
    receiver: str = Query("ACC002"),
    payment_method: str = Query("credit_card"),
    amount: float = Query(5000.0),
    location: str = Query("Mumbai"),
    hour: int = Query(14),
):
    df = _require()
    return await asyncio.to_thread(simulate_transaction, df, sender, receiver, payment_method, amount, location, hour)

@app.get("/fraud-dashboard-data")
async def fraud_dashboard_data():
    df = _require()
    return await asyncio.to_thread(build_dashboard_payload, df)

@app.get("/divide-conquer")
async def divide_conquer():
    df = _require()
    return await asyncio.to_thread(run_divide_conquer, df)

@app.get("/greedy")
async def greedy():
    df = _require()
    return await asyncio.to_thread(detect_suspicious_greedy, df)

@app.get("/dynamic")
async def dynamic():
    df = _require()
    return await asyncio.to_thread(run_dynamic_programming, df)

@app.get("/hashing")
async def hashing():
    df = _require()
    return await asyncio.to_thread(run_hashing_analysis, df)

@app.get("/hash-search")
async def hash_search(sender: str = Query(...)):
    df = _require()
    return await asyncio.to_thread(search_sender_in_hash, df, sender)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=_reload_enabled(),
    )
