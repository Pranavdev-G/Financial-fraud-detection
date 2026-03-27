# backend/main.py
import os, shutil
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from services.fraud_service import load_dataset, get_df, is_loaded, get_dataset_info, get_column_map
from ai_model import get_model
from algorithms.divide_conquer import run_divide_conquer, binary_search, merge_sort
from algorithms.greedy import detect_suspicious_greedy
from algorithms.dynamic_programming import run_dynamic_programming

app = FastAPI(title="Fraud Detection ADSA")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "dataset", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _require():
    if not is_loaded():
        raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV first.")
    return get_df()

@app.get("/")
def root():
    return {"message": "Fraud Detection ADSA API running."}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files accepted.")
    save_path = os.path.join(UPLOAD_DIR, "transactions.csv")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = load_dataset(save_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    model = get_model()
    try:
        train_result = model.train_model(get_df())
    except Exception as e:
        train_result = {"status": "training_failed", "error": str(e)}
    return {"status": "ok", "filename": file.filename, "dataset_info": info, "model_training": train_result}

@app.get("/dataset-info")
def dataset_info():
    return get_dataset_info()

@app.get("/ai-predict")
def ai_predict(
    sender: str = Query("ACC001"),
    receiver: str = Query("ACC002"),
    payment_method: str = Query("credit_card"),
    amount: float = Query(5000.0),
):
    model = get_model()
    result = model.predict_transaction(sender, receiver, payment_method, amount)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/divide-conquer")
def divide_conquer():
    df = _require()
    return run_divide_conquer(df)

@app.get("/search")
def search(value: float = Query(...)):
    df = _require()
    amounts = sorted([round(float(x), 2) for x in df["amount"].tolist()])
    idx = binary_search(amounts, round(value, 2))
    return {
        "search_value": value,
        "found": idx != -1,
        "index": idx,
        "nearby": amounts[max(0, idx-2): idx+3] if idx != -1 else [],
    }

@app.get("/greedy")
def greedy():
    df = _require()
    return detect_suspicious_greedy(df)

@app.get("/dynamic")
def dynamic():
    df = _require()
    return run_dynamic_programming(df)
