"""
Dedicated FastAPI application exposing:
- /train       : upload new CSV and (incrementally) train the model
- /predict      : pass feature parameters and get a prediction
- /metrics/history : view model evolution history
- /metrics/latest   : latest evaluation metrics
- /model/schema     : required feature names for prediction
- /health
"""
import os
import shutil
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import RAW_DATA_DIR
from src.data_handler import load_csv, archive_processed_file, DataValidationError
from src.train import train_model
from src.predict import predict_single, get_required_features, ModelNotFoundError
from src.evaluate import get_metrics_history, get_latest_metrics

app = FastAPI(
    title="XGBoost Incremental Training Pipeline API",
    description="Train, predict, and evaluate an XGBoost model incrementally.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve HTML UI at /ui
if os.path.isdir("static"):
    app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")


class PredictRequest(BaseModel):
    features: dict


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.post("/train")
async def train_endpoint(file: UploadFile = File(...)):
    """
    Upload a new CSV training file. Saved to data/raw, then used to
    (incrementally) train the XGBoost model. The file is then archived
    to data/processed.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    saved_path = os.path.join(RAW_DATA_DIR, file.filename)
    try:
        with open(saved_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        df = load_csv(saved_path)
        result = train_model(df)
        archive_processed_file(saved_path)

        return JSONResponse(content={
            "message": "Training completed successfully.",
            "result": result,
        })

    except DataValidationError as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        if os.path.exists(saved_path):
            os.remove(saved_path)
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    try:
        prediction = predict_single(request.features)
        return {"prediction": prediction}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/schema")
def model_schema():
    try:
        features = get_required_features()
        return {"required_features": features}
    except ModelNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/metrics/history")
def metrics_history():
    return {"history": get_metrics_history()}


@app.get("/metrics/latest")
def metrics_latest():
    latest = get_latest_metrics()
    if latest is None:
        raise HTTPException(status_code=404, detail="No metrics recorded yet.")
    return latest


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)