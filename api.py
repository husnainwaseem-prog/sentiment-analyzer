"""FastAPI service exposing the best sentiment model.

Endpoints:
    GET  /health          service + model status
    GET  /info            model name, classes, test metrics
    POST /predict         {"text": "..."} -> label + confidence + probabilities
    POST /predict/batch  {"texts": ["...", "..."]} -> list of predictions

Run:
    uvicorn api:app --host 127.0.0.1 --port 8000
Docs:
    http://127.0.0.1:8000/docs
"""

import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import CLASSES, METRICS_PATH
from src.predict import predict_one

app = FastAPI(title="Customer Review Sentiment Analyzer")


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class BatchRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=50)


def _metrics() -> dict:
    try:
        with open(METRICS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@app.get("/health")
def health():
    try:
        from src.predict import get_model

        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/info")
def info():
    m = _metrics()
    return {
        "model": m.get("best_model", "unknown"),
        "classes": CLASSES,
        "test_accuracy": round(m.get("accuracy", 0.0), 4),
        "test_macro_f1": round(m.get("macro_f1", 0.0), 4),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return predict_one(req.text)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m src.train")


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    try:
        return {"predictions": [predict_one(t) for t in req.texts]}
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run: python -m src.train")
