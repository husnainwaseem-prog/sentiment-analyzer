"""Tests for the FastAPI service."""

import importlib
import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import src.train as train_mod
from src.config import CLASSES


def tiny_frame() -> pd.DataFrame:
    rows = [
        ("terrible awful horrible product", "negative"),
        ("worst purchase ever bad", "negative"),
        ("hate it broken junk", "negative"),
        ("disappointing poor quality", "negative"),
        ("it is okay average fine", "neutral"),
        ("mediocre neither good nor bad", "neutral"),
        ("acceptable decent product", "neutral"),
        ("average quality as expected", "neutral"),
        ("excellent amazing wonderful", "positive"),
        ("love it best purchase", "positive"),
        ("great quality highly recommend", "positive"),
        ("fantastic works perfectly", "positive"),
    ]
    return pd.DataFrame(rows * 4, columns=["text", "label"])


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Train a tiny real model and point the API at it.
    best_name, results, _ = train_mod.train_and_evaluate(tiny_frame())
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    joblib.dump(results[best_name]["model"], model_path)
    with open(metrics_path, "w") as f:
        json.dump({"best_model": best_name, "accuracy": 0.9, "macro_f1": 0.9}, f)

    monkeypatch.setenv("SENTIMENT_MODEL_PATH", str(model_path))
    monkeypatch.setenv("SENTIMENT_METRICS_PATH", str(metrics_path))

    import src.config
    import src.predict
    import api

    importlib.reload(src.config)
    importlib.reload(src.predict)
    importlib.reload(api)
    return TestClient(api.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is True


def test_info(client):
    r = client.get("/info")
    assert r.status_code == 200
    assert r.json()["classes"] == CLASSES


def test_predict(client):
    r = client.post("/predict", json={"text": "This is an excellent amazing product"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"]) == set(CLASSES)


def test_predict_rejects_empty(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"texts": ["love it", "hate it"]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2
