"""Shared configuration for the sentiment analyzer project."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = Path(os.getenv("SENTIMENT_DATA_PATH", BASE_DIR / "data" / "reviews.csv"))
MODEL_PATH = Path(os.getenv("SENTIMENT_MODEL_PATH", BASE_DIR / "models" / "best_model.joblib"))
METRICS_PATH = Path(os.getenv("SENTIMENT_METRICS_PATH", BASE_DIR / "models" / "metrics.json"))
CM_PATH = Path(os.getenv("SENTIMENT_CM_PATH", BASE_DIR / "models" / "confusion_matrix.png"))

CLASSES = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAMS = (1, 2)
