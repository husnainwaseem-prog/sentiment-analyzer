"""Load the trained model and predict sentiment for new reviews.

Usage:
    python -m src.predict "This blender is amazing, best purchase ever!"
"""

import sys

import joblib
import numpy as np

from .config import CLASSES, MODEL_PATH
from .preprocess import clean_text

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_one(text: str) -> dict:
    model = get_model()
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "your review text here"')
        sys.exit(1)
    result = predict_one(" ".join(sys.argv[1:]))
    print(f"label: {result['label']} (confidence {result['confidence']})")
    print(f"probabilities: {result['probabilities']}")


if __name__ == "__main__":
    main()
