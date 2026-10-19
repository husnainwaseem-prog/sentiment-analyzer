"""Tests for model training and evaluation."""

import pandas as pd

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


def test_train_and_evaluate_picks_best():
    best_name, results, _ = train_mod.train_and_evaluate(tiny_frame())
    assert best_name in ("naive_bayes", "logistic_regression")
    assert set(results) == {"naive_bayes", "logistic_regression"}
    for res in results.values():
        m = res["metrics"]
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "per_class", "confusion_matrix"):
            assert key in m
        assert 0.0 <= m["accuracy"] <= 1.0
        assert len(m["confusion_matrix"]) == 3
        assert set(m["per_class"]) == set(CLASSES)


def test_evaluate_confusion_matrix_shape():
    best_name, results, _ = train_mod.train_and_evaluate(tiny_frame())
    cm = results[best_name]["metrics"]["confusion_matrix"]
    assert all(len(row) == 3 for row in cm)


def test_make_vectorizer_limits_features():
    vec = train_mod.make_vectorizer()
    assert vec.max_features == 20000
