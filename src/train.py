"""Train and compare two sentiment classifiers, keep the best one.

Models compared (both on TF-IDF features):
  1. Multinomial Naive Bayes
  2. Logistic Regression

The winner (highest macro F1 on the held-out test split) is saved to
models/best_model.joblib together with metrics.json and a confusion
matrix plot.

Usage:
    python -m src.train
"""

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import (
    CLASSES,
    CM_PATH,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAMS,
)
from .preprocess import clean_text


def load_data(csv_path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin(CLASSES)]
    return df.reset_index(drop=True)


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAMS,
        min_df=2,
        sublinear_tf=True,
    )


def candidate_models():
    return {
        "naive_bayes": Pipeline(
            [("tfidf", make_vectorizer()), ("clf", MultinomialNB())]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    report = classification_report(
        y_test, pred, labels=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_test, pred),
        "macro_precision": precision_score(y_test, pred, labels=CLASSES, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, pred, labels=CLASSES, average="macro", zero_division=0),
        "macro_f1": f1_score(y_test, pred, labels=CLASSES, average="macro", zero_division=0),
        "per_class": {
            c: {
                "precision": report[c]["precision"],
                "recall": report[c]["recall"],
                "f1": report[c]["f1-score"],
                "support": int(report[c]["support"]),
            }
            for c in CLASSES
        },
        "confusion_matrix": confusion_matrix(y_test, pred, labels=CLASSES).tolist(),
    }


def plot_confusion_matrix(cm, path=CM_PATH) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (best model, test set)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, cm[i][j], ha="center", va="center",
                color="white" if cm[i][j] > max(max(r) for r in cm) / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train_and_evaluate(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    results = {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        results[name] = {"model": model, "metrics": evaluate(model, X_test, y_test)}
    best_name = max(results, key=lambda n: results[n]["metrics"]["macro_f1"])
    return best_name, results, (X_test, y_test)


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df)} reviews: {df['label'].value_counts().to_dict()}")

    best_name, results, _ = train_and_evaluate(df)

    print(f"\n{'model':<20}{'accuracy':<10}{'precision':<10}{'recall':<10}{'f1':<10}")
    for name, res in results.items():
        m = res["metrics"]
        print(
            f"{name:<20}{m['accuracy']:<10.4f}{m['macro_precision']:<10.4f}"
            f"{m['macro_recall']:<10.4f}{m['macro_f1']:<10.4f}"
        )
    print(f"\nBest model: {best_name}")

    best_metrics = results[best_name]["metrics"]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results[best_name]["model"], MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(
            {"best_model": best_name, "test_size": TEST_SIZE, **best_metrics},
            f,
            indent=2,
        )
    plot_confusion_matrix(best_metrics["confusion_matrix"])
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved confusion matrix -> {CM_PATH}")


if __name__ == "__main__":
    main()
