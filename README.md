# Customer Review Sentiment Analyzer

A sentiment analysis system that classifies product reviews as **positive**,
**negative**, or **neutral**. Built with scikit-learn and served through a
FastAPI API.

## How it works

1. **Dataset** — 24,000 Amazon product reviews (Beauty category), balanced
   8,000 per class. Star ratings are mapped to sentiment: 1–2 → negative,
   3 → neutral, 4–5 → positive. See `data/DATASET.md` for the source.
2. **Preprocessing** — HTML unescaping, tag/URL removal, lowercasing,
   punctuation cleanup, whitespace collapsing (`src/preprocess.py`).
3. **Models compared** — TF-IDF features with two classifiers:
   - Multinomial Naive Bayes → accuracy 0.7538, macro F1 0.7547
   - Logistic Regression → **accuracy 0.7654, macro F1 0.7657** (selected)
4. **Serving** — the winning pipeline is exposed via CLI and FastAPI with
   per-class probabilities.

## Project structure

```
sentiment-analyzer/
├── src/
│   ├── build_dataset.py   stream + balance the public dataset -> data/reviews.csv
│   ├── preprocess.py      text cleaning
│   ├── train.py           train both models, compare, save the best
│   ├── predict.py         load model, predict from CLI
│   └── config.py          paths and hyperparameters
├── api.py                 FastAPI service
├── data/reviews.csv       24k balanced reviews (text,label)
├── models/
│   ├── best_model.joblib  trained winning pipeline
│   ├── metrics.json       test metrics
│   └── confusion_matrix.png
├── tests/                 pytest suite (20 tests)
├── REPORT.md              evaluation report
└── Dockerfile
```

## Setup

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Windows (cmd):        .venv\Scripts\activate
# macOS/Linux:          source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train (uses `data/reviews.csv`; rebuild it with `python -m src.build_dataset`
if deleted):

```bash
python -m src.train
```

Predict from the command line:

```bash
python -m src.predict "This cream is amazing, my skin loves it!"
```

Run the API:

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/docs

| Method | Endpoint         | Description                              |
| ------ | ---------------- | ---------------------------------------- |
| GET    | /health          | service status + whether model is loaded |
| GET    | /info            | model name, classes, test accuracy / F1  |
| POST   | /predict         | `{"text": "..."}` → label + probabilities |
| POST   | /predict/batch   | `{"texts": [...]}` → list of predictions |

Run the tests:

```bash
pytest tests/ -q
```

## Features

- Real public dataset (Amazon Reviews 2023), class-balanced, reproducible
- Two-model comparison with accuracy, precision, recall, F1 and confusion matrix
- Best model served via CLI and REST API with confidence scores
- Confusion matrix plot and metrics saved with every training run
- Dockerfile for deployment

## Limitations

- Bag-of-words TF-IDF: word order and sarcasm are mostly lost; 3-star
  "neutral" reviews are the hardest class (F1 ≈ 0.69).
- Trained on Beauty-category reviews; slang or other domains may perform worse.
- The default model is extractive-style classification, not generative.
