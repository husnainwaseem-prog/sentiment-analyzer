"""Build a balanced 3-class review dataset from a public Amazon reviews dump.

Source: McAuley-Lab/Amazon-Reviews-2023, raw/review_categories/All_Beauty.jsonl
(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

The file is streamed line-by-line (it is large), so only a balanced sample is
kept: star ratings 1-2 -> negative, 3 -> neutral, 4-5 -> positive.

Usage:
    python -m src.build_dataset [--per-class 8000] [--out data/reviews.csv]
(run from the project root)
"""

import argparse
import csv
import json
import random
import urllib.request
from pathlib import Path

from .config import BASE_DIR, RANDOM_STATE

DATA_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/All_Beauty.jsonl"
)

LABEL_ORDER = ["negative", "neutral", "positive"]


def rating_to_label(rating: float) -> str:
    r = float(rating)
    if r <= 2:
        return "negative"
    if r <= 3:
        return "neutral"
    return "positive"


def stream_reviews(url: str):
    """Yield parsed JSON review objects from the remote .jsonl file."""
    req = urllib.request.Request(url, headers={"User-Agent": "sentiment-analyzer/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def review_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("text") or "").strip()
    if title and body:
        return f"{title}. {body}"
    return title or body


def build_balanced_sample(per_class: int = 8000, seed: int = RANDOM_STATE):
    """Stream the public dump and collect a class-balanced sample."""
    buckets = {label: [] for label in LABEL_ORDER}
    scanned = 0
    for record in stream_reviews(DATA_URL):
        scanned += 1
        try:
            label = rating_to_label(record.get("rating", 0))
        except (TypeError, ValueError):
            continue
        if len(buckets[label]) >= per_class:
            continue
        text = review_text(record)
        if len(text) < 20:  # skip empty / trivial reviews
            continue
        buckets[label].append(text)
        if all(len(b) >= per_class for b in buckets.values()):
            break
    rows = [(text, label) for label, texts in buckets.items() for text in texts]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows, scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=8000)
    parser.add_argument("--out", type=str, default=str(BASE_DIR / "data" / "reviews.csv"))
    args = parser.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        print(f"{out_path} already exists, skipping download.")
        return

    print(f"Streaming {DATA_URL} ...")
    rows, scanned = build_balanced_sample(per_class=args.per_class)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    counts = {label: sum(1 for _, l in rows if l == label) for label in LABEL_ORDER}
    print(f"Scanned {scanned} reviews, kept {len(rows)}: {counts}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
