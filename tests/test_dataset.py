"""Unit tests for dataset building helpers."""

import src.build_dataset as bd


def test_rating_to_label():
    assert bd.rating_to_label(1) == "negative"
    assert bd.rating_to_label(2.0) == "negative"
    assert bd.rating_to_label(3) == "neutral"
    assert bd.rating_to_label(4) == "positive"
    assert bd.rating_to_label(5.0) == "positive"


def test_review_text_combines_title_and_body():
    rec = {"title": "Great", "text": "Loved it", "rating": 5}
    assert bd.review_text(rec) == "Great. Loved it"


def test_review_text_handles_missing_parts():
    assert bd.review_text({"title": "", "text": "ok", "rating": 3}) == "ok"
    assert bd.review_text({"title": "T", "text": "", "rating": 3}) == "T"


def test_build_balanced_sample(monkeypatch):
    fake = []
    for i in range(10):
        fake.append({"title": f"t{i}", "text": "bad terrible awful product", "rating": 1})
    for i in range(10):
        fake.append({"title": f"t{i}", "text": "it is okay average product", "rating": 3})
    for i in range(10):
        fake.append({"title": f"t{i}", "text": "excellent amazing wonderful product", "rating": 5})

    monkeypatch.setattr(bd, "stream_reviews", lambda url: iter(fake))
    rows, scanned = bd.build_balanced_sample(per_class=4, seed=1)

    # sampler stops early once every class bucket is full (4+4+4 = first 24 rows)
    assert scanned == 24
    assert len(rows) == 12
    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    assert counts == {"negative": 4, "neutral": 4, "positive": 4}
