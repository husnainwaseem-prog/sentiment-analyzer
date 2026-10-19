# Dataset

**Source:** `McAuley-Lab/Amazon-Reviews-2023` (public HuggingFace dataset),
file `raw/review_categories/All_Beauty.jsonl`
(https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

**License note:** the dataset is published for research/educational use.
Only a 24,000-review sample is redistributed here in `reviews.csv`.

**Sampling:** `python -m src.build_dataset` streams the JSONL and keeps a
balanced sample — star rating 1–2 → `negative`, 3 → `neutral`, 4–5 →
`positive`. Reviews shorter than 20 characters were skipped. Sampling is
seeded (`random_state=42`), so re-running reproduces the same CSV.

**Class distribution:**

| label    | count |
| -------- | ----- |
| negative | 8,000 |
| neutral  | 8,000 |
| positive | 8,000 |

**Columns:** `text` (review title + body), `label` (negative/neutral/positive).
