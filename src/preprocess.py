"""Text cleaning and preprocessing for product reviews."""

import html
import re

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s'!?.,]")
WS_RE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Lowercase, strip HTML/URLs, keep simple punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = text.lower()
    text = NON_TEXT_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    return text


def combine_title_text(title, text) -> str:
    """Join a review title and body into one training string."""
    title = clean_text(title)
    body = clean_text(text)
    combined = f"{title}. {body}" if title else body
    return combined[:2000]  # cap very long reviews
