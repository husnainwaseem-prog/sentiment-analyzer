"""Unit tests for text preprocessing."""

from src.preprocess import clean_text, combine_title_text


def test_lowercases_and_strips():
    assert clean_text("  GREAT Product!!  ") == "great product!!"


def test_removes_urls():
    assert "http" not in clean_text("see https://example.com/x for details")


def test_removes_html_tags():
    assert clean_text("<b>bold</b> move") == "bold move"


def test_unescapes_entities():
    assert clean_text("fish &amp; chips") == "fish chips"


def test_non_string_returns_empty():
    assert clean_text(None) == ""
    assert clean_text(123) == ""


def test_collapses_whitespace():
    assert clean_text("a   b\n\tc") == "a b c"


def test_combine_title_and_body():
    out = combine_title_text("Great!", "Loved it")
    assert "great!" in out and "loved it" in out


def test_combine_caps_long_reviews():
    out = combine_title_text("T", "x" * 5000)
    assert len(out) <= 2000
