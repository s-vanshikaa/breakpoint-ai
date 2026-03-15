import re

from guardrails.patterns import find_adversarial_pattern

HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def strip_html_comments(text: str) -> str:
    return HTML_COMMENT_RE.sub("", text).strip()


def filter_retrieved_chunks(chunks: list) -> list[tuple[str, str, float]]:
    """Sanitizes retrieved chunks. Accepts objects with .text/.source/.score
    attributes (duck-typed to avoid depending on any target app's schema) and
    returns (text, source, score) tuples, dropping chunks that still contain
    suspicious instruction-like content after comment stripping."""
    filtered = []
    for chunk in chunks:
        cleaned_text = strip_html_comments(chunk.text)
        if find_adversarial_pattern(cleaned_text):
            continue
        filtered.append((cleaned_text, chunk.source, chunk.score))
    return filtered
