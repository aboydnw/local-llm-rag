import re
from collections.abc import Iterator

import tiktoken

from rag_lab.types import Document

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")


def get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def document_label(document: Document) -> str:
    return document.metadata.get("source") or document.path.name


def token_windows(
    text: str, encoder: tiktoken.Encoding, max_tokens: int, overlap: int
) -> Iterator[str]:
    tokens = encoder.encode(text)
    if len(tokens) <= max_tokens:
        yield text
        return
    start = 0
    step = max_tokens - overlap
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        yield encoder.decode(window)
        if start + max_tokens >= len(tokens):
            return
        start += step


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def cascade_split(
    text: str, encoder: tiktoken.Encoding, max_tokens: int, overlap: int
) -> Iterator[str]:
    units = _atomize(text, encoder, max_tokens)
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and _count(candidate, encoder) > max_tokens:
            yield current
            tail = _overlap_tail(current, encoder, overlap)
            merged = f"{tail}\n\n{unit}" if tail else unit
            current = merged if _count(merged, encoder) <= max_tokens else unit
        else:
            current = candidate
    if current:
        yield current


def _split_paragraphs(text: str) -> list[str]:
    parts = _PARAGRAPH_BOUNDARY.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _count(text: str, encoder: tiktoken.Encoding) -> int:
    return len(encoder.encode(text))


def _atomize(text: str, encoder: tiktoken.Encoding, max_tokens: int) -> list[str]:
    units: list[str] = []
    for paragraph in _split_paragraphs(text):
        if _count(paragraph, encoder) <= max_tokens:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if _count(sentence, encoder) <= max_tokens:
                units.append(sentence)
            else:
                units.extend(token_windows(sentence, encoder, max_tokens, 0))
    return units


def _overlap_tail(text: str, encoder: tiktoken.Encoding, overlap: int) -> str:
    if overlap <= 0:
        return ""
    tokens = encoder.encode(text)
    return encoder.decode(tokens[-overlap:])
