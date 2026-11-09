"""Sentence-aware chunking with token-based size and overlap."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from persian_rag.schema import Chunk, Document
from persian_rag.tokenizer import count_tokens

# Sentence ends at ". ! ? ؟ …" followed by whitespace, or at any line break.
# Requiring whitespace after "." keeps decimals ("3.14") and file names intact.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟…])\s+|\s*\n+\s*")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on Persian/Latin terminators and newlines.

    >>> split_sentences("سلام. حالت چطور است؟ خوبم!")
    ['سلام.', 'حالت چطور است؟', 'خوبم!']
    """
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s and s.strip()]


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    tokens: int


def _units(text: str, chunk_size: int) -> list[_Unit]:
    """Turn text into sentence units, splitting sentences longer than ``chunk_size``."""
    units: list[_Unit] = []
    for sentence in split_sentences(text):
        n = count_tokens(sentence)
        if n == 0:
            # Pure punctuation/symbols: glue onto the previous unit, if any.
            if units:
                prev = units[-1]
                units[-1] = _Unit(f"{prev.text} {sentence}", prev.tokens)
            continue
        if n <= chunk_size:
            units.append(_Unit(sentence, n))
            continue
        words = sentence.split()
        for start in range(0, len(words), chunk_size):
            piece = " ".join(words[start : start + chunk_size])
            units.append(_Unit(piece, count_tokens(piece)))
    return units


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Pack whole sentences into chunks of at most ``chunk_size`` tokens.

    Consecutive chunks share trailing sentences totalling at most ``overlap`` tokens,
    so a fact split across a boundary is still retrievable from one chunk.

    Args:
        text: Text to split.
        chunk_size: Maximum tokens per chunk (a single over-long sentence is split
            on whitespace to respect this).
        overlap: Maximum tokens repeated from the end of the previous chunk.

    Raises:
        ValueError: If the sizes are inconsistent.

    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks: list[str] = []
    current: list[_Unit] = []
    current_tokens = 0
    for unit in _units(text, chunk_size):
        if current and current_tokens + unit.tokens > chunk_size:
            chunks.append(" ".join(u.text for u in current))
            kept: list[_Unit] = []
            kept_tokens = 0
            for u in reversed(current):
                if kept_tokens + u.tokens > overlap:
                    break
                kept.insert(0, u)
                kept_tokens += u.tokens
            while kept and kept_tokens + unit.tokens > chunk_size:
                kept_tokens -= kept.pop(0).tokens
            current, current_tokens = kept, kept_tokens
        current.append(unit)
        current_tokens += unit.tokens
    if current:
        chunks.append(" ".join(u.text for u in current))
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size: int = 120, overlap: int = 30
) -> list[Chunk]:
    """Chunk every document, producing ids of the form ``"<doc_id>#<n>"``."""
    out: list[Chunk] = []
    for doc in documents:
        for i, text in enumerate(chunk_text(doc.text, chunk_size, overlap)):
            out.append(
                Chunk(
                    id=f"{doc.id}#{i}",
                    doc_id=doc.id,
                    index=i,
                    text=text,
                    metadata=dict(doc.metadata),
                )
            )
    return out
