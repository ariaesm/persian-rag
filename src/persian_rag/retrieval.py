"""Retrievers: BM25 (lexical), dense (embedding) and hybrid (Reciprocal Rank Fusion)."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from persian_rag.schema import Chunk, SearchResult
from persian_rag.tokenizer import Analyzer


@runtime_checkable
class Retriever(Protocol):
    """Anything that can rank chunks for a query."""

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return up to ``k`` results, best first."""
        ...


def _top_k(scores: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Return the ``k`` highest-scoring (index, score) pairs, ties broken by index."""
    return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))


class BM25Retriever:
    """Okapi BM25 over an inverted index, implemented from scratch.

    Scoring uses the Lucene-style non-negative IDF
    ``log(1 + (N - df + 0.5) / (df + 0.5))`` and the classic term-frequency
    saturation ``tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))``.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        analyzer: Callable[[str], list[str]] | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Index ``chunks``.

        Args:
            chunks: Passages to index.
            analyzer: Text → terms function; defaults to the full Persian :class:`Analyzer`.
            k1: Term-frequency saturation parameter.
            b: Length normalization strength (0 = none, 1 = full).

        """
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("expected k1 >= 0 and 0 <= b <= 1")
        self.chunks = list(chunks)
        self.analyzer = analyzer or Analyzer()
        self.k1 = k1
        self.b = b

        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []
        for i, chunk in enumerate(self.chunks):
            terms = Counter(self.analyzer(chunk.text))
            self._lengths.append(sum(terms.values()))
            for term, tf in terms.items():
                self._postings[term].append((i, tf))
        n = len(self.chunks)
        self._avgdl = (sum(self._lengths) / n) if n else 0.0
        self._idf = {
            term: math.log(1.0 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct indexed terms."""
        return len(self._postings)

    def scores(self, query: str) -> dict[int, float]:
        """Return BM25 scores for every chunk sharing at least one query term."""
        scores: dict[int, float] = defaultdict(float)
        avgdl = self._avgdl or 1.0
        for term, qtf in Counter(self.analyzer(query)).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self._idf[term]
            for i, tf in postings:
                norm = self.k1 * (1.0 - self.b + self.b * self._lengths[i] / avgdl)
                scores[i] += qtf * idf * tf * (self.k1 + 1.0) / (tf + norm)
        return dict(scores)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top ``k`` chunks by BM25 score (only chunks with score > 0)."""
        ranked = _top_k(self.scores(query), k)
        return [SearchResult(self.chunks[i], s) for i, s in ranked if s > 0]
