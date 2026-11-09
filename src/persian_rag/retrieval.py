"""Retrievers: BM25 (lexical), dense (embedding) and hybrid (Reciprocal Rank Fusion)."""

from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from persian_rag.embeddings import Embedder
from persian_rag.normalizer import Normalizer
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


class DenseRetriever:
    """Cosine-similarity search over embeddings held in a numpy matrix."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        preprocess: Callable[[str], str] | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        """Embed (or adopt pre-computed embeddings for) ``chunks``.

        Args:
            chunks: Passages to index.
            embedder: Any :class:`~persian_rag.embeddings.Embedder`.
            preprocess: Applied to chunk texts and queries before embedding
                (defaults to the Persian normalizer).
            matrix: Pre-computed, row-aligned embeddings (e.g. loaded from disk).

        """
        self.chunks = list(chunks)
        self.embedder = embedder
        self.preprocess = preprocess if preprocess is not None else Normalizer()
        if matrix is None:
            texts = [self.preprocess(c.text) for c in self.chunks]
            matrix = embedder.embed(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        if matrix.shape[0] != len(self.chunks):
            raise ValueError("embedding matrix rows must match the number of chunks")
        self.matrix = matrix.astype(np.float32, copy=False)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the ``k`` chunks with the highest cosine similarity to ``query``."""
        if not self.chunks or k <= 0:
            return []
        q = self.embedder.embed([self.preprocess(query)])[0]
        sims = self.matrix @ q
        k = min(k, len(self.chunks))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.lexsort((idx, -sims[idx]))]
        return [SearchResult(self.chunks[i], float(sims[i])) for i in idx if sims[i] > 0]


class HybridRetriever:
    """Fuse several retrievers with (weighted) Reciprocal Rank Fusion.

    ``score(d) = Σ_r w_r / (rrf_k + rank_r(d))`` with 1-based ranks. RRF only uses
    ranks, so it combines BM25 and cosine scores without any calibration.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> None:
        """Configure fusion.

        Args:
            retrievers: Retrievers to fuse.
            weights: Optional per-retriever weights (default: all 1.0).
            rrf_k: RRF smoothing constant (60 in the original paper).
            candidates: How many results to pull from each retriever before fusing.

        """
        if not retrievers:
            raise ValueError("need at least one retriever")
        self.retrievers = list(retrievers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.retrievers)
        if len(self.weights) != len(self.retrievers):
            raise ValueError("weights must match retrievers")
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the top ``k`` chunks by fused RRF score."""
        fused: dict[str, float] = defaultdict(float)
        by_id: dict[str, Chunk] = {}
        for retriever, weight in zip(self.retrievers, self.weights, strict=True):
            for rank, hit in enumerate(retriever.search(query, self.candidates), start=1):
                fused[hit.chunk.id] += weight / (self.rrf_k + rank)
                by_id.setdefault(hit.chunk.id, hit.chunk)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        return [SearchResult(by_id[cid], score) for cid, score in ranked]
