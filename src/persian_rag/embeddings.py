"""Text embedders: an offline hashing embedder and an OpenAI-compatible HTTP client."""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from persian_rag.http_client import APIError, join_url, post_json
from persian_rag.tokenizer import tokenize


@runtime_checkable
class Embedder(Protocol):
    """Maps texts to L2-normalized float32 vectors of shape ``(n, dim)``."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts``."""
        ...

    def spec(self) -> dict[str, Any]:
        """Return a JSON-serializable description used to rebuild the embedder."""
        ...


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization that leaves all-zero rows untouched."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-character-n-grams embedder (the "hashing trick").

    Each word is padded (``" کتاب "``) and cut into character n-grams; each n-gram is
    hashed with CRC32 into one of ``dim`` buckets with a hash-derived sign. Counts are
    log-scaled and the vector is L2-normalized. Character n-grams make it fairly robust
    to Persian morphology (``کتاب`` / ``کتاب‌ها``) — but not to Arabic vs Persian code
    points, which is why text should be normalized first.

    No model download, no network: ideal for tests, CI and offline demos.
    """

    def __init__(self, dim: int = 1024, ngram_range: tuple[int, int] = (2, 4)) -> None:
        """Create an embedder.

        Args:
            dim: Output dimensionality (number of hash buckets).
            ngram_range: Inclusive (min, max) character n-gram lengths.

        """
        lo, hi = ngram_range
        if dim <= 0 or lo < 1 or hi < lo:
            raise ValueError("invalid dim or ngram_range")
        self.dim = dim
        self.ngram_range = (lo, hi)

    def _features(self, text: str) -> dict[int, float]:
        counts: dict[int, float] = {}
        lo, hi = self.ngram_range
        for word in tokenize(text):
            padded = f" {word.replace(chr(0x200C), '')} "
            for n in range(lo, hi + 1):
                for i in range(len(padded) - n + 1):
                    h = zlib.crc32(padded[i : i + n].encode("utf-8"))
                    bucket = h % self.dim
                    sign = 1.0 if (h >> 31) & 1 == 0 else -1.0
                    counts[bucket] = counts.get(bucket, 0.0) + sign
        return counts

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` into an ``(n, dim)`` float32 matrix."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for bucket, value in self._features(text).items():
                out[row, bucket] = math.copysign(math.log1p(abs(value)), value)
        return l2_normalize(out)

    def spec(self) -> dict[str, Any]:
        """Describe this embedder for persistence."""
        return {"type": "hashing", "dim": self.dim, "ngram_range": list(self.ngram_range)}


class OpenAIEmbedder:
    """Client for any OpenAI-compatible ``/embeddings`` endpoint.

    Works with OpenAI, OpenRouter, vLLM, LM Studio or Ollama
    (``http://localhost:11434/v1`` with e.g. ``model="bge-m3"``).
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        batch_size: int = 64,
        timeout: float = 60.0,
    ) -> None:
        """Configure the client (no request is made until :meth:`embed`)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = batch_size
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed ``texts`` in batches; returns L2-normalized vectors."""
        url = join_url(self.base_url, "embeddings")
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            data = post_json(
                url,
                {"model": self.model, "input": batch},
                api_key=self.api_key,
                timeout=self.timeout,
            )
            items = data.get("data")
            if not isinstance(items, list) or len(items) != len(batch):
                raise APIError(f"expected {len(batch)} embeddings from {url}")
            items = sorted(items, key=lambda d: d.get("index", 0))
            rows.extend(item["embedding"] for item in items)
        if not rows:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(rows, dtype=np.float32))

    def spec(self) -> dict[str, Any]:
        """Describe this embedder for persistence (credentials are never stored)."""
        return {"type": "openai", "model": self.model, "base_url": self.base_url}


def embedder_from_spec(spec: dict[str, Any], *, api_key: str | None = None) -> Embedder:
    """Rebuild an embedder from :meth:`Embedder.spec` output."""
    kind = spec.get("type")
    if kind == "hashing":
        return HashingEmbedder(dim=int(spec["dim"]), ngram_range=tuple(spec["ngram_range"]))
    if kind == "openai":
        return OpenAIEmbedder(spec["model"], base_url=spec["base_url"], api_key=api_key)
    raise ValueError(f"unknown embedder type: {kind!r}")
