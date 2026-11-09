import io
import json
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import pytest

from persian_rag.embeddings import (
    Embedder,
    HashingEmbedder,
    OpenAIEmbedder,
    embedder_from_spec,
)
from persian_rag.http_client import APIError
from persian_rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from persian_rag.schema import Chunk, SearchResult


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    e = HashingEmbedder(dim=256)
    a = e.embed(["زعفران خراسان", "متروی تهران"])
    b = HashingEmbedder(dim=256).embed(["زعفران خراسان", "متروی تهران"])
    assert a.shape == (2, 256) and a.dtype == np.float32
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)
    assert isinstance(e, Embedder)


def test_hashing_embedder_similarity_reflects_shared_ngrams() -> None:
    e = HashingEmbedder()
    q, near, far = e.embed(["کتاب‌های تاریخی", "کتاب تاریخ ایران", "متروی تهران"])
    assert float(q @ near) > float(q @ far)


def test_hashing_embedder_empty_text_is_zero_vector() -> None:
    assert not HashingEmbedder(dim=32).embed([""]).any()


def test_embedder_spec_roundtrip() -> None:
    e = HashingEmbedder(dim=128, ngram_range=(1, 3))
    rebuilt = embedder_from_spec(e.spec())
    np.testing.assert_allclose(rebuilt.embed(["سلام"]), e.embed(["سلام"]))
    with pytest.raises(ValueError):
        embedder_from_spec({"type": "magic"})


def test_openai_embedder_batches_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        body = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
        calls.append({"url": req.full_url, "body": body, "auth": req.get_header("Authorization")})
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])
        ]
        return FakeResponse(json.dumps({"data": list(reversed(data))}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    e = OpenAIEmbedder("m", base_url="http://x/v1/", api_key="sk-test", batch_size=2)
    out = e.embed(["ا", "ابجد", "اب"])
    assert out.shape == (3, 2)
    assert [c["url"] for c in calls] == ["http://x/v1/embeddings"] * 2
    assert calls[0]["body"] == {"model": "m", "input": ["ا", "ابجد"]}
    assert calls[0]["auth"] == "Bearer sk-test"
    # Responses were reversed; the client must restore input order.
    expected = np.array([[1, 1], [4, 1], [2, 1]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert "api_key" not in e.spec()


def test_openai_embedder_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"bad key"))  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(APIError) as info:
        OpenAIEmbedder("m", base_url="http://x/v1").embed(["سلام"])
    assert info.value.status == 401 and "bad key" in str(info.value)


def test_dense_retriever_ranks_relevant_chunk(chunks: list[Chunk]) -> None:
    dense = DenseRetriever(chunks, HashingEmbedder())
    assert dense.search("مترو تهران", k=1)[0].chunk.doc_id == "metro"
    assert len(dense.search("مترو", k=99)) <= len(chunks)


def test_dense_retriever_rejects_misaligned_matrix(chunks: list[Chunk]) -> None:
    with pytest.raises(ValueError):
        DenseRetriever(chunks, HashingEmbedder(dim=8), matrix=np.zeros((1, 8), np.float32))


class StaticRetriever:
    def __init__(self, ranking: list[Chunk]) -> None:
        self.ranking = ranking

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        return [SearchResult(c, 1.0 / (i + 1)) for i, c in enumerate(self.ranking[:k])]


def test_rrf_fusion_math(chunks: list[Chunk]) -> None:
    a, b, c, _ = chunks
    hybrid = HybridRetriever([StaticRetriever([a, b]), StaticRetriever([b, c])], rrf_k=60)
    results = hybrid.search("q", k=3)
    assert [r.chunk.doc_id for r in results] == ["metro", "saffron", "hafez"]
    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert results[1].score == pytest.approx(1 / 61)


def test_rrf_weights(chunks: list[Chunk]) -> None:
    a, b, *_ = chunks
    hybrid = HybridRetriever([StaticRetriever([a]), StaticRetriever([b])], weights=[1.0, 3.0])
    assert hybrid.search("q", k=1)[0].chunk is b
    with pytest.raises(ValueError):
        HybridRetriever([StaticRetriever([a])], weights=[1.0, 2.0])


def test_hybrid_end_to_end(chunks: list[Chunk]) -> None:
    hybrid = HybridRetriever([BM25Retriever(chunks), DenseRetriever(chunks, HashingEmbedder())])
    assert hybrid.search("پایتون کی منتشر شد؟", k=1)[0].chunk.doc_id == "python"
