import json
from pathlib import Path

import numpy as np
import pytest

from persian_rag.embeddings import HashingEmbedder
from persian_rag.index import Index
from persian_rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from persian_rag.schema import Document

DOCS = [
    Document("tea.txt", "چای در لاهیجان کشت می‌شود. باغ‌های چای سرسبز هستند.", {"source": "tea.txt"}),
    Document("rice.txt", "برنج در گیلان و مازندران کاشته می‌شود.", {"source": "rice.txt"}),
]


def test_build_save_load_roundtrip(tmp_path: Path) -> None:
    index = Index.build(DOCS, embedder=HashingEmbedder(dim=64), chunk_size=50, overlap=10)
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.chunks == index.chunks
    np.testing.assert_array_equal(loaded.embeddings, index.embeddings)
    assert loaded.embedder.spec() == index.embedder.spec()
    assert loaded.settings == {"chunk_size": 50, "overlap": 10}
    meta = json.loads((tmp_path / "idx" / "meta.json").read_text("utf-8"))
    assert meta["num_chunks"] == len(index.chunks)


def test_saved_json_is_readable_utf8(tmp_path: Path) -> None:
    Index.build(DOCS).save(tmp_path)
    assert "لاهیجان" in (tmp_path / "chunks.json").read_text("utf-8")


def test_retriever_modes(tmp_path: Path) -> None:
    index = Index.build(DOCS)
    assert isinstance(index.retriever("bm25"), BM25Retriever)
    assert isinstance(index.retriever("dense"), DenseRetriever)
    assert isinstance(index.retriever("hybrid"), HybridRetriever)
    assert index.retriever("hybrid").search("چاي لاهيجان", k=1)[0].chunk.doc_id == "tea.txt"
    with pytest.raises(ValueError):
        index.retriever("magic")  # type: ignore[arg-type]


def test_load_missing_or_corrupt(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Index.load(tmp_path)
    Index.build(DOCS).save(tmp_path)
    np.save(tmp_path / "embeddings.npy", np.zeros((99, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="corrupt"):
        Index.load(tmp_path)
