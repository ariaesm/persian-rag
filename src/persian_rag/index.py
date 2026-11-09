"""A simple persistent index: chunks as JSON, embeddings as a numpy ``.npy`` file."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from persian_rag import __version__
from persian_rag.chunker import chunk_documents
from persian_rag.embeddings import Embedder, HashingEmbedder, embedder_from_spec
from persian_rag.normalizer import Normalizer
from persian_rag.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from persian_rag.schema import Chunk, Document
from persian_rag.tokenizer import Analyzer

Mode = Literal["bm25", "dense", "hybrid"]

CHUNKS_FILE = "chunks.json"
EMBEDDINGS_FILE = "embeddings.npy"
META_FILE = "meta.json"
FORMAT_VERSION = 1


def _identity(text: str) -> str:
    return text


@dataclass
class Index:
    """Chunks plus their embeddings and the settings needed to query them.

    The BM25 inverted index is cheap to rebuild, so only chunks and dense vectors are
    stored on disk.
    """

    chunks: list[Chunk]
    embedder: Embedder
    embeddings: np.ndarray
    normalize: bool = True
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        documents: Sequence[Document],
        *,
        embedder: Embedder | None = None,
        chunk_size: int = 120,
        overlap: int = 30,
        normalize: bool = True,
    ) -> Index:
        """Chunk and embed ``documents``.

        Args:
            documents: Loaded documents.
            embedder: Embedder for dense retrieval (default: :class:`HashingEmbedder`).
            chunk_size: Max tokens per chunk.
            overlap: Token overlap between consecutive chunks.
            normalize: Run Persian normalization/stemming in retrieval.

        """
        embedder = embedder or HashingEmbedder()
        chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
        preprocess = Normalizer() if normalize else _identity
        texts = [preprocess(c.text) for c in chunks]
        matrix = embedder.embed(texts) if texts else np.zeros((0, 0), dtype=np.float32)
        return cls(
            chunks=chunks,
            embedder=embedder,
            embeddings=matrix,
            normalize=normalize,
            settings={"chunk_size": chunk_size, "overlap": overlap},
        )

    @property
    def preprocess(self) -> Callable[[str], str]:
        """Text preprocessing applied before embedding."""
        return Normalizer() if self.normalize else _identity

    @property
    def analyzer(self) -> Analyzer:
        """Analyzer used by BM25."""
        return Analyzer() if self.normalize else Analyzer.raw()

    def retriever(self, mode: Mode = "hybrid") -> Retriever:
        """Return a retriever over this index (``"bm25"``, ``"dense"`` or ``"hybrid"``)."""
        if mode == "bm25":
            return BM25Retriever(self.chunks, self.analyzer)
        dense = DenseRetriever(
            self.chunks, self.embedder, preprocess=self.preprocess, matrix=self.embeddings
        )
        if mode == "dense":
            return dense
        if mode == "hybrid":
            return HybridRetriever([BM25Retriever(self.chunks, self.analyzer), dense])
        raise ValueError(f"unknown retrieval mode: {mode!r}")

    def save(self, directory: str | Path) -> Path:
        """Write the index to ``directory`` (created if needed) and return its path."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "format_version": FORMAT_VERSION,
            "persian_rag_version": __version__,
            "embedder": self.embedder.spec(),
            "normalize": self.normalize,
            "settings": self.settings,
            "num_chunks": len(self.chunks),
        }
        (path / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        (path / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), "utf-8"
        )
        np.save(path / EMBEDDINGS_FILE, self.embeddings, allow_pickle=False)
        return path

    @classmethod
    def load(cls, directory: str | Path, *, api_key: str | None = None) -> Index:
        """Load an index written by :meth:`save`.

        Args:
            directory: Index directory.
            api_key: Credential for HTTP embedders (never stored on disk).

        Raises:
            FileNotFoundError: If the directory is not an index.
            ValueError: If the files are inconsistent or from an unknown format.

        """
        path = Path(directory)
        if not (path / META_FILE).exists():
            raise FileNotFoundError(f"no index found at {path} (missing {META_FILE})")
        meta = json.loads((path / META_FILE).read_text("utf-8"))
        if meta.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported index format: {meta.get('format_version')}")
        chunks = [Chunk.from_dict(d) for d in json.loads((path / CHUNKS_FILE).read_text("utf-8"))]
        embeddings = np.load(path / EMBEDDINGS_FILE, allow_pickle=False)
        if embeddings.shape[0] != len(chunks):
            raise ValueError("index is corrupt: embeddings and chunks differ in length")
        return cls(
            chunks=chunks,
            embedder=embedder_from_spec(meta["embedder"], api_key=api_key),
            embeddings=embeddings,
            normalize=bool(meta.get("normalize", True)),
            settings=dict(meta.get("settings") or {}),
        )
