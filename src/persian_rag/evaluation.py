"""Retrieval evaluation: Recall@k and MRR at the document level."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from persian_rag.embeddings import Embedder, HashingEmbedder
from persian_rag.index import Index, Mode
from persian_rag.retrieval import Retriever
from persian_rag.schema import Document

DEFAULT_KS: tuple[int, ...] = (1, 3, 5)
MODES: tuple[Mode, ...] = ("bm25", "dense", "hybrid")
MODE_LABELS: dict[str, str] = {
    "bm25": "BM25",
    "dense": "Hashing embeddings",
    "hybrid": "Hybrid (RRF)",
}


@dataclass(frozen=True, slots=True)
class QAItem:
    """A question and the ids of the documents that answer it."""

    question: str
    relevant: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvalRow:
    """Metrics for one retriever configuration."""

    retriever: str
    normalized: bool
    recall: dict[int, float]
    mrr: float


def load_qa(path: str | Path) -> list[QAItem]:
    """Load a JSONL file of ``{"question": ..., "relevant": [doc_id, ...]}`` lines."""
    items: list[QAItem] = []
    for lineno, line in enumerate(Path(path).read_text("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        relevant = row.get("relevant")
        if not row.get("question") or not relevant:
            raise ValueError(f"{path}:{lineno}: need 'question' and non-empty 'relevant'")
        items.append(QAItem(row["question"], frozenset(relevant)))
    return items


def rank_documents(retriever: Retriever, query: str, depth: int = 50) -> list[str]:
    """Return distinct document ids in the order their best chunk was retrieved."""
    seen: dict[str, None] = {}
    for hit in retriever.search(query, depth):
        seen.setdefault(hit.chunk.doc_id, None)
    return list(seen)


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of relevant documents present in the top ``k``."""
    rel = set(relevant)
    return len(rel.intersection(ranked[:k])) / len(rel) if rel else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """``1 / rank`` of the first relevant document (0 if none is retrieved)."""
    rel = set(relevant)
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0


def evaluate(
    retriever: Retriever, qa: Sequence[QAItem], ks: Sequence[int] = DEFAULT_KS
) -> tuple[dict[int, float], float]:
    """Return mean Recall@k for each ``k`` and the mean reciprocal rank."""
    if not qa:
        raise ValueError("empty QA set")
    recall = dict.fromkeys(ks, 0.0)
    mrr = 0.0
    for item in qa:
        ranked = rank_documents(retriever, item.question)
        for k in ks:
            recall[k] += recall_at_k(ranked, item.relevant, k)
        mrr += reciprocal_rank(ranked, item.relevant)
    n = len(qa)
    return {k: v / n for k, v in recall.items()}, mrr / n


def run_benchmark(
    documents: Sequence[Document],
    qa: Sequence[QAItem],
    *,
    ks: Sequence[int] = DEFAULT_KS,
    embedder: Embedder | None = None,
    chunk_size: int = 120,
    overlap: int = 30,
) -> list[EvalRow]:
    """Evaluate BM25, dense and hybrid retrieval, without and with normalization."""
    embedder = embedder or HashingEmbedder()
    rows: list[EvalRow] = []
    for normalized in (False, True):
        index = Index.build(
            documents,
            embedder=embedder,
            chunk_size=chunk_size,
            overlap=overlap,
            normalize=normalized,
        )
        for mode in MODES:
            recall, mrr = evaluate(index.retriever(mode), qa, ks)
            rows.append(EvalRow(MODE_LABELS[mode], normalized, recall, mrr))
    return rows


def format_table(rows: Sequence[EvalRow]) -> str:
    """Render rows as a GitHub-flavoured Markdown table."""
    ks = sorted(rows[0].recall) if rows else []
    header = ["Retriever", "Normalization", *[f"Recall@{k}" for k in ks], "MRR"]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * 2 + ["---:"] * (len(ks) + 1)) + "|",
    ]
    for r in rows:
        cells = [r.retriever, "on" if r.normalized else "off"]
        cells += [f"{r.recall[k]:.3f}" for k in ks] + [f"{r.mrr:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
