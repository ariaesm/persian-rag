import json
from pathlib import Path

import pytest

from persian_rag.cli import main
from persian_rag.evaluation import (
    QAItem,
    evaluate,
    format_table,
    load_qa,
    rank_documents,
    recall_at_k,
    reciprocal_rank,
    run_benchmark,
)
from persian_rag.loaders import load_documents
from persian_rag.retrieval import BM25Retriever
from persian_rag.schema import Chunk

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "examples" / "corpus"
QA = ROOT / "examples" / "qa.jsonl"


def test_metric_functions() -> None:
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, {"b"}, 1) == 0.0
    assert recall_at_k(ranked, {"b", "c"}, 3) == 1.0
    assert recall_at_k(ranked, {"b", "z"}, 3) == 0.5
    assert reciprocal_rank(ranked, {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"z"}) == 0.0


def test_rank_documents_dedupes_chunks() -> None:
    chunks = [
        Chunk("a#0", "a", 0, "زعفران خراسان"),
        Chunk("a#1", "a", 1, "زعفران پاییز"),
        Chunk("b#0", "b", 0, "زعفران"),
    ]
    assert sorted(rank_documents(BM25Retriever(chunks), "زعفران")) == ["a", "b"]


def test_evaluate_on_toy_set(chunks: list[Chunk]) -> None:
    qa = [QAItem("مترو تهران", frozenset({"metro"})), QAItem("فوتبال", frozenset({"hafez"}))]
    recall, mrr = evaluate(BM25Retriever(chunks), qa, ks=(1,))
    assert recall == {1: 0.5} and mrr == 0.5


def test_load_qa_validates(tmp_path: Path) -> None:
    f = tmp_path / "qa.jsonl"
    f.write_text('{"question": "x", "relevant": []}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_qa(f)


def test_sample_qa_references_existing_documents() -> None:
    doc_ids = {d.id for d in load_documents(CORPUS)}
    qa = load_qa(QA)
    assert len(qa) >= 20 and 6 <= len(doc_ids) <= 10
    for item in qa:
        assert item.relevant <= doc_ids


def test_benchmark_normalization_never_hurts_on_sample_corpus() -> None:
    rows = run_benchmark(load_documents(CORPUS), load_qa(QA))
    assert len(rows) == 6
    by_key = {(r.retriever, r.normalized): r for r in rows}
    for name in ("BM25", "Hashing embeddings", "Hybrid (RRF)"):
        assert by_key[(name, True)].mrr >= by_key[(name, False)].mrr
    table = format_table(rows)
    assert table.startswith("| Retriever | Normalization | Recall@1")


def test_cli_eval_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["eval", "--corpus", str(CORPUS), "--qa", str(QA), "-k", "1", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["retriever"] for r in rows} == {"BM25", "Hashing embeddings", "Hybrid (RRF)"}
    assert all(0.0 <= r["mrr"] <= 1.0 for r in rows)
