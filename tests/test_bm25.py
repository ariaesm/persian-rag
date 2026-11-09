import math

import pytest

from persian_rag.retrieval import BM25Retriever, Retriever
from persian_rag.schema import Chunk
from persian_rag.tokenizer import Analyzer
from tests.conftest import make_chunks


def test_bm25_finds_right_document_despite_arabic_letters(chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(chunks)
    # The query uses Persian ی/ک; the indexed text uses Arabic ي/ك.
    top = bm25.search("شاعر شیرازی که در قرن هشتم زندگی می‌کرد", k=1)
    assert top[0].chunk.doc_id == "hafez"


def test_bm25_handles_spaced_suffixes(chunks: list[Chunk]) -> None:
    top = BM25Retriever(chunks).search("گل‌های زعفران کی برداشت می‌شوند؟", k=1)
    assert top[0].chunk.doc_id == "saffron"


def test_bm25_digit_systems_match(chunks: list[Chunk]) -> None:
    top = BM25Retriever(chunks).search("1991", k=1)
    assert top[0].chunk.doc_id == "python"


def test_raw_analyzer_misses_what_normalization_finds(chunks: list[Chunk]) -> None:
    raw = BM25Retriever(chunks, analyzer=Analyzer.raw())
    assert raw.search("شیراز", k=1) == []  # "شيراز" in the doc uses Arabic yeh
    assert BM25Retriever(chunks).search("شیراز", k=1)[0].chunk.doc_id == "hafez"


def test_bm25_no_match_and_stopword_only_query(chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(chunks)
    assert bm25.search("فوتبال", k=3) == []
    assert bm25.search("و در به از", k=3) == []


def test_bm25_score_matches_formula() -> None:
    docs = make_chunks({"a": "سیب سیب موز", "b": "موز", "c": "انار"})
    bm25 = BM25Retriever(docs, k1=1.2, b=0.75)
    n, df, tf, dl, avgdl = 3, 1, 2, 3, (3 + 1 + 1) / 3
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * dl / avgdl))
    assert bm25.scores("سیب")[0] == pytest.approx(expected)


def test_bm25_rare_terms_weigh_more() -> None:
    docs = make_chunks({"a": "کتاب تاریخ", "b": "کتاب ریاضی", "c": "کتاب ادبیات"})
    top = BM25Retriever(docs).search("کتاب ریاضی", k=3)
    assert top[0].chunk.doc_id == "b"


def test_bm25_k_and_protocol(chunks: list[Chunk]) -> None:
    bm25 = BM25Retriever(chunks)
    assert isinstance(bm25, Retriever)
    assert len(bm25.search("در", k=10)) <= 10
    with pytest.raises(ValueError):
        BM25Retriever(chunks, b=2.0)


def test_bm25_empty_index() -> None:
    assert BM25Retriever([]).search("هر چیزی") == []
