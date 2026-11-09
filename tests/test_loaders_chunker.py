import itertools
from pathlib import Path

import pytest

from persian_rag.chunker import chunk_documents, chunk_text, split_sentences
from persian_rag.loaders import load_documents, load_file, strip_markdown
from persian_rag.schema import Document
from persian_rag.tokenizer import count_tokens


def test_load_txt_utf8_with_bom(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_bytes("﻿سلام دنیا".encode())
    doc = load_file(f)
    assert doc.text == "سلام دنیا"
    assert doc.id == "a.txt"
    assert doc.metadata["format"] == "txt"


def test_load_txt_windows_1256_fallback(tmp_path: Path) -> None:
    f = tmp_path / "old.txt"
    # cp1256 has no Persian yeh, which is why legacy files are full of Arabic ي.
    f.write_bytes("کتاب فارسي".encode("cp1256"))
    assert load_file(f).text == "کتاب فارسي"


def test_markdown_is_stripped(tmp_path: Path) -> None:
    f = tmp_path / "n.md"
    f.write_text("# عنوان\n\n- مورد **مهم** با [پیوند](https://x.ir)\n", encoding="utf-8")
    text = load_file(f).text
    assert "#" not in text and "**" not in text and "https" not in text
    assert "عنوان" in text and "مورد مهم با پیوند" in text


def test_strip_markdown_keeps_code_content() -> None:
    assert "print(1)" in strip_markdown("```python\nprint(1)\n```")


def test_unsupported_type_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.docx"
    f.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        load_file(f)


def test_load_directory_recursive_sorted_and_skips_empty(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "b.txt").write_text("دوم", encoding="utf-8")
    (tmp_path / "a.md").write_text("اول", encoding="utf-8")
    (tmp_path / "sub" / "c.txt").write_text("سوم", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("  ", encoding="utf-8")
    (tmp_path / "ignore.json").write_text("{}", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.id for d in docs] == ["a.md", "b.txt", "sub/c.txt"]


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


def test_split_sentences_persian_terminators() -> None:
    text = "تهران پایتخت است. آیا مترو دارد؟ بله!\nخط جدید"
    assert split_sentences(text) == ["تهران پایتخت است.", "آیا مترو دارد؟", "بله!", "خط جدید"]


def test_split_sentences_keeps_decimals() -> None:
    assert split_sentences("نسخه 3.11 منتشر شد. عالی است.") == [
        "نسخه 3.11 منتشر شد.",
        "عالی است.",
    ]


def test_chunk_respects_size_and_overlap() -> None:
    sentences = [f"جمله شماره {i} درباره موضوع است." for i in range(20)]  # 6 tokens each
    chunks = chunk_text(" ".join(sentences), chunk_size=20, overlap=6)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 20 for c in chunks)
    # Each chunk after the first starts with the last sentence of the previous one.
    for prev, nxt in itertools.pairwise(chunks):
        assert nxt.startswith(split_sentences(prev)[-1])


def test_chunk_splits_overlong_sentence() -> None:
    text = " ".join(["کلمه"] * 50)
    chunks = chunk_text(text, chunk_size=20, overlap=0)
    assert [count_tokens(c) for c in chunks] == [20, 20, 10]


def test_chunk_invalid_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=0)


def test_chunk_documents_ids_and_metadata() -> None:
    doc = Document(id="d.txt", text="یک. دو. سه.", metadata={"source": "d.txt"})
    chunks = chunk_documents([doc], chunk_size=2, overlap=0)
    assert [c.id for c in chunks] == ["d.txt#0", "d.txt#1"]
    assert all(c.doc_id == "d.txt" and c.metadata["source"] == "d.txt" for c in chunks)
