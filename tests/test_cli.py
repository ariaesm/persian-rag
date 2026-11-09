import io
import json
import urllib.request
from pathlib import Path

import pytest

from persian_rag.cli import main


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "tea.txt").write_text("چاي در لاهيجان کشت مي شود.", encoding="utf-8")
    (d / "rice.md").write_text("# برنج\n\nبرنج در **گیلان** کاشته می‌شود.", encoding="utf-8")
    return d


def test_cli_normalize(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["normalize", "كتاب هاي  من", "--digits", "fa"]) == 0
    assert capsys.readouterr().out.strip() == "کتاب‌های من"


def test_cli_ingest_and_search(
    docs_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    idx = tmp_path / "idx"
    assert main(["ingest", str(docs_dir), "--index", str(idx), "--dim", "128"]) == 0
    assert "2 documents" in capsys.readouterr().out
    assert main(["search", "چای لاهیجان", "--index", str(idx), "-k", "1", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["doc_id"] == "tea.txt" and rows[0]["rank"] == 1
    assert main(["search", "برنج", "--index", str(idx), "--mode", "bm25"]) == 0
    assert "rice.md#0" in capsys.readouterr().out


def test_cli_ask_with_mocked_llm(
    docs_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    idx = tmp_path / "idx"
    main(["ingest", str(docs_dir), "--index", str(idx)])
    capsys.readouterr()
    reply = {"choices": [{"message": {"content": "چای در لاهیجان کشت می‌شود [1]."}}]}
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout: FakeResponse(json.dumps(reply, ensure_ascii=False).encode()),
    )
    monkeypatch.setenv("PERSIAN_RAG_BASE_URL", "http://localhost:11434/v1")
    assert main(["ask", "چای کجا کشت می‌شود؟", "--index", str(idx), "--model", "qwen2.5"]) == 0
    out = capsys.readouterr().out
    assert "[1]" in out and "*[1] tea.txt#0" in out


def test_cli_missing_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["search", "x", "--index", str(tmp_path / "none")]) == 1
    assert "no index" in capsys.readouterr().err


def test_cli_ingest_empty_dir(tmp_path: Path) -> None:
    assert main(["ingest", str(tmp_path), "--index", str(tmp_path / "i")]) == 1
