import io
import json
import urllib.request
from collections.abc import Sequence

import pytest

from persian_rag.http_client import APIError
from persian_rag.llm import OLLAMA_BASE_URL, ChatClient, LLMSettings
from persian_rag.pipeline import RAGPipeline, extract_citations
from persian_rag.prompts import IDK, SYSTEM_PROMPT, build_messages
from persian_rag.retrieval import BM25Retriever
from persian_rag.schema import Chunk, SearchResult


class FakeResponse(io.BytesIO):
    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[Sequence[dict[str, str]]] = []

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


def test_settings_from_env_and_overrides() -> None:
    env = {"PERSIAN_RAG_BASE_URL": OLLAMA_BASE_URL, "PERSIAN_RAG_MODEL": "qwen2.5:7b"}
    s = LLMSettings.from_env(env)
    assert (s.base_url, s.model, s.api_key) == (OLLAMA_BASE_URL, "qwen2.5:7b", None)
    s2 = LLMSettings.from_env(env, model="llama3.1", api_key=None)
    assert s2.model == "llama3.1"
    assert LLMSettings.from_env({}).base_url == "https://api.openai.com/v1"


def test_chat_client_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float) -> FakeResponse:
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data.decode("utf-8"))  # type: ignore[union-attr]
        reply = {"choices": [{"message": {"role": "assistant", "content": " سلام [1] "}}]}
        return FakeResponse(json.dumps(reply, ensure_ascii=False).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = ChatClient(LLMSettings(base_url=OLLAMA_BASE_URL, model="qwen2.5", api_key=None))
    out = client.complete([{"role": "user", "content": "سلام"}])
    assert out == "سلام [1]"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["auth"] is None  # no key -> no Authorization header (Ollama)
    body = seen["body"]
    assert isinstance(body, dict) and body["model"] == "qwen2.5" and body["stream"] is False


def test_chat_client_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout: FakeResponse(b'{"choices": []}')
    )
    with pytest.raises(APIError, match="malformed"):
        ChatClient(LLMSettings(api_key="k")).complete([{"role": "user", "content": "x"}])


def test_system_prompt_requires_citations_and_idk() -> None:
    assert IDK in SYSTEM_PROMPT and "[1]" in SYSTEM_PROMPT


def test_build_messages_numbers_sources() -> None:
    hits = [
        SearchResult(Chunk("a#0", "a.md", 0, "متن اول"), 1.0),
        SearchResult(Chunk("b#0", "b.md", 0, "متن دوم"), 0.5),
    ]
    msgs = build_messages("پرسش؟", hits)
    assert msgs[0]["role"] == "system"
    assert "[1] (منبع: a.md)\nمتن اول" in msgs[1]["content"]
    assert "[2] (منبع: b.md)" in msgs[1]["content"]
    assert "پرسش: پرسش؟" in msgs[1]["content"]


def test_extract_citations_dedupes_and_validates() -> None:
    assert extract_citations("الف [2] ب [1][2] ج [۳] د [9]", num_sources=3) == [2, 1, 3]


def test_pipeline_answers_with_citations(chunks: list[Chunk]) -> None:
    llm = FakeLLM("زعفران در پاییز برداشت می‌شود [1].")
    answer = RAGPipeline(BM25Retriever(chunks), llm, k=2).ask("زعفران کی برداشت می‌شود؟")
    assert answer.citations == [1]
    assert answer.cited_sources[0].chunk.doc_id == "saffron"
    assert not answer.is_unknown
    assert "زعفران در خراسان" in llm.calls[0][1]["content"]


def test_pipeline_skips_llm_when_nothing_retrieved(chunks: list[Chunk]) -> None:
    llm = FakeLLM("should not be called")
    answer = RAGPipeline(BM25Retriever(chunks), llm).ask("فوتبال")
    assert answer.text == IDK and answer.is_unknown and llm.calls == []


def test_pipeline_detects_model_idk(chunks: list[Chunk]) -> None:
    answer = RAGPipeline(BM25Retriever(chunks), FakeLLM("نمي دانم")).ask("حافظ کجا متولد شد؟")
    assert answer.is_unknown and answer.citations == []
