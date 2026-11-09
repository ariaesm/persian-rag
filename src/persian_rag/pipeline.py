"""End-to-end RAG: retrieve passages, ask the LLM, return a cited answer."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from persian_rag.normalizer import normalize
from persian_rag.prompts import IDK, build_messages
from persian_rag.retrieval import Retriever
from persian_rag.schema import SearchResult

# Accept both Latin and Persian digits inside brackets: [1], [۲].
_CITATION = re.compile(r"\[([0-9۰-۹]+)\]")


class ChatModel(Protocol):
    """Anything with a ``complete(messages) -> str`` method (see :class:`~.llm.ChatClient`)."""

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the model's reply."""
        ...


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer with the sources it was grounded on."""

    text: str
    sources: list[SearchResult] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)

    @property
    def is_unknown(self) -> bool:
        """True when the model declined to answer (``نمی‌دانم``)."""
        return normalize(self.text).startswith(normalize(IDK))

    @property
    def cited_sources(self) -> list[SearchResult]:
        """Sources actually referenced by ``[n]`` citations in the answer."""
        return [self.sources[i - 1] for i in self.citations]


def extract_citations(text: str, num_sources: int) -> list[int]:
    """Return distinct, valid 1-based citation numbers in order of appearance."""
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        n = int(normalize(match.group(1), digits="en"))
        if 1 <= n <= num_sources and n not in seen:
            seen.append(n)
    return seen


class RAGPipeline:
    """Glue between a :class:`Retriever` and a chat model."""

    def __init__(self, retriever: Retriever, llm: ChatModel, *, k: int = 4) -> None:
        """Create a pipeline that passes the top ``k`` chunks to the model."""
        self.retriever = retriever
        self.llm = llm
        self.k = k

    def ask(self, question: str) -> Answer:
        """Answer ``question``; returns ``نمی‌دانم`` without calling the LLM if nothing is found."""
        results = self.retriever.search(question, self.k)
        if not results:
            return Answer(text=IDK)
        reply = self.llm.complete(build_messages(question, results))
        return Answer(text=reply, sources=results, citations=extract_citations(reply, len(results)))
