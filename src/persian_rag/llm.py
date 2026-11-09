"""OpenAI-compatible chat completions client (OpenAI, OpenRouter, Ollama, vLLM, ...)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from persian_rag.http_client import APIError, join_url, post_json

ENV_BASE_URL = "PERSIAN_RAG_BASE_URL"
ENV_API_KEY = "PERSIAN_RAG_API_KEY"
ENV_MODEL = "PERSIAN_RAG_MODEL"

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

Message = Mapping[str, str]


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Connection settings for an OpenAI-compatible endpoint."""

    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = None
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: str | None) -> LLMSettings:
        """Read ``PERSIAN_RAG_BASE_URL``, ``PERSIAN_RAG_API_KEY`` and ``PERSIAN_RAG_MODEL``.

        Explicit, non-``None`` keyword overrides (e.g. from CLI flags) win over the
        environment.
        """
        env = os.environ if env is None else env
        values: dict[str, Any] = {
            "base_url": env.get(ENV_BASE_URL) or DEFAULT_BASE_URL,
            "api_key": env.get(ENV_API_KEY) or None,
            "model": env.get(ENV_MODEL) or DEFAULT_MODEL,
        }
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)


class ChatClient:
    """Calls ``POST {base_url}/chat/completions`` using only the standard library."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        temperature: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        """Create a client; settings default to :meth:`LLMSettings.from_env`."""
        self.settings = settings or LLMSettings.from_env()
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: Sequence[Message], **params: Any) -> str:
        """Send ``messages`` and return the assistant's reply text.

        Raises:
            APIError: On transport errors or a malformed response.

        """
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [dict(m) for m in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        payload.update(params)
        url = join_url(self.settings.base_url, "chat/completions")
        data = post_json(url, payload, api_key=self.settings.api_key, timeout=self.timeout)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIError(f"malformed chat response from {url}: {str(data)[:200]}") from exc
        return str(content or "").strip()
