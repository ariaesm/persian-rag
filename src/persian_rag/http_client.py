"""Minimal JSON-over-HTTP helper built on :mod:`urllib` (no third-party deps)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class APIError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint returns an error or bad payload."""

    def __init__(self, message: str, status: int | None = None) -> None:
        """Store the HTTP ``status`` (if any) alongside the message."""
        super().__init__(message)
        self.status = status


def join_url(base_url: str, path: str) -> str:
    """Join an API base URL (e.g. ``.../v1``) and an endpoint path."""
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def post_json(
    url: str, payload: dict[str, Any], *, api_key: str | None = None, timeout: float = 60.0
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON response.

    Args:
        url: Full endpoint URL.
        payload: JSON-serializable request body.
        api_key: Bearer token; omitted from the request when empty (e.g. for Ollama).
        timeout: Socket timeout in seconds.

    Raises:
        APIError: On HTTP errors, connection failures or non-JSON responses.

    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise APIError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise APIError(f"cannot reach {url}: {exc.reason}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(f"invalid JSON from {url}") from exc
    if not isinstance(data, dict):
        raise APIError(f"unexpected response from {url}: {type(data).__name__}")
    return data
