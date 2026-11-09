"""Core data structures shared across the library."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Document:
    """A source document.

    Attributes:
        id: Stable identifier, usually the path relative to the ingested directory.
        text: Full text content.
        metadata: Free-form metadata (``source`` path, ``page`` count, ...).

    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage cut from a :class:`Document`.

    Attributes:
        id: Unique id of the form ``"<doc_id>#<index>"``.
        doc_id: Id of the parent document.
        index: Position of the chunk within its document (0-based).
        text: Passage text.
        metadata: Metadata inherited from the parent document.

    """

    id: str
    doc_id: str
    index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Deserialize from :meth:`to_dict` output."""
        return cls(
            id=data["id"],
            doc_id=data["doc_id"],
            index=int(data["index"]),
            text=data["text"],
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A scored retrieval hit."""

    chunk: Chunk
    score: float
