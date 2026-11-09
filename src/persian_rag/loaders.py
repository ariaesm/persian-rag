"""Load documents from ``.txt``, ``.md`` and (optionally) ``.pdf`` files."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from persian_rag.schema import Document

log = logging.getLogger(__name__)

TEXT_SUFFIXES = frozenset({".txt", ".text"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
PDF_SUFFIXES = frozenset({".pdf"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | MARKDOWN_SUFFIXES | PDF_SUFFIXES

# Encodings tried in order. Windows-1256 is still common for older Persian files.
_ENCODINGS = ("utf-8-sig", "cp1256")

_MD_FENCE = re.compile(r"^\s*(```|~~~).*$", re.MULTILINE)
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)(?=\S)(.+?)(?<=\S)\1")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+", re.MULTILINE)
_MD_QUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_MD_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_MD_HTML = re.compile(r"<[^>\n]+>")


class MissingDependencyError(ImportError):
    """Raised when an optional dependency (e.g. ``pypdf``) is not installed."""


def read_text_file(path: Path) -> str:
    """Read a text file as UTF-8, falling back to Windows-1256.

    Raises:
        UnicodeDecodeError: If the file is in none of the supported encodings.

    """
    data = path.read_bytes()
    for encoding in _ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8")  # re-raise with a meaningful error


def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax while keeping the readable text."""
    text = _MD_FENCE.sub("", text)
    text = _MD_RULE.sub("", text)
    text = _MD_HEADING.sub("", text)
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_LIST.sub("", text)
    text = _MD_QUOTE.sub("", text)
    return _MD_HTML.sub("", text)


def read_pdf(path: Path) -> str:
    """Extract text from a PDF using the optional ``pypdf`` dependency.

    Raises:
        MissingDependencyError: If ``pypdf`` is not installed.

    """
    try:
        from pypdf import PdfReader  # optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError(
            "Reading PDFs requires pypdf: pip install 'persian-rag[pdf]'"
        ) from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: str | Path, *, doc_id: str | None = None) -> Document:
    """Load a single file into a :class:`Document`.

    Args:
        path: File to load; the loader is chosen from its suffix.
        doc_id: Optional id; defaults to the file name.

    Raises:
        ValueError: For unsupported file types.

    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = read_text_file(p)
    elif suffix in MARKDOWN_SUFFIXES:
        text = strip_markdown(read_text_file(p))
    elif suffix in PDF_SUFFIXES:
        text = read_pdf(p)
    else:
        raise ValueError(f"unsupported file type: {p.suffix or '(none)'} ({p})")
    return Document(
        id=doc_id or p.name,
        text=text,
        metadata={"source": str(p), "format": suffix.lstrip(".")},
    )


def iter_files(root: Path, *, recursive: bool = True) -> Iterator[Path]:
    """Yield supported files under ``root`` in a stable (sorted) order."""
    pattern = "**/*" if recursive else "*"
    for p in sorted(root.glob(pattern)):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES and not p.name.startswith("."):
            yield p


def load_documents(
    paths: str | Path | Iterable[str | Path], *, recursive: bool = True
) -> list[Document]:
    """Load documents from files and/or directories.

    Directory entries use their path relative to the directory as the document id.
    PDFs are skipped with a warning when ``pypdf`` is not installed.
    """
    if isinstance(paths, str | Path):
        paths = [paths]
    docs: list[Document] = []
    for raw in paths:
        root = Path(raw)
        if root.is_dir():
            candidates = [
                (p, p.relative_to(root).as_posix()) for p in iter_files(root, recursive=recursive)
            ]
        elif root.is_file():
            candidates = [(root, root.name)]
        else:
            raise FileNotFoundError(root)
        for path, doc_id in candidates:
            try:
                doc = load_file(path, doc_id=doc_id)
            except MissingDependencyError as exc:
                log.warning("skipping %s: %s", path, exc)
                continue
            if doc.text.strip():
                docs.append(doc)
    return docs
