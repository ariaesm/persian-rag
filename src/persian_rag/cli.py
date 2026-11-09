"""Command-line interface: ``persian-rag {ingest,search,ask,normalize,eval}``."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections.abc import Sequence

from persian_rag import __version__
from persian_rag.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from persian_rag.evaluation import format_table, load_qa, run_benchmark
from persian_rag.http_client import APIError
from persian_rag.index import Index
from persian_rag.llm import ENV_API_KEY, ChatClient, LLMSettings
from persian_rag.loaders import load_documents
from persian_rag.normalizer import Normalizer
from persian_rag.pipeline import RAGPipeline

DEFAULT_INDEX = "./index"
MODES = ("bm25", "dense", "hybrid")


def _read_arg_or_stdin(value: str | None) -> str:
    if value is None or value == "-":
        return sys.stdin.read()
    return value


def _snippet(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def cmd_normalize(args: argparse.Namespace) -> int:
    """Print the normalized form of the given text (or stdin)."""
    print(Normalizer(digits=args.digits).normalize(_read_arg_or_stdin(args.text)))
    return 0


def _make_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "hashing":
        return HashingEmbedder(dim=args.dim)
    settings = LLMSettings.from_env(base_url=args.base_url)
    return OpenAIEmbedder(
        args.embedding_model, base_url=settings.base_url, api_key=settings.api_key
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    """Load, chunk, embed and save documents to an index directory."""
    docs = load_documents(args.paths)
    if not docs:
        print("error: no supported documents found (.txt, .md, .pdf)", file=sys.stderr)
        return 1
    index = Index.build(
        docs,
        embedder=_make_embedder(args),
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        normalize=not args.no_normalize,
    )
    path = index.save(args.index)
    print(f"indexed {len(docs)} documents → {len(index.chunks)} chunks in {path}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Print the top-k chunks for a query."""
    index = Index.load(args.index, api_key=os.environ.get(ENV_API_KEY))
    results = index.retriever(args.mode).search(args.query, args.k)
    if args.json:
        rows = [
            {
                "rank": i,
                "score": r.score,
                "id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "text": r.chunk.text,
            }
            for i, r in enumerate(results, start=1)
        ]
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not results:
        print("no results")
    for i, r in enumerate(results, start=1):
        print(f"{i}. [{r.score:.4f}] {r.chunk.id}\n   {_snippet(r.chunk.text)}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a question with retrieved, cited context."""
    settings = LLMSettings.from_env(base_url=args.base_url, model=args.model)
    index = Index.load(args.index, api_key=settings.api_key)
    pipeline = RAGPipeline(index.retriever(args.mode), ChatClient(settings), k=args.k)
    try:
        answer = pipeline.ask(args.question)
    except APIError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(answer.text)
    if answer.sources:
        print("\nمنابع:")
        for i, hit in enumerate(answer.sources, start=1):
            mark = "*" if i in answer.citations else " "
            print(f" {mark}[{i}] {hit.chunk.id}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Compare retrievers with and without normalization on a QA set."""
    docs = load_documents(args.corpus)
    qa = load_qa(args.qa)
    rows = run_benchmark(docs, qa, ks=args.k, chunk_size=args.chunk_size, overlap=args.overlap)
    if args.json:
        payload = [
            {
                "retriever": r.retriever,
                "normalized": r.normalized,
                "recall": {str(k): v for k, v in r.recall.items()},
                "mrr": r.mrr,
            }
            for r in rows
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"{len(docs)} documents, {len(qa)} questions\n")
        print(format_table(rows))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="persian-rag", description="Retrieval-Augmented Generation for Persian text."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("normalize", help="normalize Persian text")
    p.add_argument("text", nargs="?", help="text to normalize (default: stdin)")
    p.add_argument("--digits", choices=("fa", "en", "keep"), default="en")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("ingest", help="build an index from files or directories")
    p.add_argument("paths", nargs="+", help="files or directories (.txt, .md, .pdf)")
    p.add_argument("--index", default=DEFAULT_INDEX, help="index directory (default: ./index)")
    p.add_argument("--chunk-size", type=int, default=120, help="max tokens per chunk")
    p.add_argument("--overlap", type=int, default=30, help="token overlap between chunks")
    p.add_argument("--embedder", choices=("hashing", "openai"), default="hashing")
    p.add_argument("--embedding-model", default="text-embedding-3-small")
    p.add_argument("--dim", type=int, default=1024, help="hashing embedder dimension")
    p.add_argument("--base-url", default=None, help="override PERSIAN_RAG_BASE_URL")
    p.add_argument("--no-normalize", action="store_true", help="disable Persian normalization")
    p.set_defaults(func=cmd_ingest)

    for name, helptext in (("search", "retrieve chunks"), ("ask", "answer a question with an LLM")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("question" if name == "ask" else "query")
        p.add_argument("--index", default=DEFAULT_INDEX)
        p.add_argument("-k", type=int, default=4 if name == "ask" else 5)
        p.add_argument("--mode", choices=MODES, default="hybrid")
        if name == "search":
            p.add_argument("--json", action="store_true", help="print JSON")
            p.set_defaults(func=cmd_search)
        else:
            p.add_argument("--model", default=None, help="override PERSIAN_RAG_MODEL")
            p.add_argument("--base-url", default=None, help="override PERSIAN_RAG_BASE_URL")
            p.set_defaults(func=cmd_ask)
    p = sub.add_parser("eval", help="evaluate retrieval (Recall@k, MRR)")
    p.add_argument("--corpus", default="examples/corpus", help="documents directory")
    p.add_argument("--qa", default="examples/qa.jsonl", help="JSONL question set")
    p.add_argument("-k", type=int, nargs="+", default=[1, 3, 5], help="cut-offs for Recall@k")
    p.add_argument("--chunk-size", type=int, default=120)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--json", action="store_true", help="print JSON")
    p.set_defaults(func=cmd_eval)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``persian-rag`` console script."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
