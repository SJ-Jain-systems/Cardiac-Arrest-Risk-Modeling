"""Command-line interface for the RAG pipeline.

Usage:
    python -m src.rag.cli build
    python -m src.rag.cli query "What is the sensitivity at threshold 0.45?"
    python -m src.rag.cli query "..." --retrieve-only --top-k 8

``build`` (re)builds and persists the index from ``reports/``. ``query``
retrieves and generates a grounded answer; ``--retrieve-only`` prints the top-k
chunks with scores and sources and skips the LLM (no API key required).
"""

from __future__ import annotations

import argparse
import sys

from src.rag.config import TOP_K
from src.rag.generator import MissingAPIKeyError
from src.rag.pipeline import RAGPipeline


def _cmd_build() -> int:
    """Build and persist the index, printing a chunk count."""
    pipeline = RAGPipeline()
    store = pipeline.build_index()
    print(f"Built index with {len(store.chunks)} chunks.")
    return 0


def _cmd_query(question: str, top_k: int, retrieve_only: bool) -> int:
    """Run a query and print the answer (or top-k chunks for retrieve-only)."""
    pipeline = RAGPipeline()
    try:
        result = pipeline.answer(question, k=top_k, retrieve_only=retrieve_only)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except MissingAPIKeyError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if result.answer is not None:
        print(result.answer)
        print("\nSources:")
        for chunk, score in result.chunks:
            print(f"  - {chunk.source} (score {score:.3f})")
    else:
        print(f"Top {len(result.chunks)} chunks for: {question}\n")
        for rank, (chunk, score) in enumerate(result.chunks, start=1):
            print(f"[{rank}] {chunk.source}  (score {score:.3f})")
            print(chunk.text)
            print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.rag.cli",
        description="Retrieval-augmented Q&A over the project reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build and persist the retrieval index.")

    query_parser = subparsers.add_parser(
        "query", help="Retrieve and answer a question."
    )
    query_parser.add_argument("question", help="The question to ask.")
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=f"Number of chunks to retrieve (default: {TOP_K}).",
    )
    query_parser.add_argument(
        "--retrieve-only",
        action="store_true",
        help="Print retrieved chunks only; skip generation (no API key needed).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        return _cmd_build()
    if args.command == "query":
        return _cmd_query(args.question, args.top_k, args.retrieve_only)

    parser.error(f"Unknown command: {args.command}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    sys.exit(main())
