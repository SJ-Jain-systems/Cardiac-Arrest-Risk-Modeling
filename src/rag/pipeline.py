"""High-level RAG pipeline tying retrieval and generation together.

:class:`RAGPipeline` exposes two operations: ``build_index`` (offline, key-free)
and ``answer`` (retrieve, then optionally generate a grounded answer). Retrieval
works without an API key; generation requires ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunking import Chunk
from src.rag.config import TOP_K
from src.rag.generator import generate_answer
from src.rag.retriever import Retriever, build_index
from src.rag.vector_store import VectorStore


@dataclass
class AnswerResult:
    """The result of an ``answer`` call.

    Attributes:
        question: The original question.
        chunks: Retrieved ``(chunk, score)`` pairs used as context/citations.
        answer: The grounded answer, or ``None`` when ``retrieve_only`` was set.
    """

    question: str
    chunks: list[tuple[Chunk, float]]
    answer: str | None


class RAGPipeline:
    """Orchestrates index building and grounded question answering."""

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or Retriever()

    def build_index(self) -> VectorStore:
        """Build and persist the retrieval index from ``reports/``."""
        return build_index()

    def answer(
        self,
        question: str,
        k: int = TOP_K,
        retrieve_only: bool = False,
    ) -> AnswerResult:
        """Retrieve top-``k`` chunks and (unless ``retrieve_only``) generate.

        Args:
            question: The natural-language question.
            k: Number of chunks to retrieve.
            retrieve_only: If ``True``, skip generation (no API key needed).

        Returns:
            An :class:`AnswerResult` with the retrieved chunks and, unless
            ``retrieve_only``, a grounded answer.
        """
        chunks = self._retriever.retrieve(question, k=k)

        if retrieve_only:
            return AnswerResult(question=question, chunks=chunks, answer=None)

        answer_text = generate_answer(question, [chunk for chunk, _ in chunks])
        return AnswerResult(question=question, chunks=chunks, answer=answer_text)
