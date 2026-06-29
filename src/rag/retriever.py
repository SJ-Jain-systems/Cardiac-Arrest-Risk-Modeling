"""Retrieval orchestration: build the index and fetch top-k chunks for a query.

``build_index`` runs the full offline pipeline (load reports -> chunk -> embed ->
persist the NumPy index). :class:`Retriever` lazily loads that index and the
embedder once, then answers queries by embedding the question and searching the
store. No LLM or API key is involved here — this is the complete retrieval path.
"""

from __future__ import annotations

from src.rag.chunking import Chunk, chunk_documents
from src.rag.config import TOP_K
from src.rag.embeddings import Embedder
from src.rag.loaders import load_reports
from src.rag.vector_store import VectorStore


def build_index(embedder: Embedder | None = None) -> VectorStore:
    """Load reports, chunk, embed, and persist the index to disk.

    Args:
        embedder: Optional embedder (injectable for tests). Defaults to a fresh
            :class:`~src.rag.embeddings.Embedder`.

    Returns:
        The built :class:`~src.rag.vector_store.VectorStore` (also saved to disk).
    """
    embedder = embedder or Embedder()

    documents = load_reports()
    chunks = chunk_documents(documents)
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])

    store = VectorStore.build(vectors, chunks)
    store.save()
    return store


class Retriever:
    """Loads the persisted index once and retrieves top-k chunks per query."""

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()
        self._store: VectorStore | None = None

    def _ensure_store(self) -> VectorStore:
        """Load the index from disk on first use and cache it."""
        if self._store is None:
            self._store = VectorStore.load()
        return self._store

    def retrieve(self, question: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        """Embed ``question`` and return the top-``k`` ``(chunk, score)`` pairs."""
        store = self._ensure_store()
        query_vec = self._embedder.encode_query(question)
        return store.search(query_vec, k=k)
