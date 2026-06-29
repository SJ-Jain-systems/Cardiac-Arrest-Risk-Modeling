"""Lightweight NumPy cosine-similarity vector index, persisted to disk.

Stores chunk embeddings as a single NumPy matrix plus parallel chunk metadata.
Because both document and query vectors are L2-normalized upstream, cosine
similarity is a plain dot product, so ``search`` is one matrix-vector product
followed by a top-k selection. Persistence is ``embeddings.npz`` (vectors) +
``chunks.json`` (metadata) — no FAISS/Chroma, no extra dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.rag.chunking import Chunk
from src.rag.config import CHUNKS_PATH, EMBEDDINGS_PATH, RAG_INDEX_DIR, TOP_K


class VectorStore:
    """An in-memory matrix of chunk embeddings with cosine-similarity search."""

    def __init__(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors ({len(vectors)}) and chunks ({len(chunks)}) "
                "must have matching lengths"
            )
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.chunks = chunks

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk]) -> VectorStore:
        """Build an index from precomputed vectors and their chunks."""
        return cls(vectors, chunks)

    def save(
        self,
        embeddings_path: Path = EMBEDDINGS_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> None:
        """Persist vectors (``.npz``) and chunk metadata (``.json``) to disk."""
        RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(embeddings_path, vectors=self.vectors)
        payload = [
            {"source": c.source, "chunk_id": c.chunk_id, "text": c.text}
            for c in self.chunks
        ]
        chunks_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls,
        embeddings_path: Path = EMBEDDINGS_PATH,
        chunks_path: Path = CHUNKS_PATH,
    ) -> VectorStore:
        """Load a previously saved index from disk."""
        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"No index found at {embeddings_path} / {chunks_path}. "
                "Build it first (e.g. `python -m src.rag.cli build`)."
            )
        with np.load(embeddings_path) as data:
            vectors = data["vectors"]
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(source=item["source"], chunk_id=item["chunk_id"], text=item["text"])
            for item in payload
        ]
        return cls(vectors, chunks)

    def search(
        self, query_vec: np.ndarray, k: int = TOP_K
    ) -> list[tuple[Chunk, float]]:
        """Return the top-``k`` chunks by cosine similarity to ``query_vec``.

        Args:
            query_vec: An L2-normalized ``(dim,)`` query vector.
            k: Number of results to return (clamped to the index size).

        Returns:
            A list of ``(chunk, score)`` pairs sorted by descending score.
        """
        if len(self.chunks) == 0:
            return []

        scores = self.vectors @ np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(self.chunks))
        # argpartition for the top-k, then sort just those by descending score.
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]
