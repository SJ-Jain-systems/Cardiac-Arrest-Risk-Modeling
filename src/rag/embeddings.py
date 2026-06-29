"""Local sentence-transformers wrapper for encoding documents and queries.

Wraps a :class:`~sentence_transformers.SentenceTransformer` model and returns
L2-normalized ``float32`` vectors so cosine similarity reduces to a dot product
in the vector store. The model is lazy-loaded (only on the first ``encode`` call)
so importing this module is cheap and the embedder is easy to stub in tests.

Embeddings run fully offline once the model is cached locally; no API key is
required for the retrieval path.
"""

from __future__ import annotations

import numpy as np

from src.rag.config import EMBEDDING_MODEL_NAME


class Embedder:
    """Encodes text into normalized embedding vectors via sentence-transformers."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None  # lazy-loaded on first encode

    def _ensure_model(self):
        """Load the SentenceTransformer model on first use."""
        if self._model is None:
            # Imported lazily so module import stays cheap and dependency-light.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized float32 vectors."""
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document chunks into a ``(n, dim)`` normalized matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._encode(texts)

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query into a ``(dim,)`` normalized vector."""
        return self._encode([text])[0]
