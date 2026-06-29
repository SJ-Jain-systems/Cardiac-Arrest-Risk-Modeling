"""Unit tests for the RAG pipeline.

These tests use a tiny fixture corpus, a deterministic stub embedder, and a
mocked Anthropic client, so they require no model download and no network — they
are CI-safe and fast. Coverage: loaders find the known reports, chunking is
deterministic, the vector store round-trips and search returns the planted
chunk, the retriever wires embed -> search, and generation is exercised through
a mocked client (including the missing-key path).
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from src.rag.chunking import Chunk, chunk_documents
from src.rag.loaders import Document, load_reports
from src.rag.vector_store import VectorStore

# --- Deterministic stub embedder (no model download / network) ---

_FAKE_DIM = 32


def _fake_vector(text: str) -> np.ndarray:
    """Map text to a normalized character-frequency vector (deterministic)."""
    vec = np.zeros(_FAKE_DIM, dtype=np.float32)
    for char in text.lower():
        vec[ord(char) % _FAKE_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class FakeEmbedder:
    """Stand-in for :class:`~src.rag.embeddings.Embedder` used in tests."""

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, _FAKE_DIM), dtype=np.float32)
        return np.array([_fake_vector(t) for t in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return _fake_vector(text)


# --- Fixtures ---


def _fixture_documents() -> list[Document]:
    """A tiny in-memory corpus exercising both markdown and CSV paths."""
    markdown = (
        "# Threshold Analysis\n\n"
        "At threshold 0.45 the sensitivity is 0.82 and the specificity is 0.74.\n\n"
        "## Definitions\n\n"
        "Creatinine abnormality is defined as a value above 120 umol/L.\n"
    )
    csv_text = (
        "Table from metrics.csv:\n"
        "| metric | value |\n| --- | --- |\n| auc | 0.88 |\n| brier | 0.10 |"
    )
    return [
        Document("final_report.md", "markdown", markdown),
        Document("metrics.csv", "csv", csv_text),
    ]


# --- Loaders ---


def test_load_reports_finds_known_report_files() -> None:
    """The real reports directory should yield the known aggregate reports."""
    documents = load_reports()
    sources = {doc.source for doc in documents}

    assert "data_dictionary.md" in sources
    assert "final_report.md" in sources
    assert all(doc.doc_type in {"markdown", "csv"} for doc in documents)


# --- Chunking ---


def test_chunk_documents_is_deterministic() -> None:
    """Chunking the same documents twice yields identical chunks."""
    documents = _fixture_documents()

    first = chunk_documents(documents)
    second = chunk_documents(documents)

    assert [(c.chunk_id, c.text) for c in first] == [
        (c.chunk_id, c.text) for c in second
    ]


def test_chunk_documents_assigns_stable_ids_and_keeps_small_csv_whole() -> None:
    """Chunk ids follow '<source>#<index>' and a small CSV stays one chunk."""
    chunks = chunk_documents(_fixture_documents())

    assert all(chunk.chunk_id == f"{chunk.source}#0" for chunk in chunks[:1])
    csv_chunks = [c for c in chunks if c.source == "metrics.csv"]
    assert len(csv_chunks) == 1  # small table kept whole
    assert "Table from metrics.csv" in csv_chunks[0].text


# --- Vector store ---


def test_vector_store_save_load_round_trip(tmp_path) -> None:
    """Build -> save -> load preserves vectors and chunk metadata."""
    chunks = [
        Chunk("a.md", "a.md#0", "alpha"),
        Chunk("b.md", "b.md#0", "beta"),
    ]
    embedder = FakeEmbedder()
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore.build(vectors, chunks)

    emb_path = tmp_path / "embeddings.npz"
    chunks_path = tmp_path / "chunks.json"
    store.save(emb_path, chunks_path)
    loaded = VectorStore.load(emb_path, chunks_path)

    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in chunks]
    assert np.allclose(loaded.vectors, store.vectors)


def test_search_returns_planted_chunk_for_near_duplicate_query() -> None:
    """A near-duplicate query retrieves its planted chunk as the top hit."""
    embedder = FakeEmbedder()
    chunks = chunk_documents(_fixture_documents())
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore.build(vectors, chunks)

    query_vec = embedder.encode_query("creatinine abnormality definition")
    results = store.search(query_vec, k=1)

    assert len(results) == 1
    top_chunk, score = results[0]
    assert "creatinine" in top_chunk.text.lower()
    assert score > 0


def test_search_clamps_k_to_index_size() -> None:
    """Requesting more results than chunks returns all chunks, no error."""
    embedder = FakeEmbedder()
    chunks = [Chunk("a.md", "a.md#0", "alpha")]
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore.build(vectors, chunks)

    results = store.search(embedder.encode_query("alpha"), k=99)

    assert len(results) == 1


# --- Retriever (embed -> search), with stubbed embedder and in-memory index ---


def test_retriever_returns_top_k_for_query(monkeypatch) -> None:
    """The retriever embeds the query and returns top-k chunks from the index."""
    from src.rag import retriever as retriever_module

    embedder = FakeEmbedder()
    chunks = chunk_documents(_fixture_documents())
    vectors = embedder.encode_documents([c.text for c in chunks])
    store = VectorStore.build(vectors, chunks)

    # Avoid touching disk: load() returns our in-memory store.
    monkeypatch.setattr(VectorStore, "load", classmethod(lambda cls: store))

    retr = retriever_module.Retriever(embedder=embedder)
    results = retr.retrieve("sensitivity at threshold 0.45", k=2)

    assert 1 <= len(results) <= 2
    assert all(isinstance(score, float) for _, score in results)


# --- Generation (mocked Anthropic client) ---


def _install_fake_anthropic(monkeypatch, answer_text: str) -> None:
    """Inject a fake `anthropic` module whose client returns ``answer_text``."""
    fake = types.ModuleType("anthropic")

    text_block = types.SimpleNamespace(type="text", text=answer_text)
    thinking_block = types.SimpleNamespace(type="thinking", thinking="")
    response = types.SimpleNamespace(content=[thinking_block, text_block])

    class FakeMessages:
        def create(self, **kwargs):
            return response

    class FakeAnthropic:
        def __init__(self, *args, **kwargs):
            self.messages = FakeMessages()

    fake.Anthropic = FakeAnthropic
    fake.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.APIStatusError = type("APIStatusError", (Exception,), {})
    fake.APIConnectionError = type("APIConnectionError", (Exception,), {})

    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_generate_answer_uses_mocked_client(monkeypatch) -> None:
    """Generation returns the model's text, ignoring empty thinking blocks."""
    from src.rag.generator import generate_answer

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    _install_fake_anthropic(monkeypatch, "Sensitivity is 0.82 (final_report.md).")

    chunks = chunk_documents(_fixture_documents())
    answer = generate_answer("What is the sensitivity?", chunks)

    assert answer == "Sensitivity is 0.82 (final_report.md)."


def test_generate_answer_raises_without_api_key(monkeypatch) -> None:
    """Generation without a key raises the actionable MissingAPIKeyError."""
    from src.rag.generator import MissingAPIKeyError, generate_answer

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingAPIKeyError):
        generate_answer("anything", [])
