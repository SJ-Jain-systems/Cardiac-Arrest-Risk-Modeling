"""Split loaded documents into retrieval chunks with source metadata.

Markdown is split on paragraph/heading boundaries and packed into ~``CHUNK_SIZE``
character chunks with a small overlap so passages stay coherent. CSV documents
are kept whole when small and split by rows (repeating the naming + column header
for context) when large. Chunking is fully deterministic so it is unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.loaders import Document

# CSV tables under this size stay a single chunk; larger tables are split by rows.
CSV_SINGLE_CHUNK_MAX = 2 * CHUNK_SIZE


@dataclass
class Chunk:
    """A retrieval chunk derived from a source document.

    Attributes:
        source: Source file (relative path), carried through for citations.
        chunk_id: Stable identifier ``"<source>#<index>"``.
        text: The chunk text.
    """

    source: str
    chunk_id: str
    text: str


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Hard-split a single oversized block into overlapping character windows."""
    pieces: list[str] = []
    start = 0
    n = len(text)
    step = max(1, chunk_size - overlap)
    while start < n:
        pieces.append(text[start : start + chunk_size])
        if start + chunk_size >= n:
            break
        start += step
    return pieces


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each chunk to the next so passages overlap slightly."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        result.append(f"{prev_tail}\n{chunks[i]}")
    return result


def _chunk_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack markdown paragraphs into chunks, hard-splitting oversized blocks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    packed: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            packed.append(current)
            current = ""

        if len(para) > chunk_size:
            pieces = _split_long_text(para, chunk_size, overlap)
            packed.extend(pieces[:-1])
            current = pieces[-1]
        else:
            current = para

    if current:
        packed.append(current)

    return _add_overlap(packed, overlap)


def _chunk_csv(text: str, chunk_size: int) -> list[str]:
    """Keep small CSV tables whole; split large ones by rows, repeating headers."""
    if len(text) <= CSV_SINGLE_CHUNK_MAX:
        return [text]

    lines = text.split("\n")
    naming = lines[0]  # "Table from <file>:"
    body = lines[1:]

    # Repeat the column header (and markdown separator, if present) on each chunk
    # so every split table fragment stays self-describing.
    header_rows: list[str] = []
    rows = body
    if len(body) >= 2 and set(body[1].replace("|", "").replace(":", "").strip()) <= {
        "-",
        "",
    }:
        header_rows = body[:2]
        rows = body[2:]
    elif body:
        header_rows = body[:1]
        rows = body[1:]

    prefix = "\n".join([naming, *header_rows])

    chunks: list[str] = []
    current_rows: list[str] = []
    current_len = len(prefix)
    for row in rows:
        if current_rows and current_len + len(row) + 1 > chunk_size:
            chunks.append("\n".join([prefix, *current_rows]))
            current_rows = []
            current_len = len(prefix)
        current_rows.append(row)
        current_len += len(row) + 1

    if current_rows:
        chunks.append("\n".join([prefix, *current_rows]))

    return chunks


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Split documents into retrieval chunks with stable source metadata.

    Args:
        documents: Documents from :func:`src.rag.loaders.load_reports`.

    Returns:
        A flat, deterministically ordered list of :class:`Chunk` objects.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        if doc.doc_type == "markdown":
            texts = _chunk_markdown(doc.text, CHUNK_SIZE, CHUNK_OVERLAP)
        else:
            texts = _chunk_csv(doc.text, CHUNK_SIZE)

        for index, chunk_text in enumerate(texts):
            chunks.append(
                Chunk(
                    source=doc.source,
                    chunk_id=f"{doc.source}#{index}",
                    text=chunk_text,
                )
            )

    return chunks


if __name__ == "__main__":
    from src.rag.loaders import load_reports

    docs = load_reports()
    all_chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(all_chunks)} chunks")
    for chunk in all_chunks[:5]:
        print(f"- {chunk.chunk_id} ({len(chunk.text)} chars)")
