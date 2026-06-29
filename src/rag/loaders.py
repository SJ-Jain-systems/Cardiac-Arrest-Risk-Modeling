"""Discover and read the project's report artifacts into raw documents.

Walks ``REPORTS_DIR`` for Markdown and CSV files and returns them as plain-text
:class:`Document` objects. Markdown is read verbatim; CSV tables are rendered to
a compact text table (prefixed with a header line naming the file) so numeric
results become retrievable text. Only aggregate reports are indexed; the raw
patient CSV lives under ``data/`` and is never touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR


@dataclass
class Document:
    """A single source document discovered under the reports directory.

    Attributes:
        source: Path to the file, relative to ``REPORTS_DIR`` (used as a citation).
        doc_type: ``"markdown"`` or ``"csv"``.
        text: The document content as plain text.
    """

    source: str
    doc_type: str
    text: str


def _render_csv(path: Path) -> str:
    """Render a CSV file as a compact text table with a naming header line."""
    df = pd.read_csv(path)
    header = f"Table from {path.name}:"
    try:
        # Preferred: GitHub-style markdown table (needs the optional `tabulate`).
        body = df.to_markdown(index=False)
    except ImportError:
        # Fallback keeps loaders dependency-free if `tabulate` is unavailable.
        body = df.to_string(index=False)
    return f"{header}\n{body}"


def load_reports(reports_dir: Path = REPORTS_DIR) -> list[Document]:
    """Load all Markdown and CSV reports under ``reports_dir``.

    Markdown files are read as text; CSV files are rendered to a compact text
    table. Results are sorted by relative path so the corpus is deterministic.

    Args:
        reports_dir: Root directory to walk. Defaults to ``REPORTS_DIR``.

    Returns:
        A list of :class:`Document` objects, one per discovered file.
    """
    documents: list[Document] = []

    for path in sorted(reports_dir.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        relative_source = str(path.relative_to(reports_dir))

        if suffix == ".md":
            text = path.read_text(encoding="utf-8")
            documents.append(Document(relative_source, "markdown", text))
        elif suffix == ".csv":
            text = _render_csv(path)
            documents.append(Document(relative_source, "csv", text))

    return documents


if __name__ == "__main__":
    docs = load_reports()
    print(f"Loaded {len(docs)} documents from {REPORTS_DIR}")
    for doc in docs:
        print(f"- [{doc.doc_type}] {doc.source} ({len(doc.text)} chars)")
