"""Grounded answer generation via the Claude API.

Builds a context block from retrieved chunks plus a grounding system prompt and
calls ``anthropic.Anthropic().messages.create`` with ``claude-opus-4-8``. The
system prompt instructs the model to answer only from the provided context, cite
sources by filename, and decline when the context is insufficient — important
for a clinical project where hallucination must be minimized.

Retrieval never needs an API key; only this module does. The key is resolved
from ``ANTHROPIC_API_KEY``; if it is missing, :func:`generate_answer` raises a
clear, actionable error pointing the user to set it or use ``--retrieve-only``.
"""

from __future__ import annotations

import os

from src.rag.chunking import Chunk
from src.rag.config import GENERATION_MODEL_NAME, MAX_TOKENS

SYSTEM_PROMPT = (
    "You answer questions about a cardiac arrest risk modeling project using "
    "ONLY the provided report excerpts. Follow these rules strictly:\n"
    "- Base every statement solely on the provided context. Do not use outside "
    "knowledge or make assumptions.\n"
    "- Cite the source filename(s) in parentheses for each claim, e.g. "
    "(final_report.md).\n"
    "- If the context does not contain enough information to answer, reply "
    'exactly: "I don\'t know based on the reports."\n'
    "- This project is a research/analysis artifact, not a medical device; do "
    "not give clinical advice."
)


class MissingAPIKeyError(RuntimeError):
    """Raised when generation is attempted without an Anthropic API key."""


def _build_context_block(chunks: list[Chunk]) -> str:
    """Render retrieved chunks into a numbered, source-labeled context block."""
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(f"[{index}] Source: {chunk.source}\n{chunk.text}")
    return "\n\n".join(sections)


def generate_answer(question: str, chunks: list[Chunk]) -> str:
    """Generate a grounded answer to ``question`` from retrieved ``chunks``.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved chunks providing the grounding context.

    Returns:
        The model's grounded answer text.

    Raises:
        MissingAPIKeyError: If ``ANTHROPIC_API_KEY`` is not set.
        RuntimeError: If the Claude API call fails.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Generation requires a Claude API key. "
            "Set it (export ANTHROPIC_API_KEY=...) or run with --retrieve-only to "
            "skip generation and see the retrieved excerpts."
        )

    # Imported lazily so the module (and retrieval-only paths) don't require the
    # anthropic package to be importable.
    import anthropic

    context_block = _build_context_block(chunks)
    user_content = (
        f"Context from the project reports:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=GENERATION_MODEL_NAME,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.RateLimitError as exc:
        raise RuntimeError(
            f"Claude API rate limit hit while generating an answer: {exc}"
        ) from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(
            f"Claude API returned an error ({exc.status_code}): {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to the Claude API: {exc}") from exc

    answer_parts = [block.text for block in response.content if block.type == "text"]
    return "\n".join(answer_parts).strip()
