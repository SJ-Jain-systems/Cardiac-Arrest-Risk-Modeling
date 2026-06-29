"""Retrieval-augmented generation (RAG) layer for the project's reports.

This subpackage indexes the natural-language and tabular knowledge artifacts
under ``reports/`` and answers grounded, cited questions about them. It is
isolated from the analysis code: importing or installing the core project does
not pull in the RAG dependencies, which live in the optional ``rag`` extra
(``pip install -e ".[rag]"``).

Retrieval (local ``sentence-transformers`` embeddings + a NumPy cosine-similarity
index) needs no API key. Only generation calls the Claude API and requires
``ANTHROPIC_API_KEY``.
"""
