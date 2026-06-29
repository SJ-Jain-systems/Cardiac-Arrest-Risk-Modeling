"""RAG-specific configuration: single source of truth for the retrieval layer.

Paths derive from :mod:`src.config` so there is no new path hardcoding. Model
ids, chunking parameters, and retrieval defaults live here and are imported by
the other ``src.rag`` modules.
"""

from src.config import MODELS_DIR

# --- Models ---
# Local embedding model: small, fast, CPU-friendly. Downloaded once on first use.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Generation model: the Claude API model used to synthesize grounded answers.
GENERATION_MODEL_NAME = "claude-opus-4-8"

# --- Chunking ---
# Markdown is split into ~500-800 char chunks with a small overlap so retrieval
# returns coherent passages without dropping context at boundaries.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- Retrieval ---
# Number of top chunks returned by the retriever / fed to the generator.
TOP_K = 5

# --- Generation ---
# Bounded output; well under the streaming threshold, so a plain non-streaming
# call is fine.
MAX_TOKENS = 4096

# --- Index persistence ---
# The index is written under MODELS_DIR (already git-ignored for build artifacts).
RAG_INDEX_DIR = MODELS_DIR / "rag_index"
EMBEDDINGS_PATH = RAG_INDEX_DIR / "embeddings.npz"
CHUNKS_PATH = RAG_INDEX_DIR / "chunks.json"
