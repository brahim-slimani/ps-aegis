"""Central configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Settings:
    # --- ingestion / embeddings ---
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(ROOT / "data" / "docs"))))
    chunks_path: Path = field(default_factory=lambda: Path(_env("CHUNKS_PATH", str(ROOT / "data" / "chunks.jsonl"))))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    chunk_max_words: int = field(default_factory=lambda: int(_env("CHUNK_MAX_WORDS", "180")))
    chunk_overlap_words: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP_WORDS", "40")))

    # --- Qdrant: set QDRANT_URL for a server, otherwise embedded local mode is used ---
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", ""))
    qdrant_path: str = field(default_factory=lambda: _env("QDRANT_PATH", str(ROOT / "data" / "qdrant")))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "company_docs"))

    # --- retrieval ---
    candidates: int = field(default_factory=lambda: int(_env("RETRIEVAL_CANDIDATES", "30")))
    top_k: int = field(default_factory=lambda: int(_env("RETRIEVAL_TOP_K", "5")))

    # --- LLM (Gemini | Grock). Without a key, the system falls back to offline heuristics. ---
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", ""))
    grock_api_key: str = field(default_factory=lambda: _env("GROCK_API_KEY", ""))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-3.8-flash"))

    # --- mock enterprise backend ---
    backend_version: str = field(default_factory=lambda: _env("BACKEND_VERSION", "1.0"))
    backend_context_path: str = field(default_factory=lambda: _env("BACKEND_CONTEXT_PATH", "/mock/v1"))
    backend_url: str = field(default_factory=lambda: _env("BACKEND_URL", "http://localhost:8000"))
    backend_db: str = field(default_factory=lambda: _env("BACKEND_DB", str(ROOT / "data" / "enterprise.db")))

    # --- agent memory ---
    memory_db: str = field(default_factory=lambda: _env("MEMORY_DB", ""))  # "" => in-memory checkpointer
    history_window: int = field(default_factory=lambda: int(_env("HISTORY_WINDOW", "8")))
    max_turns: int = field(default_factory=lambda: int(_env("MAX_TURNS", "20")))              # user messages / conversation
    agent_timeout: float = field(default_factory=lambda: float(_env("AGENT_TIMEOUT", "60")))  # seconds / request
 
    rag_api_context_path: str = field(default_factory=lambda: _env("RAG_API_CONTEXT_PATH", "/rag/v1"))

def get_settings() -> Settings:
    return Settings()