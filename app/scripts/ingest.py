"""Ingestion CLI:  python -m app.ingestion.ingest [--recreate]

docs_dir -> chunks -> BGE embeddings -> Qdrant, plus data/chunks.jsonl that
feeds the BM25 index at query time.
"""
from __future__ import annotations

import argparse
import json

from app.config import Settings, get_settings

from app.ingestion.chunker import Chunk, chunk_directory
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore


def write_chunks(chunks: list[Chunk], path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def run_ingestion(settings: Settings, embedder=None, store: VectorStore | None = None, recreate: bool = False) -> int:
    chunks = chunk_directory(settings.docs_dir, settings.chunk_max_words, settings.chunk_overlap_words)
    if not chunks:
        raise SystemExit(f"No supported documents found in {settings.docs_dir}")
    embedder = embedder or Embedder(settings.embedding_model)
    store = store or VectorStore(settings.qdrant_collection, settings.qdrant_url, settings.qdrant_path)
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store.ensure_collection(vectors.shape[1], recreate=recreate)
    store.upsert(chunks, vectors)
    write_chunks(chunks, settings.chunks_path)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest company documents into Qdrant + BM25 chunk file")
    parser.add_argument("--recreate", action="store_true", help="drop and recreate the Qdrant collection first")
    args = parser.parse_args()
    n = run_ingestion(get_settings(), recreate=args.recreate)
    print(f"Ingested {n} chunks")


if __name__ == "__main__":
    main()