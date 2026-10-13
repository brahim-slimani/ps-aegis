"""Hybrid retrieval: dense (Qdrant) + lexical (BM25) fused with Reciprocal Rank Fusion,
then re-scored by a cross-encoder for the final ordering."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.ingestion.embedder import Embedder
from app.retrieval.vector_store import VectorStore

from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranker import Reranker

@dataclass
class Hit:
    id: str
    text: str
    source: str
    title: str
    section: str
    score: float = 0.0           # final score (rerank score if reranker used, else RRF)
    rrf: float = 0.0
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    payload: dict = field(default_factory=dict, repr=False)


def rrf_fuse(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank). Ranks start at 1."""
    scores: dict[str, float] = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, embedder, store, bm25, reranker=None, candidates: int = 30, rrf_k: int = 60):
        self.embedder = embedder
        self.store = store
        self.bm25 = bm25
        self.reranker = reranker
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5, source: str | None = None) -> list[Hit]:
        qvec = self.embedder.embed_query(query)
        vec_hits = self.store.search(qvec, limit=self.candidates, source=source)
        bm25_hits = self.bm25.search(query, limit=self.candidates, source=source)

        pool: dict[str, Hit] = {}
        vec_order, bm_order = [], []
        for rank, h in enumerate(vec_hits, start=1):
            p = h.payload
            pool[h.id] = Hit(h.id, p.get("text", ""), p.get("source", ""), p.get("title", ""), p.get("section", ""),
                             vector_rank=rank, payload=p)
            vec_order.append(h.id)
        for rank, (c, _s) in enumerate(bm25_hits, start=1):
            hit = pool.get(c["id"]) or Hit(c["id"], c.get("text", ""), c.get("source", ""), c.get("title", ""),
                                           c.get("section", ""), payload=c)
            hit.bm25_rank = rank
            pool[c["id"]] = hit
            bm_order.append(c["id"])

        fused = rrf_fuse([vec_order, bm_order], k=self.rrf_k)
        for doc_id, s in fused.items():
            pool[doc_id].rrf = s
            pool[doc_id].score = s
        ranked = sorted(pool.values(), key=lambda h: h.rrf, reverse=True)[: self.candidates]

        if self.reranker and ranked:
            passages = [f"{h.title} > {h.section}\n{h.text}" if h.section else h.text for h in ranked]
            for h, s in zip(ranked, self.reranker.score(query, passages)):
                h.rerank_score = s
                h.score = s
            ranked.sort(key=lambda h: h.rerank_score, reverse=True)
        return ranked[:top_k]


def build_retriever(settings: Settings) -> HybridRetriever:
    bm25 = BM25Index.from_jsonl(settings.chunks_path)
    if not bm25.chunks:
        raise RuntimeError(
            f"{settings.chunks_path} is missing or empty. Run `python -m app.ingestion.ingest --recreate` first."
        )
    # An empty RERANKER_MODEL disables the cross-encoder (faster, lower precision).
    reranker = Reranker(settings.reranker_model) if settings.reranker_model else None
    return HybridRetriever(
        Embedder(settings.embedding_model),
        VectorStore(settings.qdrant_collection, settings.qdrant_url, settings.qdrant_path),
        bm25,
        reranker,
        candidates=settings.candidates,
    )
 