"""Inspect retrieval without the API:  python -m app.retrieval.search_cli "how many leave days?"

Prints, for each hit, its rank in each retriever (vector / BM25) next to the fused and reranked
scores, which is the quickest way to see what each stage contributes.
"""
from __future__ import annotations

import argparse

from app.config import get_settings

from app.retrieval.fusion import build_retriever


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid search over the ingested documents")
    p.add_argument("query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--source", help="restrict to one document, e.g. leave_policy.md")
    args = p.parse_args()

    retriever = build_retriever(get_settings())
    hits = retriever.retrieve(args.query, top_k=args.top_k, source=args.source)
    if not hits:
        print("No results.")
        return
    print(f"{'#':<3}{'vec':>5}{'bm25':>6}{'rrf':>9}{'rerank':>9}  source > section")
    for i, h in enumerate(hits, start=1):
        rr = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"
        print(f"{i:<3}{h.vector_rank or '-':>5}{h.bm25_rank or '-':>6}{h.rrf:>9.4f}{rr:>9}  {h.source} > {h.section}")
        print(f"     {h.text[:120].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    main()