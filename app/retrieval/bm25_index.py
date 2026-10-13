"""BM25 lexical index built from data/chunks.jsonl (written by ingestion)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "it",
    "this", "that", "with", "as", "by", "at", "from", "do", "does", "how", "what", "can", "i", "my", "we",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(self._doc_text(c)) for c in chunks]) if chunks else None

    @staticmethod
    def _doc_text(c: dict) -> str:
        return f"{c.get('title', '')} {c.get('section', '')} {c.get('text', '')}"

    @classmethod
    def from_jsonl(cls, path: Path) -> "BM25Index":
        path = Path(path)
        if not path.exists():
            return cls([])
        with path.open(encoding="utf-8") as f:
            return cls([json.loads(line) for line in f if line.strip()])

    def search(self, query: str, limit: int = 30, source: str | None = None) -> list[tuple[dict, float]]:
        if not self._bm25:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if source and self.chunks[i].get("source") != source:
                continue
            out.append((self.chunks[i], float(scores[i])))
            if len(out) >= limit:
                break
        return out