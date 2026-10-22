"""Thin Qdrant wrapper. Uses a server when QDRANT_URL is set, else embedded local mode."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient, models

from app.ingestion.chunker import Chunk


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict


class VectorStore:
    def __init__(self, collection: str, url: str = "", path: str = ":memory:", client: QdrantClient | None = None):
        self.collection = collection
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url)
        else:
            self.client = QdrantClient(path=path) if path != ":memory:" else QdrantClient(":memory:")

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                self.collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            self.client.create_payload_index(
                self.collection, field_name="source", field_schema=models.PayloadSchemaType.KEYWORD
            )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch: int = 128) -> None:
        for i in range(0, len(chunks), batch):
            points = [
                models.PointStruct(id=c.id, vector=v.tolist(), payload=c.to_dict())
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch])
            ]
            self.client.upsert(self.collection, points=points)

    def search(self, vector: np.ndarray, limit: int = 30, source: str | None = None) -> list[VectorHit]:
        flt = None
        if source:
            flt = models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=source))])
        try:    
            res = self.client.query_points(
                self.collection, query=vector.tolist(), limit=limit, query_filter=flt, with_payload=True
            )
        finally:
            print('EOF search >>') #self.client.close()
        return [VectorHit(id=str(p.id), score=float(p.score), payload=p.payload or {}) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count