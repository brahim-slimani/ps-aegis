"""BGE embedder (sentence-transformers). Vectors are L2-normalised so cosine == dot product.

BGE v1.5 models expect an instruction prefix on *queries* (not on documents).
For bge-m3 the prefix is not needed; set EMBEDDING_MODEL accordingly and the
prefix is skipped automatically.
"""
from __future__ import annotations

import numpy as np

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy: heavy import

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    @property
    def _use_prefix(self) -> bool:
        return "bge-m3" not in self.model_name.lower()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vecs = self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 64
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        text = (QUERY_INSTRUCTION + query) if self._use_prefix else query
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)