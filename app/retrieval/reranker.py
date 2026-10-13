"""Cross-encoder reranker (BGE reranker via sentence-transformers CrossEncoder)."""
from __future__ import annotations


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # lazy: heavy import

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], show_progress_bar=False)
        return [float(s) for s in scores]