"""uvicorn app.api.main:app --port 8000

POST /ask  -> hybrid retrieval (vector + BM25 -> RRF -> cross-encoder) and a cited answer.
(The /agent endpoint is added in Phase 5.)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.retrieval.fusion import build_retriever
from app.retrieval.fusion import Hit

@dataclass
class Services:
    settings: Settings
    retriever: Any
    llm: Any


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    return Services(s, build_retriever(s), None)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20, description="Number of passages to retrieve")
    source: str | None = Field(None, description="Restrict to one document, e.g. 'leave_policy.md'")


class Citation(BaseModel):
    n: int
    source: str
    title: str
    section: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool

def _citation(n: int, h: Hit) -> dict:
    return {
        "n": n,
        "source": h.source,
        "title": h.title,
        "section": h.section,
        "score": round(h.score, 4),
        "snippet": h.text[:300],
    }

router = APIRouter(prefix="/rag/v1")
@router.get("/health")
def health():
    svc: Services = app.state.services
    return {"status": "ok", "llm": svc.llm is not None, "reranker": svc.retriever.reranker is not None}

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    svc: Services = app.state.services
    
    retriever = build_retriever(get_settings())
    hits = retriever.retrieve(query=req.question, top_k=req.top_k, source=req.source)
    citations = [_citation(i, h) for i, h in enumerate(hits, start=1)]

    best = hits[0]
    answer = f"(No LLM configured yet, best matching from {best.source}) {best.text}" 
    return {"answer": answer, "citations": citations, "grounded" : True}


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        yield

    app = FastAPI(title="PS Enterprise RAG", version="1.0", lifespan=lifespan)
    app.include_router(router)
    return app

def _default_app() -> FastAPI:
    return create_app()

app = _default_app()