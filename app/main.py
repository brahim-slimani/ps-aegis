"""uvicorn app.api.main:app --port 8000

POST /ask  -> hybrid retrieval (vector + BM25 -> RRF -> cross-encoder) and a cited answer.
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
from app.backend_mock import routes
from app.backend_mock import service

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

rag_router = APIRouter(prefix="/rag/v1")
@rag_router.get("/health")
def health():
    svc: Services = app.state.services
    return {"status": "ok", "llm": svc.llm is not None, "reranker": svc.retriever.reranker is not None}

@rag_router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    svc: Services = app.state.services
    
    retriever = build_retriever(get_settings())
    hits = retriever.retrieve(query=req.question, top_k=req.top_k, source=req.source)
    citations = [_citation(i, h) for i, h in enumerate(hits, start=1)]

    best = hits[0]
    answer = f"(No LLM configured yet, best matching from {best.source}) {best.text}" 
    return {"answer": answer, "citations": citations, "grounded" : True}


def create_app(services: Services | None = None, db_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        path = db_path or get_settings().backend_db
        routes._initialise(path)
        app.state.db_path = path
        yield

    app = FastAPI(
        title="PS RAG Enterprise Backend", 
        version=get_settings().backend_version, 
        description=routes.DESCRIPTION,
        openapi_tags=routes.TAGS,
        lifespan=lifespan,
    )

    # ---- domain errors -> HTTP ----
    def _handler(code: int):
        async def handle(_: Request, exc: Exception):
            return JSONResponse(status_code=code, content={"detail": str(exc)})
        return handle

    app.add_exception_handler(service.NotFound, _handler(404))
    app.add_exception_handler(service.Conflict, _handler(409))
    app.add_exception_handler(service.Invalid, _handler(422))


    app.include_router(rag_router)
    app.include_router(routes.getBackendMockRouter())
    return app

def _default_app() -> FastAPI:
    return create_app()

app = _default_app()