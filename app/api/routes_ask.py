""" POST /ask  -> hybrid retrieval (vector + BM25 -> RRF -> cross-encoder) and a cited answer. 
"""

from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Annotated
from fastapi import Body, Depends, FastAPI, Path, Query, Request, APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.llm import client
from app.retrieval.fusion import build_retriever
from app.agent.factory import build_agent_service
from app.agent.runner import AgentService

@dataclass
class Services:
    settings: Settings
    retriever: Any
    llm: Any
    agent: AgentService

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


def build_services(settings: Settings | None = None) -> Services:
    s = settings or get_settings()
    retriever = build_retriever(s)
    llm = client.LLM(s.llm_api_key, s.llm_model)
    agent = build_agent_service(s, retriever=retriever, llm=llm)
    return Services(s, retriever, llm, agent)


rag_router = APIRouter(prefix=get_settings().rag_api_context_path)
@rag_router.get("/health")
def health():
    svc: Services = build_services()
    return {"status": "ok", "llm": svc.llm is not None and svc.llm.available, "reranker": svc.retriever.reranker is not None, "agent": svc.agent is not None}

@rag_router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    s: Settings = get_settings()
    retriever = build_retriever(s)
    hits = retriever.retrieve(query=req.question, top_k=req.top_k, source=req.source)
    return client.hybrid_answer(req.question, hits, client.LLM(s.llm_api_key, s.llm_model))

def get_rag_router() -> APIRouter:
    return rag_router