"""Builds the real agent: hybrid retriever + mock backend client + LLM, wired into the graph."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings, get_settings
from app.llm.client import LLM
from app.retrieval.fusion import build_retriever

from app.agent.nodes import AgentDeps
from app.agent.graph import build_graph
from app.agent.tools import BackendClient, ToolRuntime, make_retrieve_fn
from app.agent.checkpoint import make_checkpointer
from app.agent.runner import AgentService

def build_agent(settings: Settings, retriever=None, backend=None, llm=None, checkpointer=None):
    """Returns (graph, deps). Any argument can be replaced, which is how the tests inject fakes."""
    retriever = retriever or build_retriever(settings)
    backend_url = get_settings().backend_url + get_settings().backend_context_path
    runtime = ToolRuntime(
        backend=backend or BackendClient(backend_url),
        retrieve=make_retrieve_fn(retriever, top_k=settings.top_k),
    )
    deps = AgentDeps(
        llm=LLM(settings.llm_api_key, settings.llm_model) if settings.llm_api_key is not None and settings.llm_api_key.strip() else None,
        runtime=runtime,
        history_window=settings.history_window,
    )
    if checkpointer is None:
        checkpointer = make_checkpointer(settings.memory_db)   # SQLite unless MEMORY_DB is empty
    return build_graph(deps, checkpointer or MemorySaver()), deps

def build_agent_service(settings: Settings, retriever=None, backend=None, llm=None, checkpointer=None) -> AgentService:
    graph, _ = build_agent(settings, retriever, backend, llm, checkpointer)
    return AgentService(graph, max_turns=settings.max_turns, timeout=settings.agent_timeout)