"""Builds the real agent: hybrid retriever + mock backend client + LLM, wired into the graph."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from app.config import Settings, get_settings
#from app.llm import LLM
from app.retrieval.fusion import build_retriever

from app.agent.nodes import AgentDeps
from app.agent.graph import build_graph
from app.agent.tools import BackendClient, ToolRuntime, make_retrieve_fn


def build_agent(settings: Settings, retriever=None, backend=None, llm=None, checkpointer=None):
    """Returns (graph, deps). Any argument can be replaced, which is how the tests inject fakes."""
    retriever = retriever or build_retriever(settings)
    backend_url = get_settings().backend_url + get_settings().backend_context_path
    runtime = ToolRuntime(
        backend=backend or BackendClient(backend_url),
        retrieve=make_retrieve_fn(retriever, top_k=settings.top_k),
    )
    deps = AgentDeps(
        llm=llm or None, #LLM(settings.llm_api_key, settings.llm_model),
        runtime=runtime,
        history_window=settings.history_window,
    )
    return build_graph(deps, checkpointer or MemorySaver()), deps