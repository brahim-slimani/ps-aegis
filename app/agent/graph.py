"""LangGraph orchestration (Phase 4).

    START -> planner -> router --tools--> executor --more?--> planner (loop, max `max_iterations` rounds)
                           |                  |
                           +--respond---------+--done--------> responder -> END

planner   reads the conversation, writes `plan`
router    validates the plan's tool calls, stages writes as `pending_action`, picks the route
executor  runs the READ tools: hybrid retrieval (search_docs) and the mock backend
responder writes `draft_answer` / `answer` and appends it to the conversation

Pass a checkpointer (e.g. MemorySaver) to keep `messages` between calls that share a thread_id.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import AgentDeps, executor_node, planner_node, responder_node, router_node, after_executor, after_router
from app.agent.state import AgentState


def build_graph(deps: AgentDeps, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("planner", lambda state: planner_node(state, deps))
    g.add_node("router", lambda state: router_node(state, deps))
    g.add_node("executor", lambda state: executor_node(state, deps))
    g.add_node("responder", lambda state: responder_node(state, deps))

    g.add_edge(START, "planner")
    g.add_edge("planner", "router")
    g.add_conditional_edges("router", after_router, {"executor": "executor", "responder": "responder"})
    g.add_conditional_edges(
        "executor",
        lambda state: after_executor(state, deps.max_iterations),
        {"planner": "planner", "responder": "responder"},
    )
    g.add_edge("responder", END)
    return g.compile(checkpointer=checkpointer)