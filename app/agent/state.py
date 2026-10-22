"""The graph state: everything the nodes read and write.

Two kinds of fields:

  persistent  `messages` (conversation history) and `pending_action` (a write waiting for confirmation).
  per-turn    everything else. `new_turn()` resets them on every invoke so one turn never leaks into the next.

Only `messages` has a reducer (operator.add: nodes *append*). All other fields are overwritten by the node
that returns them, which is also what makes the per-turn reset work.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

MAX_STEPS = 5          # tool calls the planner may request in one plan
MAX_ITERATIONS = 3     # plan -> tools -> re-plan rounds per turn

Role = Literal["employee", "support_agent", "manager", "admin"]
Intent = Literal["knowledge", "action", "chitchat"]
Route = Literal["tools", "respond"]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class PlanStep(TypedDict):
    tool: str
    args: dict[str, Any]


class Plan(TypedDict, total=False):
    intent: Intent
    question: str          # standalone rewrite of the user's question (pronouns resolved)
    steps: list[PlanStep]
    reply: str             # only for chitchat
    more: bool             # True when the planner needs these steps' results before planning further steps


class Observation(TypedDict, total=False):
    tool: str
    args: dict[str, Any]
    ok: bool
    result: Any
    error: str


class ContextChunk(TypedDict):
    """One retrieved passage; `n` is the citation number the answer may use as [n]."""
    n: int
    source: str
    title: str
    section: str
    score: float
    text: str


class PendingAction(TypedDict):
    """A write the agent wants to do but has not done: it needs the user's explicit confirmation first."""
    tool: str
    args: dict[str, Any]
    summary: str


class AgentState(TypedDict, total=False):
    # ---- persistent ----
    messages: Annotated[list[Message], operator.add]   # conversation history (append-only)
    pending_action: PendingAction | None

    # ---- per turn: input ----
    user_input: str
    user_id: int
    role: Role

    # ---- per turn: planning / routing ----
    plan: Plan | None
    route: Route
    steps: list[PlanStep]            # validated read steps the executor will run
    deferred_steps: list[PlanStep]   # steps queued behind a pending write (Phase 5 resumes them)
    iteration: int

    # ---- per turn: results ----
    observations: list[Observation]
    retrieved_context: list[ContextChunk]
    draft_answer: str                # the text the LLM (or offline fallback) wrote
    answer: str                      # draft + confirmation question, what the user sees


def new_turn(message: str, *, user_id: int = 1, role: Role = "employee",
             history: list[Message] | None = None) -> dict:
    """Initial graph input for one user message. Does NOT touch `pending_action` (it is persistent)."""
    turn: dict[str, Any] = {
        "user_input": message,
        "user_id": user_id,
        "role": role,
        "plan": None,
        "route": "respond",
        "steps": [],
        "deferred_steps": [],
        "iteration": 0,
        "observations": [],
        "retrieved_context": [],
        "draft_answer": "",
        "answer": "",
    }
    if history:
        turn["messages"] = list(history)   # stateless use: caller supplies the history; Phase 5 uses a checkpointer
    return turn


def recent_history(messages: list[Message], window: int) -> list[Message]:
    """Last `window` messages, trimmed so the slice starts with a user turn (LLM APIs require that)."""
    tail = messages[-window:] if window > 0 else list(messages)
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]
    return tail