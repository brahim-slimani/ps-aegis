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
GateDecision = Literal["plan", "execute", "respond"]
 
 
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
    """A write the agent wants to do but has not done. It runs only when a request carries confirm=true
    together with this `id`."""
    id: str                # random, 12 hex characters: what the client must send back to decide
    tool: str
    args: dict[str, Any]
    summary: str
 
 
class ExecutedAction(TypedDict):
    tool: str
    args: dict[str, Any]
    summary: str
    result: Any
 
 
class AgentState(TypedDict, total=False):
    # ---- persistent ----
    messages: Annotated[list[Message], operator.add]   # conversation history (append-only)
    pending_action: PendingAction | None
    deferred_steps: list[PlanStep]   # steps queued behind a pending write; resumed once it has run
 
    # ---- per turn: input ----
    user_input: str                  # "" on a confirmation turn
    user_id: int
    role: Role
    confirm: bool | None             # None: ordinary message. True / False: the user's decision on pending_action
    action_id: str | None            # id of the action the decision is about
    deadline: float | None           # time.monotonic() after which no new tool call or write is started
 
    # ---- per turn: guard / gate ----
    blocked: bool                    # a guardrail refused the message (nothing is stored in the history)
    gate_decision: GateDecision      # where the gate sends this turn
    resume: bool                     # a confirmed write ran: continue with the steps queued behind it
 
    # ---- per turn: planning / routing ----
    plan: Plan | None
    route: Route
    steps: list[PlanStep]            # validated read steps the executor will run
    iteration: int
 
    # ---- per turn: results ----
    observations: list[Observation]
    executed: list[ExecutedAction]   # writes that ran this turn
    retrieved_context: list[ContextChunk]
    draft_answer: str                # the text the LLM (or offline fallback) wrote
    answer: str                      # draft + confirmation question, what the user sees
 
 
def _fresh_turn(user_id: int, role: Role) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "role": role,
        "confirm": None,
        "action_id": None,
        "deadline": None,
        "blocked": False,
        "gate_decision": "plan",
        "resume": False,
        "plan": None,
        "route": "respond",
        "steps": [],
        "iteration": 0,
        "observations": [],
        "executed": [],
        "retrieved_context": [],
        "draft_answer": "",
        "answer": "",
    }
 
 
def new_turn(message: str, *, user_id: int = 1, role: Role = "employee",
             history: list[Message] | None = None) -> dict:
    """Initial graph input for one user message.
 
    Does not touch `pending_action` (persistent): the gate drops it, because a new message supersedes it.
    `deferred_steps` is cleared for the same reason.
    """
    turn = {**_fresh_turn(user_id, role), "user_input": message, "deferred_steps": []}
    if history:
        turn["messages"] = list(history)   # stateless use: caller supplies the history; /agent uses the checkpointer
    return turn
 
 
def confirm_turn(confirm: bool, action_id: str | None, *, user_id: int = 1, role: Role = "employee") -> dict:
    """Initial graph input for the user's decision on the pending action.
 
    Leaves `pending_action` and `deferred_steps` alone: they are what the decision is about.
    """
    return {**_fresh_turn(user_id, role), "user_input": "", "confirm": confirm, "action_id": action_id}
 
 
def recent_history(messages: list[Message], window: int) -> list[Message]:
    """Last `window` messages, trimmed so the slice starts with a user turn (LLM APIs require that)."""
    tail = messages[-window:] if window > 0 else list(messages)
    while tail and tail[0]["role"] != "user":
        tail = tail[1:]
    return tail