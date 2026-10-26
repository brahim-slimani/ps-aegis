"""AgentService: everything between the HTTP layer and the graph, in plain Python (so it is testable without FastAPI).

  * one request per conversation at a time (per-conversation lock), so a double click or a retry cannot run a
    confirmed action twice;
  * MAX_TURNS: a conversation accepts that many user messages (counted from the saved history); deciding a
    pending action is not a message and always goes through;
  * AGENT_TIMEOUT: a hard limit on the caller (504) plus a soft deadline inside the graph at 80% of it. After
    the soft deadline the graph starts no new tool call and no write, but still answers.

A Python thread cannot be killed. When a request times out its worker keeps running until the current call
returns, and keeps the conversation lock meanwhile. A request that was queued behind it notices on waking up that
its own time is over and is dropped instead of running late.

Why 80%: a write that was started before the soft deadline lasts at most BACKEND_TIMEOUT seconds, so with the
20% margin >= BACKEND_TIMEOUT (default: 12s >= 10s) it has finished before the 504 is sent. A shorter
AGENT_TIMEOUT loses that guarantee; a warning is logged.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.agent.guardrails import audit
from app.agent.checkpoint import count_user_turns
from app.agent.schemas import (
    AgentRequest,
    AgentResponse,
    Citation,
    ExecutedAction,
    HistoryMessage,
    HistoryResponse,
    PendingConfirmation,
)
from app.agent.state import confirm_turn, new_turn
from app.agent.tools import BACKEND_TIMEOUT

log = logging.getLogger(__name__)


class TurnLimitReached(Exception):
    """The conversation has used all its messages (HTTP 429)."""


class AgentTimeout(Exception):
    """The request ran out of time (HTTP 504)."""


class _KeyedLocks:
    """One lock per key, created on demand and forgotten when nobody holds or waits for it."""

    def __init__(self):
        self._guard = threading.Lock()
        self._locks: dict[str, list] = {}      # key -> [lock, holders + waiters]

    @contextmanager
    def hold(self, key: str):
        with self._guard:
            entry = self._locks.setdefault(key, [threading.Lock(), 0])
            entry[1] += 1
        entry[0].acquire()
        try:
            yield
        finally:
            entry[0].release()
            with self._guard:
                entry[1] -= 1
                if entry[1] == 0:
                    del self._locks[key]

    def __len__(self) -> int:
        with self._guard:
            return len(self._locks)


@dataclass
class _Job:
    conversation_id: str
    soft_deadline: float                 # time.monotonic(): graph starts no new tool call / write after this
    hard_deadline: float                 # time.monotonic(): the caller gets its 504 at this point
    cancelled: threading.Event = field(default_factory=threading.Event)   # the caller gave up

    def expired(self) -> bool:
        return self.cancelled.is_set() or time.monotonic() >= self.hard_deadline


def pending_view(pending: dict | None) -> PendingConfirmation | None:
    if not pending:
        return None
    return PendingConfirmation(action_id=pending["id"], tool=pending["tool"], args=pending["args"],
                               summary=pending["summary"])


def response_from_state(conversation_id: str, state: dict, max_turns: int) -> AgentResponse:
    used = count_user_turns(state.get("messages", []))
    return AgentResponse(
        conversation_id=conversation_id,
        answer=state.get("answer", ""),
        blocked=bool(state.get("blocked")),
        pending_confirmation=pending_view(state.get("pending_action")),
        executed_actions=[ExecutedAction(**a) for a in state.get("executed", [])],
        citations=[Citation(n=c["n"], source=c["source"], title=c["title"], section=c["section"],
                            score=c["score"], snippet=c["text"][:300]) for c in state.get("retrieved_context", [])],
        turns_used=used,
        turns_remaining=max(0, max_turns - used),
    )


class AgentService:
    def __init__(self, graph, *, max_turns: int = 20, timeout: float = 60.0, soft_fraction: float = 0.8,
                 workers: int = 8):
        if max_turns < 1 or timeout <= 0 or not 0 < soft_fraction < 1:
            raise ValueError("max_turns >= 1, timeout > 0 and 0 < soft_fraction < 1 are required")
        self.graph = graph
        self.max_turns = max_turns
        self.timeout = timeout
        self.soft_fraction = soft_fraction
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="agent")
        self._locks = _KeyedLocks()
        if timeout * (1 - soft_fraction) < BACKEND_TIMEOUT:
            log.warning("AGENT_TIMEOUT=%gs leaves %gs after the soft deadline, less than the %gs backend call "
                        "limit: a write started just before it may still be running when the 504 is sent",
                        timeout, timeout * (1 - soft_fraction), BACKEND_TIMEOUT)

    # ------------------------------------------------------------------ running a turn
    def _job(self, req: AgentRequest) -> _Job:
        now = time.monotonic()
        return _Job(req.conversation_id or uuid.uuid4().hex,
                    soft_deadline=now + self.timeout * self.soft_fraction,
                    hard_deadline=now + self.timeout)

    def run(self, req: AgentRequest) -> AgentResponse:
        """Blocking version (terminal chat, tests). The soft deadline applies; there is no hard 504."""
        out = self._work(req, self._job(req))
        if out is None:
            raise AgentTimeout(self._timeout_message())
        return out

    async def arun(self, req: AgentRequest) -> AgentResponse:
        """What the HTTP endpoint calls: runs in a worker thread, gives up after `timeout` seconds."""
        job = self._job(req)
        fut = asyncio.get_running_loop().run_in_executor(self._pool, self._work, req, job)
        try:
            out = await asyncio.wait_for(fut, timeout=self.timeout)
        except asyncio.TimeoutError:
            job.cancelled.set()          # a queued retry behind this one must not run late
            audit.warning("TIMEOUT conversation=%s after %gs", job.conversation_id, self.timeout)
            raise AgentTimeout(self._timeout_message()) from None
        except asyncio.CancelledError:   # the client went away
            job.cancelled.set()
            raise
        if out is None:
            raise AgentTimeout(self._timeout_message())
        return out

    def _timeout_message(self) -> str:
        return (f"The request took longer than {self.timeout:g} seconds. "
                "If it was a confirmation, check the conversation history before trying again.")

    def _work(self, req: AgentRequest, job: _Job) -> AgentResponse | None:
        if job.expired():                                # waited in the pool queue too long
            return None
        with self._locks.hold(job.conversation_id):
            if job.expired():                            # waited for the conversation lock too long: do not run late
                return None
            return self._turn(req, job)

    def _turn(self, req: AgentRequest, job: _Job) -> AgentResponse:
        config = {"configurable": {"thread_id": job.conversation_id}}
        if req.confirm is None:
            used = count_user_turns((self.graph.get_state(config).values or {}).get("messages", []))
            if used >= self.max_turns:
                audit.info("TURN LIMIT conversation=%s used=%d", job.conversation_id, used)
                raise TurnLimitReached(
                    f"This conversation has reached its limit of {self.max_turns} messages. "
                    "Start a new conversation.")
            turn = new_turn(req.message, user_id=req.user_id, role=req.role)
        else:
            turn = confirm_turn(req.confirm, req.action_id, user_id=req.user_id, role=req.role)
        turn["deadline"] = job.soft_deadline
        state = self.graph.invoke(turn, config)
        return response_from_state(job.conversation_id, state, self.max_turns)

    # ------------------------------------------------------------------ reading a conversation
    def history(self, conversation_id: str) -> HistoryResponse | None:
        values = self.graph.get_state({"configurable": {"thread_id": conversation_id}}).values
        if not values:
            return None
        messages = values.get("messages", [])
        used = count_user_turns(messages)
        return HistoryResponse(
            conversation_id=conversation_id,
            messages=[HistoryMessage(**m) for m in messages],
            pending_confirmation=pending_view(values.get("pending_action")),
            turns_used=used,
            max_turns=self.max_turns,
            turns_remaining=max(0, self.max_turns - used),
        )

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)