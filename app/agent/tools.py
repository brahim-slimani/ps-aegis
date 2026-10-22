"""Tool registry: the only things the agent can do.

Each tool = a name, a pydantic argument model (validated BEFORE anything runs), a read/write flag,
and a `run(runtime, args)` function bound to either hybrid retrieval or the mock enterprise backend.
The planner can only name tools from TOOLS; the router rejects anything else.

    search_docs                 -> HybridRetriever            (Phase 2)
    list_/get_ tickets/orders/employees, create_ticket, update_ticket,
    update_order_status, cancel_order  -> HTTP calls to the backend  (Phase 3)

`write=True` tools never run from the executor in Phase 4: the router stages them as a pending action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, Field

TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
Priority = Literal["low", "medium", "high", "urgent"]
TicketCategory = Literal["general", "hardware", "software", "network", "access", "security", "hr"]
OrderStatus = Literal["pending", "processing", "shipped", "delivered", "cancelled"]


class ToolError(Exception):
    """A tool failed in an expected way (backend 4xx, unreachable, ...). The message is safe to show the user."""


class BackendClient:
    """Thin HTTP client for the mock enterprise backend."""

    def __init__(self, base_url: str, client: httpx.Client | None = None):
        self.http = client or httpx.Client(base_url=base_url, timeout=10.0)

    def call(self, method: str, path: str, params: dict | None = None, json: dict | None = None) -> Any:
        try:
            r = self.http.request(method, path, params=params, json=json)
        except httpx.HTTPError as e:
            raise ToolError(f"backend unreachable: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:
                detail = r.text
            raise ToolError(f"backend error {r.status_code}: {detail}")
        return r.json()


@dataclass
class ToolRuntime:
    """What tools need at run time. Injected, so tests can swap in fakes."""
    backend: Any                                  # object with .call(method, path, params=None, json=None)
    retrieve: Callable[[str], list[dict]]         # question -> list of context chunks


def hits_to_context(hits: list) -> list[dict]:
    return [
        {"n": i, "source": h.source, "title": h.title, "section": h.section, "score": round(float(h.score), 4),
         "text": h.text}
        for i, h in enumerate(hits, start=1)
    ]


def make_retrieve_fn(retriever, top_k: int = 5) -> Callable[[str], list[dict]]:
    return lambda question: hits_to_context(retriever.retrieve(question, top_k=top_k))


# ------------------------------ argument models ------------------------------
class SearchDocsArgs(BaseModel):
    question: str = Field(min_length=3)


class ListTicketsArgs(BaseModel):
    status: TicketStatus | None = None
    priority: Priority | None = None
    category: TicketCategory | None = None
    assignee_id: int | None = None
    requester_id: int | None = None


class GetTicketArgs(BaseModel):
    ticket_id: int


class CreateTicketArgs(BaseModel):
    title: str = Field(min_length=3, max_length=200)
    description: str = ""
    priority: Priority = "medium"
    category: TicketCategory = "general"
    requester_id: int | None = None   # defaults to the calling user


class UpdateTicketArgs(BaseModel):
    ticket_id: int
    status: TicketStatus | None = None
    priority: Priority | None = None
    assignee_id: int | None = None    # explicit null un-assigns


class ListOrdersArgs(BaseModel):
    status: OrderStatus | None = None
    customer: str | None = None


class GetOrderArgs(BaseModel):
    order_id: int


class UpdateOrderStatusArgs(BaseModel):
    order_id: int
    status: OrderStatus
    note: str | None = None


class CancelOrderArgs(BaseModel):
    order_id: int


class FindEmployeeArgs(BaseModel):
    q: str = Field(min_length=2)
    department: str | None = None


class GetEmployeeArgs(BaseModel):
    employee_id: int


def _params(args: BaseModel, exclude: set[str] | None = None, keep_explicit_none: bool = False) -> dict:
    """Only what the caller actually set. Explicit nulls are dropped unless they mean something (un-assign)."""
    data = args.model_dump(exclude_unset=True, exclude=exclude or set())
    return {k: v for k, v in data.items() if v is not None or keep_explicit_none}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    write: bool
    run: Callable[[ToolRuntime, Any], Any]
    summarize: Callable[[Any], str]
    returns_context: bool = False   # True: result is a list of retrieved passages, not backend data


def _get(path_fmt: str, id_field: str):
    return lambda rt, a: rt.backend.call("GET", path_fmt.format(getattr(a, id_field)))


_SPECS = [
    ToolSpec("search_docs", "Search the internal company documents (policies, handbooks, SOPs) for passages.",
             SearchDocsArgs, False, lambda rt, a: rt.retrieve(a.question),
             lambda a: f"Search documents for: {a.question}", returns_context=True),
    ToolSpec("list_tickets", "List support tickets, optionally filtered by status/priority/category/assignee/requester.",
             ListTicketsArgs, False, lambda rt, a: rt.backend.call("GET", "/tickets", params=_params(a)),
             lambda a: "List tickets"),
    ToolSpec("get_ticket", "Get one ticket by id.", GetTicketArgs, False, _get("/tickets/{}", "ticket_id"),
             lambda a: f"Get ticket #{a.ticket_id}"),
    ToolSpec("create_ticket", "Create a new support ticket.", CreateTicketArgs, True,
             lambda rt, a: rt.backend.call("POST", "/tickets", json=_params(a)),
             lambda a: f"Create a {a.priority}-priority ticket titled '{a.title}' (requester #{a.requester_id})"),
    ToolSpec("update_ticket", "Change a ticket's status, priority or assignee.", UpdateTicketArgs, True,
             lambda rt, a: rt.backend.call("PATCH", f"/tickets/{a.ticket_id}",
                                           json=_params(a, {"ticket_id"}, keep_explicit_none=True)),
             lambda a: f"Update ticket #{a.ticket_id} with "
                       + ", ".join(f"{k}={v}" for k, v in _params(a, {"ticket_id"}, True).items())),
    ToolSpec("list_orders", "List customer orders, optionally filtered by status or customer name.",
             ListOrdersArgs, False, lambda rt, a: rt.backend.call("GET", "/orders", params=_params(a)),
             lambda a: "List orders"),
    ToolSpec("get_order", "Get one order (with its status history) by id.", GetOrderArgs, False,
             _get("/orders/{}", "order_id"), lambda a: f"Get order #{a.order_id}"),
    ToolSpec("update_order_status",
             "Move an order to a new status: pending->processing->shipped->delivered, or cancelled from pending/processing.",
             UpdateOrderStatusArgs, True,
             lambda rt, a: rt.backend.call("PATCH", f"/orders/{a.order_id}/status", json=_params(a, {"order_id"})),
             lambda a: f"Change order #{a.order_id} status to '{a.status}'"),
    ToolSpec("cancel_order", "Cancel an order (only allowed while pending or processing).", CancelOrderArgs, True,
             lambda rt, a: rt.backend.call("POST", f"/orders/{a.order_id}/cancel"),
             lambda a: f"Cancel order #{a.order_id}"),
    ToolSpec("find_employee", "Search employees by name, email or job title.", FindEmployeeArgs, False,
             lambda rt, a: rt.backend.call("GET", "/employees", params=_params(a)),
             lambda a: f"Find employee '{a.q}'"),
    ToolSpec("get_employee", "Get one employee by id, with manager and ticket workload.", GetEmployeeArgs, False,
             _get("/employees/{}", "employee_id"), lambda a: f"Get employee #{a.employee_id}"),
]

TOOLS: dict[str, ToolSpec] = {s.name: s for s in _SPECS}


def _type_label(v: dict) -> str:
    if "enum" in v:
        return "|".join(str(x) for x in v["enum"])
    if "anyOf" in v:
        opts = [_type_label(o) for o in v["anyOf"] if o.get("type") != "null"]
        return "|".join(opts) or "any"
    return str(v.get("type", "any"))


def tool_catalog() -> str:
    """Compact tool description block for the planner prompt."""
    lines = []
    for s in TOOLS.values():
        props = s.args_model.model_json_schema().get("properties", {})
        sig = ", ".join(f"{k}: {_type_label(v)}" for k, v in props.items())
        lines.append(f"- {s.name}({sig}) [{'WRITE' if s.write else 'read'}]: {s.description}")
    return "\n".join(lines)