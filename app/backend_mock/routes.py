"""Mock enterprise backend (FastAPI + SQLite).


The three headline operations (operation ids are what an agent tool would call):
    create_ticket        POST  /tickets
    update_order_status  PATCH /orders/{order_id}/status
    get_employee_info    GET   /employees/{employee_id}
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, Depends, FastAPI, Path, Query, Request, APIRouter
from fastapi.responses import JSONResponse

from app.config import get_settings

from app.backend_mock import service
from app.backend_mock.db import connect, init_db
from app.backend_mock.models import (
    EmployeeInfo,
    EmployeeOut,
    EmployeeStatus,
    ErrorResponse,
    OrderCancel,
    OrderDetail,
    OrderOut,
    OrderStatus,
    OrderStatusUpdate,
    Priority,
    TicketCategory,
    TicketCreate,
    TicketOut,
    TicketStatus,
    TicketUpdate,
)
from app.backend_mock.seed_data import seed

TAGS = [
    {"name": "employees", "description": "Staff directory."},
    {"name": "tickets", "description": "IT / HR support tickets."},
    {"name": "orders", "description": "Customer orders and their status lifecycle."},
    {"name": "system", "description": "Health check."},
]

NOT_FOUND = {404: {"model": ErrorResponse, "description": "Record not found"}}
CONFLICT = {409: {"model": ErrorResponse, "description": "Not allowed in the record's current state"}}
INVALID = {422: {"description": "Validation error (malformed body, or it refers to something that does not exist)"}}

DESCRIPTION = """
Small in-company system of record used by the agent. Data lives in SQLite and is seeded on first start.

**Order lifecycle:** `pending → processing → shipped → delivered`; `pending` and `processing` orders can also go
to `cancelled`. `delivered` and `cancelled` are final.

**Errors** always look like `{"detail": "..."}` with status 404 (not found), 409 (rule violated) or 422 (invalid input).
"""


def _initialise(db_path: str) -> None:
    init_db(db_path)
    conn = connect(db_path)
    try:
        seed(conn)  # no-op when the database already has data
    finally:
        conn.close()


def get_db(request: Request):
    """One connection per request, closed afterwards. The path is stored on app.state by create_app."""
    conn = connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


# Module level on purpose: FastAPI resolves string annotations (from __future__ import annotations)
# against module globals, so this alias must not live inside create_app().
DB = Annotated[sqlite3.Connection, Depends(get_db)]

router = APIRouter(prefix=get_settings().backend_context_path)
# ------------------------------------------------------------------ system
@router.get("/health", tags=["system"], operation_id="health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------------------ employees
@router.get("/employees", response_model=list[EmployeeOut], tags=["employees"], operation_id="list_employees",
            summary="Search employees")
def list_employees(
    db: DB,
    q: Annotated[str | None, Query(description="Matches name, email or job title (case-insensitive)")] = None,
    department: Annotated[str | None, Query(description="Exact department name, e.g. IT")] = None,
    status: Annotated[EmployeeStatus | None, Query()] = None,
):
    return service.list_employees(db, q, department, status)

@router.get(
    "/employees/{employee_id}",
    response_model=EmployeeInfo,
    tags=["employees"],
    operation_id="get_employee_info",
    summary="Get employee info",
    description="Returns the employee record plus their manager, number of direct reports and "
                "how many open tickets they raised or are assigned.",
    responses={**NOT_FOUND},
)
def get_employee_info(employee_id: Annotated[int, Path(ge=1, description="Employee id")], db: DB):
    return service.get_employee_info(db, employee_id)

# ------------------------------------------------------------------ tickets
@router.get("/tickets", response_model=list[TicketOut], tags=["tickets"], operation_id="list_tickets",
            summary="List tickets (newest first)")
def list_tickets(
    db: DB,
    status: TicketStatus | None = None,
    priority: Priority | None = None,
    category: TicketCategory | None = None,
    assignee_id: Annotated[int | None, Query(ge=1)] = None,
    requester_id: Annotated[int | None, Query(ge=1)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    return service.list_tickets(db, status, priority, category, assignee_id, requester_id, limit)

@router.get("/tickets/{ticket_id}", response_model=TicketOut, tags=["tickets"], operation_id="get_ticket",
            summary="Get one ticket", responses={**NOT_FOUND})
def get_ticket(ticket_id: Annotated[int, Path(ge=1)], db: DB):
    return service.get_ticket(db, ticket_id)

@router.post(
    "/tickets",
    response_model=TicketOut,
    status_code=201,
    tags=["tickets"],
    operation_id="create_ticket",
    summary="Create a ticket",
    description="Creates an `open`, unassigned ticket. The requester must be an existing, non-inactive employee.",
    responses={**INVALID},
)
def create_ticket(body: TicketCreate, db: DB):
    return service.create_ticket(db, **body.model_dump())

@router.patch(
    "/tickets/{ticket_id}",
    response_model=TicketOut,
    tags=["tickets"],
    operation_id="update_ticket",
    summary="Update status, priority or assignee",
    description="Only the fields you send are changed. Send `assignee_id: null` to un-assign. "
                "A closed ticket can only be reopened (`status: open`).",
    responses={**NOT_FOUND, **CONFLICT, **INVALID},
)
def update_ticket(ticket_id: Annotated[int, Path(ge=1)], body: TicketUpdate, db: DB):
    return service.update_ticket(db, ticket_id, body.model_dump(exclude_unset=True))

# ------------------------------------------------------------------ orders
@router.get("/orders", response_model=list[OrderOut], tags=["orders"], operation_id="list_orders",
            summary="List orders (newest first)")
def list_orders(
    db: DB,
    status: OrderStatus | None = None,
    customer: Annotated[str | None, Query(description="Part of the customer name")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
):
    return service.list_orders(db, status, customer, limit)

@router.get(
    "/orders/{order_id}",
    response_model=OrderDetail,
    tags=["orders"],
    operation_id="get_order",
    summary="Get one order with its history",
    responses={**NOT_FOUND},
)
def get_order(order_id: Annotated[int, Path(ge=1)], db: DB):
    return service.get_order(db, order_id)

@router.patch(
    "/orders/{order_id}/status",
    response_model=OrderDetail,
    tags=["orders"],
    operation_id="update_order_status",
    summary="Move an order to a new status",
    description="Enforces the order lifecycle (see the API description). Every change is recorded in the "
                "order history, optionally with a note and the employee who made it. "
                "A request that breaks the lifecycle returns 409 and lists the allowed next statuses.",
    responses={**NOT_FOUND, **CONFLICT, **INVALID},
)
def update_order_status(order_id: Annotated[int, Path(ge=1)], body: OrderStatusUpdate, db: DB):
    return service.update_order_status(db, order_id, body.status, body.note, body.changed_by)

@router.post(
    "/orders/{order_id}/cancel",
    response_model=OrderDetail,
    tags=["orders"],
    operation_id="cancel_order",
    summary="Cancel an order",
    description="Shortcut for `update_order_status` with `status: cancelled`. Only `pending` and "
                "`processing` orders can be cancelled. The body is optional.",
    responses={**NOT_FOUND, **CONFLICT},
)
def cancel_order(order_id: Annotated[int, Path(ge=1)], db: DB, body: Annotated[OrderCancel | None, Body()] = None):
    body = body or OrderCancel()
    return service.cancel_order(db, order_id, body.note, body.changed_by)


def getBackendMockRouter() -> APIRouter:
    return router


