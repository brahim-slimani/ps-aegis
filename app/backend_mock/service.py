"""Business logic for the mock enterprise backend (plain Python + sqlite3, no web framework).

Every function takes an open connection and returns plain dicts. Problems are reported with three
exceptions that main.py maps to HTTP codes:

    NotFound  -> 404   the record does not exist
    Conflict  -> 409   the request is valid but not allowed in the record's current state
    Invalid   -> 422   the request refers to something that does not make sense (unknown assignee, ...)
"""
from __future__ import annotations

import sqlite3

TICKET_STATUSES = ("open", "in_progress", "resolved", "closed")
PRIORITIES = ("low", "medium", "high", "urgent")
TICKET_CATEGORIES = ("general", "hardware", "software", "network", "access", "security", "hr")
ORDER_STATUSES = ("pending", "processing", "shipped", "delivered", "cancelled")
EMPLOYEE_STATUSES = ("active", "on_leave", "inactive")

# Order lifecycle. Anything not listed here is refused with a 409.
ORDER_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "pending": ("processing", "cancelled"),
    "processing": ("shipped", "cancelled"),
    "shipped": ("delivered",),
    "delivered": (),
    "cancelled": (),
}
CANCELLABLE = ("pending", "processing")
OPEN_TICKET_STATUSES = ("open", "in_progress")


class NotFound(Exception):
    pass


class Conflict(Exception):
    pass


class Invalid(Exception):
    pass


def _row(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> dict | None:
    r = conn.execute(sql, args).fetchone()
    return dict(r) if r else None


def _rows(conn: sqlite3.Connection, sql: str, args: list | tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, args).fetchall()]


def _choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise Invalid(f"Invalid {name} '{value}'. Allowed: {', '.join(allowed)}")


def _active_employee(conn: sqlite3.Connection, employee_id: int, role: str) -> dict:
    emp = _row(conn, "SELECT * FROM employees WHERE id = ?", (employee_id,))
    if emp is None:
        raise Invalid(f"{role} {employee_id} does not exist")
    if emp["status"] == "inactive":
        raise Invalid(f"{role} {employee_id} ({emp['name']}) is no longer active")
    return emp


# ------------------------------------------------------------------ employees
def list_employees(conn, q: str | None = None, department: str | None = None, status: str | None = None) -> list[dict]:
    sql, args = "SELECT * FROM employees WHERE 1=1", []
    if q:
        sql += " AND (name LIKE ? OR email LIKE ? OR job_title LIKE ?)"
        args += [f"%{q}%"] * 3
    if department:
        sql += " AND department = ? COLLATE NOCASE"
        args.append(department)
    if status:
        _choice("status", status, EMPLOYEE_STATUSES)
        sql += " AND status = ?"
        args.append(status)
    return _rows(conn, sql + " ORDER BY id", args)


def get_employee_info(conn, employee_id: int) -> dict:
    """Employee record plus context: manager, number of direct reports, and ticket workload."""
    emp = _row(conn, "SELECT * FROM employees WHERE id = ?", (employee_id,))
    if emp is None:
        raise NotFound(f"Employee {employee_id} not found")
    manager = None
    if emp["manager_id"] is not None:
        manager = _row(conn, "SELECT id, name, job_title FROM employees WHERE id = ?", (emp["manager_id"],))
    open_marks = ",".join("?" * len(OPEN_TICKET_STATUSES))
    count = lambda col: conn.execute(  # noqa: E731
        f"SELECT COUNT(*) FROM tickets WHERE {col} = ? AND status IN ({open_marks})",
        (employee_id, *OPEN_TICKET_STATUSES),
    ).fetchone()[0]
    return {
        **emp,
        "manager": manager,
        "direct_reports": conn.execute(
            "SELECT COUNT(*) FROM employees WHERE manager_id = ?", (employee_id,)
        ).fetchone()[0],
        "open_tickets_requested": count("requester_id"),
        "open_tickets_assigned": count("assignee_id"),
    }


# ------------------------------------------------------------------ tickets
def list_tickets(conn, status=None, priority=None, category=None, assignee_id=None, requester_id=None,
                 limit: int = 50) -> list[dict]:
    sql, args = "SELECT * FROM tickets WHERE 1=1", []
    for col, val in (("status", status), ("priority", priority), ("category", category),
                     ("assignee_id", assignee_id), ("requester_id", requester_id)):
        if val is not None:
            sql += f" AND {col} = ?"      # col comes from the literal tuple above, never from user input
            args.append(val)
    return _rows(conn, sql + " ORDER BY id DESC LIMIT ?", [*args, limit])


def get_ticket(conn, ticket_id: int) -> dict:
    t = _row(conn, "SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    if t is None:
        raise NotFound(f"Ticket {ticket_id} not found")
    return t


def create_ticket(conn, *, title: str, requester_id: int, description: str = "", priority: str = "medium",
                  category: str = "general") -> dict:
    title = (title or "").strip()
    if len(title) < 3:
        raise Invalid("Ticket title must be at least 3 characters")
    _choice("priority", priority, PRIORITIES)
    _choice("category", category, TICKET_CATEGORIES)
    _active_employee(conn, requester_id, "Requester")
    with conn:
        cur = conn.execute(
            "INSERT INTO tickets (title, description, priority, category, requester_id) VALUES (?,?,?,?,?)",
            (title, description.strip(), priority, category, requester_id),
        )
    return get_ticket(conn, cur.lastrowid)


def update_ticket(conn, ticket_id: int, changes: dict) -> dict:
    """changes may contain: status, priority, assignee_id (None un-assigns)."""
    allowed = {"status", "priority", "assignee_id"}
    unknown = set(changes) - allowed
    if unknown:
        raise Invalid(f"Cannot change: {', '.join(sorted(unknown))}")
    if not changes:
        raise Invalid("Provide at least one of: status, priority, assignee_id")
    ticket = get_ticket(conn, ticket_id)

    if "status" in changes:
        _choice("status", changes["status"], TICKET_STATUSES)
    if "priority" in changes:
        _choice("priority", changes["priority"], PRIORITIES)
    if changes.get("assignee_id") is not None:
        _active_employee(conn, changes["assignee_id"], "Assignee")
    if ticket["status"] == "closed" and changes.get("status", "closed") != "open":
        raise Conflict(f"Ticket {ticket_id} is closed; reopen it (status=open) before changing anything else")

    sets = ", ".join(f"{k} = ?" for k in changes)
    with conn:
        conn.execute(f"UPDATE tickets SET {sets}, updated_at = datetime('now') WHERE id = ?",
                     [*changes.values(), ticket_id])
    return get_ticket(conn, ticket_id)


# ------------------------------------------------------------------ orders
def list_orders(conn, status=None, customer=None, limit: int = 50) -> list[dict]:
    sql, args = "SELECT * FROM orders WHERE 1=1", []
    if status:
        _choice("status", status, ORDER_STATUSES)
        sql += " AND status = ?"
        args.append(status)
    if customer:
        sql += " AND customer_name LIKE ?"
        args.append(f"%{customer}%")
    return _rows(conn, sql + " ORDER BY id DESC LIMIT ?", [*args, limit])


def get_order(conn, order_id: int) -> dict:
    """Order plus its status history and the statuses it may move to next."""
    o = _row(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
    if o is None:
        raise NotFound(f"Order {order_id} not found")
    o["allowed_next_statuses"] = list(ORDER_TRANSITIONS[o["status"]])
    o["history"] = _rows(
        conn,
        "SELECT from_status, to_status, note, changed_by, created_at FROM order_events"
        " WHERE order_id = ? ORDER BY id",
        (order_id,),
    )
    return o


def update_order_status(conn, order_id: int, new_status: str, note: str | None = None,
                        changed_by: int | None = None) -> dict:
    order = _row(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
    if order is None:
        raise NotFound(f"Order {order_id} not found")
    _choice("status", new_status, ORDER_STATUSES)
    current = order["status"]
    if new_status == current:
        raise Conflict(f"Order {order_id} is already '{current}'")
    allowed = ORDER_TRANSITIONS[current]
    if new_status not in allowed:
        hint = f"allowed: {', '.join(allowed)}" if allowed else "this is a final status"
        raise Conflict(f"Order {order_id} is '{current}' and cannot move to '{new_status}' ({hint})")
    if changed_by is not None:
        _active_employee(conn, changed_by, "changed_by employee")

    with conn:  # status change and history entry are committed together or not at all
        cur = conn.execute(
            "UPDATE orders SET status = ?, updated_at = datetime('now') WHERE id = ? AND status = ?",
            (new_status, order_id, current),
        )
        if cur.rowcount != 1:  # someone else changed it between our read and write
            raise Conflict(f"Order {order_id} was modified concurrently; reload and retry")
        conn.execute(
            "INSERT INTO order_events (order_id, from_status, to_status, note, changed_by) VALUES (?,?,?,?,?)",
            (order_id, current, new_status, note, changed_by),
        )
    return get_order(conn, order_id)


def cancel_order(conn, order_id: int, note: str | None = None, changed_by: int | None = None) -> dict:
    order = _row(conn, "SELECT status FROM orders WHERE id = ?", (order_id,))
    if order is None:
        raise NotFound(f"Order {order_id} not found")
    if order["status"] not in CANCELLABLE:
        raise Conflict(f"Order {order_id} is '{order['status']}' and can no longer be cancelled")
    return update_order_status(conn, order_id, "cancelled", note, changed_by)