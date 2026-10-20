"""Deterministic seed data.

    python -m app.backend.seed            # seed if the database is empty
    python -m app.backend.seed --reset    # delete the database file and re-seed

The first 10 rows of each table are stable reference data (tests and docs rely on them);
extra rows below them add variety. Dates are relative to "now" so the data always looks fresh.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from app.config import get_settings

from app.backend_mock.db import connect, init_db

# name, email, department, job_title, manager_id, location, hire_date, status
EMPLOYEES = [
    ("Alice Martin", "alice.martin@ps.be", "Executive", "CEO", None, "Brussels", "2012-03-01", "active"),
    ("Bruno Peeters", "bruno.peeters@ps.be", "IT", "IT Director", 1, "Brussels", "2015-09-14", "active"),
    ("Chloe Janssens", "chloe.janssens@ps.be", "IT", "Support Engineer", 2, "Brussels", "2019-02-04", "active"),
    ("David Claes", "david.claes@ps.be", "IT", "Support Engineer", 2, "Ghent", "2020-06-15", "active"),
    ("Emma Dubois", "emma.dubois@ps.be", "Sales", "Sales Manager", 1, "Brussels", "2016-01-11", "active"),
    ("Farid Benali", "farid.benali@ps.be", "Sales", "Account Executive", 5, "Antwerp", "2021-04-01", "active"),
    ("Greta Willems", "greta.willems@ps.be", "HR", "HR Manager", 1, "Brussels", "2017-05-22", "active"),
    ("Hugo Lambert", "hugo.lambert@ps.be", "Logistics", "Warehouse Lead", 1, "Liege", "2014-11-03", "active"),
    ("Ines Mertens", "ines.mertens@ps.be", "Logistics", "Support Agent", 8, "Liege", "2022-01-10", "active"),
    ("Jonas Vermeulen", "jonas.vermeulen@ps.be", "Finance", "Accountant", 1, "Brussels", "2018-08-27", "active"),
    # --- extra rows ---
    ("Karim Haddad", "karim.haddad@ps.be", "IT", "Systems Administrator", 2, "Brussels", "2021-09-06", "active"),
    ("Lea Fontaine", "lea.fontaine@ps.be", "Sales", "Account Executive", 5, "Remote", "2023-03-13", "on_leave"),
    ("Marc Delvaux", "marc.delvaux@ps.be", "Logistics", "Support Agent", 8, "Liege", "2017-02-20", "inactive"),
]

# title, description, status, priority, category, requester_id, assignee_id, created_days_ago, updated_days_ago
TICKETS = [
    ("VPN keeps disconnecting", "VPN drops every 10 minutes on the home network.", "open", "high", "network", 6, 3, 2, 1),
    ("Laptop battery swollen", "Battery of my laptop is swollen, please replace.", "in_progress", "urgent", "hardware", 5, 4, 3, 1),
    ("MFA locked after phone change", "Cannot sign in, new phone not enrolled.", "open", "high", "access", 9, 3, 1, 1),
    ("Request second monitor", "Need a second monitor for finance reporting.", "open", "low", "hardware", 10, None, 6, 6),
    ("Printer on floor 2 offline", "Printer shows offline for everyone.", "resolved", "medium", "hardware", 7, 4, 9, 7),
    ("Install Figma license", "Need Figma for the marketing campaign.", "closed", "low", "software", 6, 3, 20, 15),
    ("CRM export fails", "Export to CSV gives a 500 error.", "in_progress", "high", "software", 5, 3, 4, 2),
    ("Badge not opening warehouse door", "Badge denied at warehouse door B.", "open", "medium", "access", 9, None, 1, 1),
    ("Email signature template", "Please update the company signature template.", "open", "low", "software", 7, None, 12, 12),
    ("Suspicious phishing email", "Received a phishing email pretending to be HR.", "open", "urgent", "security", 10, 2, 0, 0),
    # --- extra rows ---
    ("Access to finance shared drive", "New accountant needs read access to /finance/reports.", "open", "medium", "access", 10, 3, 2, 2),
    ("Outlook crashes on startup", "Outlook closes right after the splash screen since the last update.", "in_progress", "high", "software", 8, 4, 3, 1),
    ("New joiner laptop setup", "Prepare a laptop and accounts for the new HR assistant starting Monday.", "open", "medium", "hardware", 7, 4, 5, 4),
    ("Wi-Fi unstable in meeting room 3", "Video calls drop several times per hour.", "open", "medium", "network", 5, None, 2, 2),
    ("Reset shared mailbox password", "Support mailbox password was rotated and agents are locked out.", "resolved", "low", "access", 9, 3, 8, 8),
    ("Expense tool shows wrong currency", "Expenses default to USD instead of EUR.", "closed", "low", "software", 10, 3, 25, 18),
]

# customer, item, quantity, unit_price, status, created_days_ago, cancelled_from (only for 'cancelled')
ORDERS = [
    ("Northwind Traders", "Standing desk", 4, 499.00, "pending", 1, None),
    ("Contoso Ltd", "Ergonomic chair", 10, 285.00, "processing", 4, None),
    ("Fabrikam", "Monitor 27in", 6, 299.00, "shipped", 8, None),
    ("Globex", "Laptop stand", 25, 35.00, "delivered", 20, None),
    ("Initech", "Noise-cancelling headset", 12, 199.00, "pending", 0, None),
    ("Umbrella Corp", "Webcam HD", 8, 79.00, "processing", 3, None),
    ("Hooli", "Docking station", 15, 199.00, "shipped", 9, None),
    ("Stark Industries", "Standing desk", 2, 499.00, "cancelled", 11, "pending"),
    ("Wayne Enterprises", "Ergonomic chair", 3, 285.00, "delivered", 25, None),
    ("Soylent Corp", "Monitor 27in", 20, 299.00, "pending", 2, None),
    # --- extra rows ---
    ("Acme Logistics", "Mechanical keyboard", 30, 79.00, "processing", 5, None),
    ("Wonka Industries", "Ergonomic chair", 5, 285.00, "pending", 1, None),
    ("Cyberdyne Systems", "Docking station", 10, 199.00, "delivered", 30, None),
    ("Tyrell Corp", "Webcam HD", 40, 79.00, "shipped", 10, None),
    ("Oscorp", "Laptop stand", 12, 35.00, "cancelled", 14, "processing"),
    ("Gringotts Bank", "Standing desk", 8, 499.00, "processing", 6, None),
]

_PATHS = {
    "pending": ["pending"],
    "processing": ["pending", "processing"],
    "shipped": ["pending", "processing", "shipped"],
    "delivered": ["pending", "processing", "shipped", "delivered"],
}


def _order_path(status: str, cancelled_from: str | None) -> list[str]:
    if status == "cancelled":
        return _PATHS[cancelled_from or "pending"] + ["cancelled"]
    return _PATHS[status]


def seed(conn: sqlite3.Connection) -> bool:
    """Insert seed data if the employees table is empty. Returns True if data was inserted."""
    if conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] > 0:
        return False
    with conn:
        conn.executemany(
            "INSERT INTO employees (name, email, department, job_title, manager_id, location, hire_date, status)"
            " VALUES (?,?,?,?,?,?,?,?)",
            EMPLOYEES,
        )
        for title, desc, status, prio, cat, req, asg, created, updated in TICKETS:
            conn.execute(
                "INSERT INTO tickets (title, description, status, priority, category, requester_id, assignee_id,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?, datetime('now', ?), datetime('now', ?))",
                (title, desc, status, prio, cat, req, asg, f"-{created} days", f"-{updated} days"),
            )
        for customer, item, qty, price, status, created, cancelled_from in ORDERS:
            cur = conn.execute(
                "INSERT INTO orders (customer_name, item, quantity, unit_price, total, status, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?, datetime('now', ?), datetime('now', ?))",
                (customer, item, qty, price, round(qty * price, 2), status,
                 f"-{created} days", f"-{max(created - 1, 0)} days"),
            )
            path = _order_path(status, cancelled_from)
            for i, to_status in enumerate(path):
                conn.execute(
                    "INSERT INTO order_events (order_id, from_status, to_status, note, created_at)"
                    " VALUES (?,?,?,?, datetime('now', ?, ?))",
                    (cur.lastrowid, path[i - 1] if i else None, to_status,
                     "Order received" if i == 0 else None, f"-{created} days", f"+{i * 4} hours"),
                )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the mock enterprise database")
    parser.add_argument("--reset", action="store_true", help="delete the database file first")
    args = parser.parse_args()

    path = get_settings().backend_db
    if args.reset and Path(path).exists():
        Path(path).unlink()
    init_db(path)
    conn = connect(path)
    try:
        print("Seeded" if seed(conn) else "Already seeded (use --reset to start over)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()