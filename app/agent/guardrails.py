"""Guardrails: input screening, role-based write permissions, action ids and the request time budget.

"""
from __future__ import annotations

import logging
import re
import time
import uuid

# Every staged, executed, failed, denied and blocked action is logged here.
audit = logging.getLogger("agent.audit")

MAX_INPUT_CHARS = 2000

# Deliberately narrow: "show the instructions for resetting my password" must pass.
_INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)"
    r"|(reveal|show|print|repeat)\s+(me\s+)?(your\s+(system\s+prompt|instructions|hidden\s+prompt|rules)"
    r"|the\s+(system|hidden)\s+prompt)"
    r"|you\s+are\s+now\s+(?:dan|an?\s+unrestricted)",
    re.IGNORECASE,
)

# Write tools each role may request. Read tools are open to every known role. A tool missing from a role's
# set is denied, so a newly added write tool is off for everyone except "*" roles until listed here.
WRITE_ALLOW: dict[str, set[str] | str] = {
    "employee": {"create_ticket"},
    "support_agent": {"create_ticket", "update_ticket", "cancel_order"},
    "manager": "*",
    "admin": "*",
}


def check_input(text: str | None) -> tuple[bool, str]:
    """(ok, refusal message). Used as the first node of every message turn."""
    if not text or not text.strip():
        return False, "Please type a question or request."
    if len(text) > MAX_INPUT_CHARS:
        return False, f"Your message is too long (max {MAX_INPUT_CHARS} characters)."
    if _INJECTION.search(text):
        return False, "I can't follow instructions that try to override my rules or reveal my configuration."
    return True, ""


def authorize(role: str, tool: str, is_write: bool) -> tuple[bool, str]:
    """(allowed, reason). Checked when a write is staged and again when it is confirmed."""
    if role not in WRITE_ALLOW:
        return False, f"Unknown role '{role}'."
    if not is_write:
        return True, ""
    allowed = WRITE_ALLOW[role]
    if allowed == "*" or tool in allowed:
        return True, ""
    return False, f"Role '{role}' is not allowed to use '{tool}'."


def new_action_id() -> str:
    """Random 12-hex-character id for a pending action (48 bits; it is not a secret, it prevents mix-ups)."""
    return uuid.uuid4().hex[:12]


def out_of_time(state: dict) -> bool:
    """True once the request's soft deadline has passed: no new tool call or write may start."""
    deadline = state.get("deadline")
    return deadline is not None and time.monotonic() >= deadline