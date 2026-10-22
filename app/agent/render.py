"""Text rendering shared by the planner (re-plan prompt) and the responder (answer prompt / offline answer)."""
from __future__ import annotations

from typing import Any

from app.agent.state import ContextChunk, Observation, PendingAction

_PREFERRED_KEYS = ["id", "title", "name", "customer_name", "item", "status", "priority", "department",
                   "job_title", "email", "total", "assignee_id"]
MAX_LISTED = 10


def fmt_item(d: Any) -> str:
    if not isinstance(d, dict):
        return str(d)
    return ", ".join(f"{k}={d[k]}" for k in _PREFERRED_KEYS if k in d and d[k] is not None)


def render_observations(observations: list[Observation], skip_tools: tuple[str, ...] = ()) -> str:
    lines: list[str] = []
    for o in observations:
        if o["tool"] in skip_tools:
            continue
        if not o.get("ok"):
            lines.append(f"- {o['tool']} failed: {o.get('error')}")
            continue
        r = o.get("result")
        if isinstance(r, list):
            lines.append(f"- {o['tool']}: {len(r)} result(s)")
            lines += [f"    • {fmt_item(x)}" for x in r[:MAX_LISTED]]
            if len(r) > MAX_LISTED:
                lines.append(f"    … and {len(r) - MAX_LISTED} more")
        elif isinstance(r, dict):
            extra = ""
            if r.get("history"):          # order detail: show how it got here
                extra = " | history: " + " > ".join(e["to_status"] for e in r["history"])
            lines.append(f"- {o['tool']}: {fmt_item(r)}{extra}")
        else:
            lines.append(f"- {o['tool']}: {r}")
    return "\n".join(lines)


def render_context(context: list[ContextChunk]) -> str:
    return "\n\n".join(
        f"[{c['n']}] ({c['source']} > {c['section'] or c['title']})\n{c['text']}" for c in context
    )


def confirmation_prompt(pending: PendingAction) -> str:
    return f"I'm about to: {pending['summary']}. Reply 'yes' to confirm or 'no' to cancel."