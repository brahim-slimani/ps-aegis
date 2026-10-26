"""Talk to the agent in the terminal:   python -m app.agent.chat [--trace] [--role manager] [--user-id 6]

Needs the Phase of ingestion done and the mock backend running (uvicorn app.backend.main:app --port 8000).
Conversations are saved in MEMORY_DB like any other, so you can reopen one with --conversation-id.
When the agent prepares a write it asks "Confirm? [y/N]" and sends your answer as the confirm flag.
--trace prints what each node decided: the plan, the tool results and the retrieved passages.
"""
from __future__ import annotations

import argparse
import json
import uuid

from app.config import get_settings

from app.agent.factory import build_agent_service
from app.agent.runner import AgentService, AgentTimeout, TurnLimitReached
from app.agent.schemas import AgentRequest, AgentResponse


def _print_trace(service: AgentService, conversation_id: str) -> None:
    values = service.graph.get_state({"configurable": {"thread_id": conversation_id}}).values or {}
    plan = values.get("plan") or {}
    print(f"  · plan: intent={plan.get('intent')} more={plan.get('more')} rounds={values.get('iteration')}")
    for s in plan.get("steps", []):
        print(f"    - {s['tool']}({json.dumps(s['args'])})")
    for o in values.get("observations", []):
        status = "ok" if o.get("ok") else f"FAILED: {o.get('error')}"
        print(f"  · {o['tool']}: {status}")
    for c in values.get("retrieved_context", []):
        print(f"  · [{c['n']}] {c['source']} > {c['section']} (score {c['score']})")


def _show(resp: AgentResponse) -> None:
    print(f"\nagent> {resp.answer}")
    print(f"  ({resp.turns_used} message(s) used, {resp.turns_remaining} left)")


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with the enterprise agent")
    p.add_argument("--trace", action="store_true", help="show plan, tool results and retrieved passages")
    p.add_argument("--role", default="employee", choices=["employee", "support_agent", "manager", "admin"])
    p.add_argument("--user-id", type=int, default=1, help="employee id of the person talking (used for new tickets)")
    p.add_argument("--conversation-id", default=None, help="continue a saved conversation (default: a new one)")
    args = p.parse_args()

    service = build_agent_service(get_settings())
    cid = args.conversation_id or uuid.uuid4().hex
    print(f"conversation {cid}  |  role={args.role} user={args.user_id}  |  Ctrl-C to quit")
    ident = {"role": args.role, "user_id": args.user_id, "conversation_id": cid}
    try:
        while True:
            text = input("\nyou> ").strip()
            if not text:
                continue
            try:
                resp = service.run(AgentRequest(message=text, **ident))
                _show(resp)
                while resp.pending_confirmation:
                    answer = input("Confirm? [y/N] ").strip().lower()
                    resp = service.run(AgentRequest(confirm=answer in ("y", "yes"),
                                                    action_id=resp.pending_confirmation.action_id, **ident))
                    _show(resp)
            except (TurnLimitReached, AgentTimeout) as e:
                print(f"\n! {e}")
                continue
            if args.trace:
                _print_trace(service, cid)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        service.close()


if __name__ == "__main__":
    main()