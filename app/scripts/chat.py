"""Talk to the agent in the terminal:   python -m app.agent.chat [--trace] [--role manager] [--user-id 2]

Needs the Phase 1 ingestion done and the mock backend running (uvicorn app.backend.main:app --port 8001).
--trace prints what each node decided: the plan, the tool results and the retrieved passages.
"""
from __future__ import annotations

import argparse
import json
import uuid

from app.config import get_settings

from app.agent.factory import build_agent
from app.agent.state import new_turn


def _print_trace(result: dict) -> None:
    plan = result.get("plan") or {}
    print(f"  · plan: intent={plan.get('intent')} more={plan.get('more')} rounds={result.get('iteration')}")
    for s in plan.get("steps", []):
        print(f"    - {s['tool']}({json.dumps(s['args'])})")
    for o in result.get("observations", []):
        status = "ok" if o.get("ok") else f"FAILED: {o.get('error')}"
        print(f"  · {o['tool']}: {status}")
    for c in result.get("retrieved_context", []):
        print(f"  · [{c['n']}] {c['source']} > {c['section']} (score {c['score']})")
    if result.get("pending_action"):
        print(f"  · pending write: {json.dumps(result['pending_action'])}")


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with the enterprise agent")
    p.add_argument("--trace", action="store_true", help="show plan, tool results and retrieved passages")
    p.add_argument("--role", default="employee", choices=["employee", "support_agent", "manager", "admin"])
    p.add_argument("--user-id", type=int, default=1, help="employee id of the person talking (used for new tickets)")
    args = p.parse_args()

    graph, deps = build_agent(get_settings())
    print(f"LLM: {'on' if deps.llm else 'off (regex planner, extractive answers)'}  |  Ctrl-C to quit")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        while True:
            text = input("\nyou> ").strip()
            if not text:
                continue
            result = graph.invoke(new_turn(text, user_id=args.user_id, role=args.role), config)
            print(f"\nagent> {result['answer']}")
            if args.trace:
                _print_trace(result)
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()