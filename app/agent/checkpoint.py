"""Conversation memory = the LangGraph checkpointer, keyed by thread_id (our `conversation_id`).

MEMORY_DB set    -> SQLite file: conversations survive restarts and can be shared by several workers.
MEMORY_DB empty  -> in-process memory: lost on restart.

If a SQLite file is requested but the saver package is missing we raise instead of quietly falling back
to memory: a server that forgets every conversation on restart should refuse to start, not pretend.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def make_checkpointer(memory_db: str = ""):
    if not memory_db:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as e:
        raise RuntimeError(
            "MEMORY_DB is set but the SQLite checkpointer is not installed. "
            "Run `pip install langgraph-checkpoint-sqlite`, or set MEMORY_DB= (empty) to keep "
            "conversations in process memory only."
        ) from e
    if memory_db != ":memory:":
        Path(memory_db).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: requests run in worker threads. SqliteSaver serialises access with its own lock.
    return SqliteSaver(sqlite3.connect(memory_db, check_same_thread=False))


def count_user_turns(messages: list[dict]) -> int:
    """Number of user messages in a conversation. Confirmations are not messages, so they never count."""
    return sum(1 for m in messages if m.get("role") == "user")