# Enterprise RAG + Multi-Agent Assistant

Searchable company knowledge (hybrid RAG) plus an agent that can act on a mock enterprise backend
(tickets, orders, employees). Writes need an explicit confirmation, roles limit who may request what,
and every conversation is remembered in SQLite.

```
data/docs/*.md ──► chunker ──► BGE embedder ──► Qdrant ─┐
                       └──────► data/chunks.jsonl ─► BM25 ┴─► RRF fusion ─► cross-encoder ─► POST /ask
                                                                                   │
POST /agent ─► AgentService (lock, turn limit, timeout) ─► LangGraph, state saved per conversation_id
                 guard ─► gate ─► planner ─► router ─► executor ─► responder          (SQLite checkpointer)
                           │                  │            │
                           │ confirm=true     │ write      └─ tools: hybrid retrieval, mock backend (HTTP)
                           └► execute_pending │ staged as pending_action, waits for the confirm flag
                                              ▼
                                   Mock backend (FastAPI + SQLite, :8001)
```

## Run it

```bash
pip install -r requirements.txt


python -m app.ingestion.ingest --recreate     # phase 1: embeds data/docs into Qdrant (local mode)
uvicorn app.main:app --port 8000          # phases 2, 3: reterival & mock backend + 4, 5: /ask and /agent
python -m app.agent.chat --trace --role admin --user-id 6    # or talk to the agent in the terminal
```

Use a Qdrant server instead of embedded mode: `docker compose up -d` and set `QDRANT_URL=http://localhost:6333`.

## Using the API

```bash
curl -s localhost:8000/rag/v1/ask -H 'content-type: application/json' \
  -d '{"question":"How many days of annual leave do I get?"}'
```

**Writes take two requests.** The first one only prepares the action and returns an `action_id`:

```bash
curl -s localhost:8000/rag/v1/agent -H 'content-type: application/json' \
  -d '{"message":"cancel order 1","conversation_id":"c1","role":"support_agent","user_id":2}'
# -> {"answer": "I'm about to: Cancel order #1. Please confirm ...",
#     "pending_confirmation": {"action_id": "3f9a1c0b7d2e", "tool": "cancel_order", ...}, ...}
```

The second one carries the decision. Typing "yes" in a message never confirms anything:

```bash
curl -s localhost:8000/rag/v1/agent -H 'content-type: application/json' \
  -d '{"conversation_id":"c1","confirm":true,"action_id":"3f9a1c0b7d2e","role":"support_agent","user_id":2}'
# confirm=false cancels instead. A new message instead of a decision drops the pending action.
```

`GET /agent/{conversation_id}/history` returns the messages, the pending action and the turn counters.
Omit `conversation_id` on the first request to start a new conversation; the response contains the generated id.

| Status | Meaning |
|---|---|
| 200 | Answer, or `blocked: true` with the reason when a guardrail refused the message |
| 422 | Invalid body (bad `conversation_id`, unknown role, `confirm` without `conversation_id`/`action_id`, ...) |
| 429 | The conversation used all `MAX_TURNS` messages. Confirmations do not count, so a pending action can still be decided |
| 504 | The request took longer than `AGENT_TIMEOUT` |
| 503 | The server was started without an agent |

## Phase map

| Phase | Files |
|---|---|
| 1 Ingestion & embeddings | `app/ingestion/{chunker,embedder,vector_store,ingest}.py` |
| 2 Hybrid retrieval + `/ask` | `app/retrieval/{bm25,hybrid,reranker,factory}.py`, `app/rag.py`, `app/api/main.py` |
| 3 Mock backend | `app/backend_mock/{db,seed,service,schemas,main}.py` |
| 4 LangGraph | `app/agent/{state,graph,routing,planning,tools,render,deps}.py`, `app/agent/nodes/{planner,router,executor,responder}.py` |
| 5 `/agent`, memory, guardrails | `app/api/agent.py`, `app/agent/{runner,schemas,memory,guardrails,factory,chat}.py`, `app/agent/nodes/{guard,confirm}.py` |

## Design notes

- **Chunking** is heading-aware; each chunk is embedded with its `title > section` prefix. IDs are deterministic, so re-ingesting is idempotent.
- **BGE** query instruction prefix is applied to queries only (skipped for `bge-m3`).
- **Fusion** uses Reciprocal Rank Fusion (k=60) over the vector and BM25 rankings, then the top candidates are rescored by `bge-reranker-base`.
- **No LLM key?** The planner falls back to regex rules and `/ask` returns the best passage, so everything is testable offline.
- **Memory** is the LangGraph checkpointer keyed by `conversation_id` (the `thread_id`). `MEMORY_DB` (default `data/agent_memory.sqlite`) makes it survive restarts and lets several workers share it; `MEMORY_DB=` keeps it in process memory. If the SQLite saver package is missing the server refuses to start rather than silently losing memory.
- **Confirmation** is a structured decision, not a sentence: the router stages a write as `pending_action` with a random 12-character id, and only a request with `confirm: true` and that id executes it. A wrong id is refused and leaves the real action pending. A double click or retry executes once, because requests for one conversation run one at a time and the action is cleared when it runs.
- **Roles** are checked when the write is staged and again when it is confirmed. `employee`: create tickets. `support_agent`: also update tickets and cancel orders. `manager`/`admin`: every write tool. Edit `WRITE_ALLOW` in `app/agent/guardrails.py`.
- **Input screening**: length limit and a prompt-injection regex. Blocked turns get a refusal and are never stored in memory.
- **Limits**: `MAX_TURNS` user messages per conversation (HTTP 429). `AGENT_TIMEOUT` per request (HTTP 504); at 80% of it the graph stops starting new tool calls or writes. A Python thread cannot be killed, so a timed-out request keeps its conversation lock until the worker finishes, and a retry that queued behind it is dropped instead of running late. After a 504, check `/history` before retrying a write.
- **Audit log**: every staged, executed, failed, denied and blocked action is logged on the `agent.audit` logger.