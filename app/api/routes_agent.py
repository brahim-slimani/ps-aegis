"""The agent's HTTP endpoints. Thin on purpose: validation is in the schemas, behaviour in AgentService.

    POST /agent                              a message, or the decision (confirm + action_id) about a pending write
    GET  /agent/{conversation_id}/history    messages, pending action and turn counters

    200 answer (or blocked=true with the reason)   422 invalid body   429 turn limit   504 timeout
    503 the server was started without an agent
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.config import get_settings
from app.retrieval.fusion import build_retriever
from app.agent.factory import build_agent_service
from app.llm.client import LLM
from app.agent.runner import AgentService, AgentTimeout, TurnLimitReached
from app.agent.schemas import AgentRequest, AgentResponse, ConversationId, HistoryResponse

router = APIRouter(tags=["agent"])

_UNAVAILABLE = {503: {"description": "The server was started without an agent"}}


def _agent(request: Request) -> AgentService:
    #agent = getattr(request.app.state, "agent", None)
    s = get_settings()
    retriever = build_retriever(s)
    llm = LLM(s.llm_api_key, s.llm_model)
    agent = build_agent_service(s, retriever=retriever, llm=llm)
    if agent is None:
        raise HTTPException(status_code=503, detail="The server was started without an agent")
    return agent

router = APIRouter(prefix=get_settings().rag_api_context_path)

@router.post(
    "/agent",
    response_model=AgentResponse,
    summary="Talk to the agent, or decide a pending action",
    description="Writes take two requests. The first only prepares the action and returns "
                "`pending_confirmation.action_id`. The second sends `confirm: true` (run it) or `false` (cancel) "
                "with that id. Typing 'yes' in a message never confirms anything, and a new message instead of a "
                "decision drops the pending action.",
    responses={
        429: {"description": "The conversation used all its messages (MAX_TURNS)"},
        504: {"description": "The request took longer than AGENT_TIMEOUT; check the history before retrying a write"},
        **_UNAVAILABLE,
    },
)
async def post_agent(body: AgentRequest, request: Request):
    agent = _agent(request)
    try:
        return await agent.arun(body)
    except TurnLimitReached as e:
        raise HTTPException(status_code=429, detail=str(e)) from None
    except AgentTimeout as e:
        raise HTTPException(status_code=504, detail=str(e)) from None


@router.get(
    "/agent/{conversation_id}/history",
    response_model=HistoryResponse,
    summary="Messages, pending action and turn counters of a conversation",
    responses={404: {"description": "Unknown conversation"}, **_UNAVAILABLE},
)
def get_history(conversation_id: ConversationId, request: Request):
    history = _agent(request).history(conversation_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Unknown conversation")
    return history

def get_agent_router() -> APIRouter: 
    return router