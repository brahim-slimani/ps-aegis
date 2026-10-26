"""Request / response models of POST /agent and GET /agent/{conversation_id}/history (pydantic only)."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.agent.state import Role

ConversationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")]
ActionId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{12}$")]


class AgentRequest(BaseModel):
    """Either a message, or the decision about the action that is waiting for confirmation. Never both."""

    message: str | None = Field(None, max_length=10_000, description="What the user says or asks")
    conversation_id: ConversationId | None = Field(
        None, description="Omit on the first request; the response returns the generated id to reuse")
    confirm: bool | None = Field(
        None, description="true runs the pending action, false cancels it. Needs conversation_id and action_id")
    action_id: ActionId | None = Field(None, description="`pending_confirmation.action_id` from the last response")
    role: Role = "employee"
    user_id: int = Field(1, ge=1, description="Employee making the request (used for new tickets)")

    @model_validator(mode="after")
    def _message_xor_decision(self):
        if self.confirm is None:
            if self.action_id is not None:
                raise ValueError("action_id only goes with confirm")
            if self.message is None or not self.message.strip():
                raise ValueError("Send a message, or a decision (confirm) about the pending action")
        else:
            if self.message is not None:
                raise ValueError("Send either a message or a decision (confirm), not both")
            if self.conversation_id is None or self.action_id is None:
                raise ValueError("A decision needs conversation_id and action_id")
        return self

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"message": "cancel order 1", "conversation_id": "c1", "role": "support_agent", "user_id": 2},
        {"conversation_id": "c1", "confirm": True, "action_id": "3f9a1c0b7d2e", "role": "support_agent", "user_id": 2},
    ]})


class PendingConfirmation(BaseModel):
    action_id: str
    tool: str
    args: dict[str, Any]
    summary: str


class ExecutedAction(BaseModel):
    tool: str
    args: dict[str, Any]
    summary: str
    result: Any = None


class Citation(BaseModel):
    n: int
    source: str
    title: str
    section: str
    score: float
    snippet: str


class AgentResponse(BaseModel):
    conversation_id: str
    answer: str
    blocked: bool = Field(False, description="A guardrail refused the message; `answer` says why")
    pending_confirmation: PendingConfirmation | None = Field(
        None, description="A write is prepared but NOT done; decide it with confirm + this action_id")
    executed_actions: list[ExecutedAction] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    turns_used: int
    turns_remaining: int


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    conversation_id: str
    messages: list[HistoryMessage]
    pending_confirmation: PendingConfirmation | None = None
    turns_used: int
    max_turns: int
    turns_remaining: int