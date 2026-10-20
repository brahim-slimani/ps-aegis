"""Pydantic models = the API contract. They drive request validation and the OpenAPI docs at /docs."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TicketStatus = Literal["open", "in_progress", "resolved", "closed"]
Priority = Literal["low", "medium", "high", "urgent"]
TicketCategory = Literal["general", "hardware", "software", "network", "access", "security", "hr"]
OrderStatus = Literal["pending", "processing", "shipped", "delivered", "cancelled"]
EmployeeStatus = Literal["active", "on_leave", "inactive"]


class ErrorResponse(BaseModel):
    detail: str


# ------------------------------------------------------------------ employees
class EmployeeOut(BaseModel):
    id: int
    name: str
    email: str
    department: str
    job_title: str
    manager_id: int | None
    location: str
    hire_date: str
    status: EmployeeStatus


class ManagerRef(BaseModel):
    id: int
    name: str
    job_title: str


class EmployeeInfo(EmployeeOut):
    manager: ManagerRef | None = Field(None, description="The employee's direct manager, if any")
    direct_reports: int = Field(description="Number of people reporting directly to this employee")
    open_tickets_requested: int = Field(description="Open / in-progress tickets this employee raised")
    open_tickets_assigned: int = Field(description="Open / in-progress tickets assigned to this employee")


# ------------------------------------------------------------------ tickets
class TicketCreate(BaseModel):
    title: str = Field(min_length=3, max_length=200, description="Short summary of the problem")
    description: str = Field("", max_length=4000)
    priority: Priority = "medium"
    category: TicketCategory = "general"
    requester_id: int = Field(ge=1, description="Employee raising the ticket (must exist and be active)")

    model_config = ConfigDict(json_schema_extra={"examples": [{
        "title": "Cannot connect to VPN",
        "description": "Client shows error 809 since this morning.",
        "priority": "high",
        "category": "network",
        "requester_id": 6,
    }]})


class TicketUpdate(BaseModel):
    status: TicketStatus | None = None
    priority: Priority | None = None
    assignee_id: int | None = Field(None, ge=1, description="Employee to assign; send null to un-assign")

    @model_validator(mode="after")
    def _at_least_one_field(self):
        if not self.model_fields_set:
            raise ValueError("Provide at least one of: status, priority, assignee_id")
        return self

    model_config = ConfigDict(json_schema_extra={"examples": [{"status": "in_progress", "assignee_id": 3}]})


class TicketOut(BaseModel):
    id: int
    title: str
    description: str
    status: TicketStatus
    priority: Priority
    category: TicketCategory
    requester_id: int
    assignee_id: int | None
    created_at: str
    updated_at: str


# ------------------------------------------------------------------ orders
class OrderOut(BaseModel):
    id: int
    customer_name: str
    item: str
    quantity: int
    unit_price: float
    total: float
    status: OrderStatus
    created_at: str
    updated_at: str


class OrderEvent(BaseModel):
    from_status: OrderStatus | None
    to_status: OrderStatus
    note: str | None
    changed_by: int | None
    created_at: str


class OrderDetail(OrderOut):
    allowed_next_statuses: list[OrderStatus] = Field(description="Statuses this order may move to right now")
    history: list[OrderEvent] = Field(description="Status changes, oldest first")


class OrderStatusUpdate(BaseModel):
    status: OrderStatus = Field(description="Target status; must be a legal next step (see allowed_next_statuses)")
    note: str | None = Field(None, max_length=500, description="Reason or comment, stored in the order history")
    changed_by: int | None = Field(None, ge=1, description="Employee making the change (audit trail)")

    model_config = ConfigDict(json_schema_extra={"examples": [
        {"status": "shipped", "note": "Handed to carrier, tracking 1Z999AA10123456784", "changed_by": 8}
    ]})


class OrderCancel(BaseModel):
    note: str | None = Field(None, max_length=500)
    changed_by: int | None = Field(None, ge=1)