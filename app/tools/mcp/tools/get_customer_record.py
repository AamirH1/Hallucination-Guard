"""MCP tool: fetch a fictional customer account record.

Backed by a small in-memory mock dataset (no real CRM exists in this sandbox); used to
exercise access-control/audit/data-leakage behavior for records that must never be
fabricated or leaked across sessions.
"""
from __future__ import annotations

from app.tools.mcp.schemas import GetCustomerRecordInput

_MOCK_CUSTOMERS: dict[str, dict] = {
    "cust_1001": {"customer_id": "cust_1001", "plan": "Business", "status": "active", "seats": 12},
    "cust_1002": {"customer_id": "cust_1002", "plan": "Team", "status": "active", "seats": 5},
    "cust_1003": {"customer_id": "cust_1003", "plan": "Enterprise", "status": "past_due", "seats": 400},
}


async def get_customer_record(payload: GetCustomerRecordInput) -> dict:
    record = _MOCK_CUSTOMERS.get(payload.customer_id)
    if record is None:
        return {"found": False, "customer_id": payload.customer_id}
    return {"found": True, **record}
