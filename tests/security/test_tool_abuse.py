import pytest

from app.state.memory_store import MemoryStateStore
from app.tools.mcp.access_control import get_access_controller
from app.tools.mcp.server import dispatch_tool_call


@pytest.mark.asyncio
async def test_session_without_tool_in_allowlist_is_rejected():
    store = MemoryStateStore()
    access = get_access_controller()
    access.restrict("restricted-session", {"search_documents"})  # no get_customer_record

    response = await dispatch_tool_call(
        store, "restricted-session", "get_customer_record", {"customer_id": "cust_1001"}
    )
    assert "error" in response
    assert "not permitted" in response["error"]


@pytest.mark.asyncio
async def test_allowed_tool_still_works_for_restricted_session():
    store = MemoryStateStore()
    access = get_access_controller()
    access.restrict("restricted-session-2", {"search_documents"})

    response = await dispatch_tool_call(
        store, "restricted-session-2", "search_documents", {"query": "refund", "top_k": 2}
    )
    assert "error" not in response


@pytest.mark.asyncio
async def test_invalid_arguments_rejected_not_crashed():
    store = MemoryStateStore()
    response = await dispatch_tool_call(store, "sess-x", "search_documents", {"query": "", "top_k": 999})
    assert "error" in response


@pytest.mark.asyncio
async def test_unknown_tool_name_returns_structured_error():
    store = MemoryStateStore()
    response = await dispatch_tool_call(store, "sess-x", "delete_everything", {})
    assert "error" in response
    assert response["tool"] == "delete_everything"


@pytest.mark.asyncio
async def test_audit_log_records_every_call():
    store = MemoryStateStore()
    await dispatch_tool_call(store, "sess-audit", "search_documents", {"query": "refund", "top_k": 1})
    entries = await store.read_log("mcp_audit_log")
    assert any(e["session_id"] == "sess-audit" and e["tool"] == "search_documents" for e in entries)
