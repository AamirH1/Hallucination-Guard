import io
import json
import logging

import pytest
import structlog

from app.main import _redact_secrets_processor
from app.security.sanitization import redact_secrets
from app.state.memory_store import MemoryStateStore
from app.tools.mcp.audit import record_audit_entry


def test_redact_secrets_processor_masks_api_key_fields():
    event_dict = {"event": "startup", "openai_api_key": "sk-realsecretvalue12345", "other": "fine"}
    result = _redact_secrets_processor(None, "info", dict(event_dict))
    assert result["openai_api_key"] == "[REDACTED]"
    assert result["other"] == "fine"


def test_redact_secrets_helper_masks_inline_secret_strings():
    text = "here is a key sk-abcdefghijklmnop for you"
    redacted = redact_secrets(text)
    assert "sk-abcdefghijklmnop" not in redacted
    assert "[REDACTED]" in redacted


@pytest.mark.asyncio
async def test_audit_log_redacts_customer_id():
    store = MemoryStateStore()
    await record_audit_entry(store, "sess1", "get_customer_record", {"customer_id": "cust_1001"}, "ok", True)
    entries = await store.read_log("mcp_audit_log")
    assert entries[-1]["arguments"]["customer_id"] != "cust_1001"


@pytest.mark.asyncio
async def test_cross_session_state_isolation():
    store = MemoryStateStore()
    await store.set("session-a", "secret", "a-only-value")
    await store.set("session-b", "secret", "b-only-value")
    assert await store.get("session-a", "secret") == "a-only-value"
    assert await store.get("session-b", "secret") == "b-only-value"
    # session A must never see a key it never wrote under a different session id
    assert "secret" in store.session_keys("session-a")
    a_values = [await store.get("session-a", k) for k in store.session_keys("session-a")]
    assert "b-only-value" not in a_values


def test_api_response_schema_has_no_raw_provider_fields():
    from app.api.schemas import QueryResponse

    fields = set(QueryResponse.model_fields.keys())
    assert not any("key" in f.lower() or "secret" in f.lower() or "prompt" in f.lower() for f in fields)
