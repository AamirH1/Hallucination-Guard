"""Audit logging for every MCP tool invocation, persisted via the state store."""
from __future__ import annotations

import datetime as dt

from app.security.sanitization import redact_secrets
from app.state.base import StateStore

AUDIT_LOG_NAME = "mcp_audit_log"

_REDACT_KEYS = {"customer_id"}


def _redact_args(arguments: dict) -> dict:
    redacted = {}
    for k, v in arguments.items():
        if k in _REDACT_KEYS and isinstance(v, str):
            redacted[k] = v[:2] + "***" if len(v) > 2 else "***"
        elif isinstance(v, str):
            redacted[k] = redact_secrets(v)
        else:
            redacted[k] = v
    return redacted


async def record_audit_entry(
    state_store: StateStore,
    session_id: str,
    tool_name: str,
    arguments: dict,
    result_summary: str,
    success: bool,
) -> None:
    entry = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session_id": session_id,
        "tool": tool_name,
        "arguments": _redact_args(arguments),
        "result_summary": result_summary[:300],
        "success": success,
    }
    await state_store.append_log(AUDIT_LOG_NAME, entry)
