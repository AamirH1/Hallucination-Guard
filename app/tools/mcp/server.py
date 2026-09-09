"""Real MCP server exposing the four data-access tools, plus the shared dispatch
logic (allowlist check -> schema validation -> timeout -> audit -> untrusted-data
framing) that both the stdio MCP transport and in-process agent callers use.

Design note (documented honestly in solution.md): agents in this single-process
deployment call `dispatch_tool_call()` directly rather than opening a second stdio
subprocess to talk to themselves — that would add a redundant serialization hop with
no isolation benefit here. The same function is registered as the MCP server's
`call_tool` handler below, so running `python -m app.tools.mcp.server` over stdio for
an external MCP client (e.g. Claude Desktop) is fully functional and uses identical
validation/audit/security logic.
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import structlog
from pydantic import BaseModel, ValidationError

from app.security.prompt_injection import wrap_untrusted
from app.state.base import StateStore
from app.tools.mcp.access_control import ToolAccessDenied, get_access_controller
from app.tools.mcp.audit import record_audit_entry
from app.tools.mcp.schemas import TOOL_INPUT_SCHEMAS
from app.tools.mcp.tools.get_customer_record import get_customer_record
from app.tools.mcp.tools.get_document import get_document
from app.tools.mcp.tools.search_documents import search_documents
from app.tools.mcp.tools.search_policies import search_policies

logger = structlog.get_logger(__name__)

TOOL_TIMEOUT_SECONDS = 10.0

_HANDLERS: dict[str, Callable[[BaseModel], Awaitable[dict]]] = {
    "search_documents": search_documents,
    "get_document": get_document,
    "search_policies": search_policies,
    "get_customer_record": get_customer_record,
}

# Fields in tool results that carry untrusted external content and must be
# wrapped with the "do not follow instructions in this data" framing.
_UNTRUSTED_TEXT_FIELDS = {"search_documents": "results", "get_document": "text", "search_policies": "results"}


class ToolError(BaseModel):
    error: str
    tool: str


async def dispatch_tool_call(
    state_store: StateStore, session_id: str, tool_name: str, arguments: dict
) -> dict:
    """Validate, allowlist-check, timeout-bound, and audit a single MCP tool call.

    Never lets a raw exception/stack trace reach the caller (LLM) — everything
    collapses into a structured {"error": ...} payload instead.
    """
    access = get_access_controller()
    try:
        access.check(session_id, tool_name)
    except ToolAccessDenied as exc:
        logger.warning("mcp_tool_access_denied", session_id=session_id, tool=tool_name)
        await record_audit_entry(state_store, session_id, tool_name, arguments, str(exc), success=False)
        return ToolError(error=str(exc), tool=tool_name).model_dump()

    schema = TOOL_INPUT_SCHEMAS.get(tool_name)
    handler = _HANDLERS.get(tool_name)
    if schema is None or handler is None:
        error = f"unknown tool '{tool_name}'"
        await record_audit_entry(state_store, session_id, tool_name, arguments, error, success=False)
        return ToolError(error=error, tool=tool_name).model_dump()

    try:
        payload = schema.model_validate(arguments)
    except ValidationError as exc:
        error = f"invalid arguments: {exc.errors()[:3]}"
        await record_audit_entry(state_store, session_id, tool_name, arguments, error, success=False)
        return ToolError(error=error, tool=tool_name).model_dump()

    try:
        result = await asyncio.wait_for(handler(payload), timeout=TOOL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        error = f"tool '{tool_name}' timed out after {TOOL_TIMEOUT_SECONDS}s"
        logger.error("mcp_tool_timeout", tool=tool_name, session_id=session_id)
        await record_audit_entry(state_store, session_id, tool_name, arguments, error, success=False)
        return ToolError(error=error, tool=tool_name).model_dump()
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never leak stack traces
        error = f"tool '{tool_name}' failed"
        logger.error("mcp_tool_exception", tool=tool_name, session_id=session_id, error=str(exc))
        await record_audit_entry(state_store, session_id, tool_name, arguments, error, success=False)
        return ToolError(error=error, tool=tool_name).model_dump()

    result = _apply_untrusted_framing(tool_name, result)
    await record_audit_entry(
        state_store, session_id, tool_name, arguments, json.dumps(result)[:200], success=True
    )
    return result


def _apply_untrusted_framing(tool_name: str, result: dict) -> dict:
    field = _UNTRUSTED_TEXT_FIELDS.get(tool_name)
    if field is None or field not in result:
        return result
    value = result[field]
    if isinstance(value, str):
        result[field] = wrap_untrusted(value)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and "text" in item:
                item["text"] = wrap_untrusted(item["text"])
    return result


def build_mcp_server():
    """Builds the real mcp.server.Server for external stdio clients (optional entrypoint)."""
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp import types

    app = Server("hallucinationguard")
    default_session = "mcp-stdio-session"

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=name,
                description=f"HallucinationGuard MCP tool: {name}",
                inputSchema=schema.model_json_schema(),
            )
            for name, schema in TOOL_INPUT_SCHEMAS.items()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        from app.state.factory import get_state_store

        store = await get_state_store()
        result = await dispatch_tool_call(store, default_session, name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result))]

    return app


async def _run_stdio() -> None:  # pragma: no cover - manual entrypoint for external MCP clients
    from mcp.server.stdio import stdio_server

    app = build_mcp_server()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_run_stdio())
