"""Per-session tool allowlisting for the MCP server."""
from __future__ import annotations

DEFAULT_ALLOWED_TOOLS = frozenset({"search_documents", "get_document", "search_policies", "get_customer_record"})


class ToolAccessDenied(Exception):
    pass


class AccessController:
    """Tracks which tools each session is permitted to call.

    Sessions default to the full tool set; call `restrict()` to scope a session
    down (e.g. for a lower-trust caller), which is what tests/security/test_tool_abuse.py
    exercises.
    """

    def __init__(self) -> None:
        self._allowlists: dict[str, frozenset[str]] = {}

    def restrict(self, session_id: str, allowed_tools: set[str]) -> None:
        self._allowlists[session_id] = frozenset(allowed_tools)

    def allowed_tools(self, session_id: str) -> frozenset[str]:
        return self._allowlists.get(session_id, DEFAULT_ALLOWED_TOOLS)

    def check(self, session_id: str, tool_name: str) -> None:
        if tool_name not in self.allowed_tools(session_id):
            raise ToolAccessDenied(f"session '{session_id}' is not permitted to call tool '{tool_name}'")


_singleton = AccessController()


def get_access_controller() -> AccessController:
    return _singleton
