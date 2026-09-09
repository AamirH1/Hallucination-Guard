"""Input validation/sanitization helpers for API and tool boundaries."""
from __future__ import annotations

import re

_SECRET_KEY_PATTERN = re.compile(r"(sk-[a-zA-Z0-9]{10,}|api[_-]?key\s*[:=]\s*\S+)", re.IGNORECASE)

MAX_QUERY_LENGTH = 2000


class ValidationError(Exception):
    pass


def validate_query(query: str) -> str:
    query = query.strip()
    if not query:
        raise ValidationError("query must not be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"query exceeds max length of {MAX_QUERY_LENGTH}")
    return query


def redact_secrets(text: str) -> str:
    """Best-effort redaction so accidental secret-shaped strings never leak into logs/responses."""
    return _SECRET_KEY_PATTERN.sub("[REDACTED]", text)
