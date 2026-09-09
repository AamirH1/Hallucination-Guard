"""MCP tool: fetch a single document's full text/metadata by id."""
from __future__ import annotations

from app.retrieval.corpus import DOCUMENT_REGISTRY
from app.tools.mcp.schemas import GetDocumentInput


async def get_document(payload: GetDocumentInput) -> dict:
    entry = DOCUMENT_REGISTRY.get(payload.doc_id)
    if entry is None:
        return {"found": False, "doc_id": payload.doc_id}
    text, metadata = entry
    return {"found": True, "doc_id": payload.doc_id, "text": text, "metadata": metadata}
