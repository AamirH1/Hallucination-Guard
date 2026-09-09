"""Pydantic input schemas for every MCP tool exposed by this server."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SearchDocumentsInput(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, str] | None = None


class GetDocumentInput(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)


class SearchPoliciesInput(BaseModel):
    topic: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class GetCustomerRecordInput(BaseModel):
    customer_id: str = Field(min_length=1, max_length=100)


TOOL_INPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "search_documents": SearchDocumentsInput,
    "get_document": GetDocumentInput,
    "search_policies": SearchPoliciesInput,
    "get_customer_record": GetCustomerRecordInput,
}
