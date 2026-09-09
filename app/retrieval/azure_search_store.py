"""Azure AI Search vector store adapter.

Implemented against the real `azure-search-documents` SDK for correctness, but not
exercised in this sandbox (no Azure resource available). See solution.md.
"""
from __future__ import annotations

from app.retrieval.vector_store_base import VectorHit


class AzureSearchVectorStore:
    name = "azure_search"

    def __init__(self, endpoint: str, key: str, index_name: str = "documents") -> None:
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents.aio import SearchClient

        self._client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))

    async def upsert(self, doc_id: str, text: str, embedding: list[float], metadata: dict[str, object]) -> None:
        document = {"id": doc_id, "content": text, "content_vector": embedding, **metadata}
        await self._client.merge_or_upload_documents(documents=[document])

    async def query(
        self, embedding: list[float], top_k: int, filters: dict[str, object] | None = None
    ) -> list[VectorHit]:
        from azure.search.documents.models import VectorizedQuery

        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="content_vector")
        results = await self._client.search(search_text=None, vector_queries=[vector_query], top=top_k)
        hits: list[VectorHit] = []
        async for result in results:
            data = dict(result)
            doc_id = data.pop("id")
            text = data.pop("content", "")
            score = data.pop("@search.score", 0.0)
            data.pop("content_vector", None)
            hits.append(VectorHit(doc_id=doc_id, text=text, metadata=data, score=score))
        return hits

    async def delete(self, doc_id: str) -> None:
        await self._client.delete_documents(documents=[{"id": doc_id}])
