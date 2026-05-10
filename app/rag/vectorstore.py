from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from langchain_core.documents import Document
from pinecone import Pinecone

logger = logging.getLogger(__name__)


class PineconeStore:
    """
    Simplified VectorStore to interact with Pinecone using the SDK directly.
    """
    def __init__(self, index_name: str, embedding: Any, api_key: str, namespace: str) -> None:
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding = embedding
        self.namespace = namespace

    def add_documents(self, documents: list[Document]) -> None:
        """Add LangChain documents to Pinecone."""
        texts = [doc.page_content for doc in documents]
        metadatas: list[dict[str, Any]] = []
        ingestion_timestamp = str(int(time.time()))

        for i, doc in enumerate(documents):
            meta: dict[str, Any] = dict(doc.metadata) if doc.metadata else {}
            meta["text"] = texts[i]  # Store text for retrieval
            meta["namespace"] = self.namespace
            metadatas.append(meta)
        embeddings = self.embedding.embed_documents(texts)

        vectors: list[dict[str, Any]] = []
        for i, (text, meta, vector) in enumerate(zip(texts, metadatas, embeddings, strict=True)):
            # Extract document identifier from metadata (source, doc_id, or fallback)
            doc_identifier = meta.get('source', meta.get('doc_id', 'unknown'))

            # Build hash input with disambiguating fields
            hash_input = f"{self.namespace}_{doc_identifier}_{i}_{ingestion_timestamp}_{text}"
            content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            vectors.append({
                "id": f"{self.namespace}_{doc_identifier}_{i}_{content_hash}",
                "values": vector,
                "metadata": meta
            })

        self.index.upsert(
            vectors=vectors,  # type: ignore[arg-type]
            namespace=self.namespace
        )

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        """Perform a similarity search."""
        query_embedding = self.embedding.embed_query(query)
        results = self.index.query(
            namespace=self.namespace,
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )

        docs = []
        for res in results.matches:  # type: ignore[union-attr]
            if "text" in res.metadata:
                text = res.metadata.pop("text")
                docs.append(Document(page_content=text, metadata=res.metadata))
        return docs

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None) -> Retriever:
        """Mock LangChain retriever interface."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return Retriever(self, search_kwargs)


class Retriever:
    def __init__(self, store: PineconeStore, kwargs: dict[str, Any]) -> None:
        self.store = store
        self.kwargs = kwargs

    def invoke(self, query: str) -> list[Document]:
        return self.store.similarity_search(query, k=self.kwargs.get("k", 5))

    def __or__(self, other: Any) -> Any:
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(self.invoke) | other