from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag import GeminiEmbeddings, PineconeStore
from app.services.llm import build_llm

logger = logging.getLogger(__name__)

# Propagates the calling user's ID into the search_knowledge_base tool.
# Set before any agent run; asyncio copies it into executor threads automatically.
rag_owner_id: ContextVar[str | None] = ContextVar("rag_owner_id", default=None)

# Standard RAG Prompt
RAG_PROMPT_TEMPLATE = """
You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""


def _build_namespace(owner_id: str | None = None) -> str:
    """
    Compute a per-user Pinecone namespace, falling back to the global default.
    """
    resolved = owner_id or rag_owner_id.get()
    if resolved:
        return f"{settings.pinecone_namespace}_{resolved}"
    return settings.pinecone_namespace


def vectorstore_service(owner_id: str | None = None) -> PineconeStore:
    """
    Helper to create a Pinecone vectorstore scoped to owner_id.
    """
    try:
        vectorstore = PineconeStore(
            index_name=settings.pinecone_index_name,
            embedding=GeminiEmbeddings(),
            api_key=settings.pinecone_api_key,
            namespace=_build_namespace(owner_id),
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise


def ingest_service(text: str, owner_id: str | None = None) -> int:
    """
    Split text into chunks and store in the owner's Pinecone namespace.
    """
    try:
        # Text Splitting & Embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([text])
        ns = _build_namespace(owner_id)
        vectorstore_service(owner_id).add_documents(docs)
        logger.info(f"Ingested {len(docs)} chunks into namespace: {ns}")

        return len(docs)

    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise


def query_service(
    query: str, model: str, provider: str = "gemini", owner_id: str | None = None
) -> str:
    """
    Construct a RAG chain and execute a query against the owner's namespace.
    """
    try:
        # Retriever & LLM
        retriever = vectorstore_service(owner_id).as_retriever(search_kwargs={"k": 5})
        llm = build_llm(model)
        prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Chain: retrieve -> augment -> generate
        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        chain: Any = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return str(chain.invoke(query))

    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        raise


def search_documents(query: str, owner_id: str | None = None) -> list[Document]:
    """
    Similarity search scoped to owner_id (or the rag_owner_id context var).
    """
    try:
        output = vectorstore_service(owner_id).as_retriever(search_kwargs={"k": 5})
        return output.invoke(query)

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise
