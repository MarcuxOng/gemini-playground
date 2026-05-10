from __future__ import annotations

import logging
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any

from app.config import settings
from app.rag import GeminiEmbeddings, PineconeStore
from app.services.llm import build_llm

logger = logging.getLogger(__name__)

# Standard RAG Prompt
RAG_PROMPT_TEMPLATE = """
You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:
"""


def vectorstore_service() -> PineconeStore:
    """
    Helper to create a Pinecone vectorstore instance.
    """
    try:
        embedding = GeminiEmbeddings()
        vectorstore = PineconeStore(
            index_name=settings.pinecone_index_name,
            embedding=embedding,
            api_key=settings.pinecone_api_key,
            namespace=settings.pinecone_namespace
        )
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise


def ingest_service(text: str) -> int:
    """
    Split text into chunks and store in Pinecone using the direct store.
    """
    try:
        # Text Splitting & Embeddings
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        docs = text_splitter.create_documents([text])
        vectorstore_service().add_documents(docs)
        logger.info(f"Ingested {len(docs)} chunks into namespace: {settings.pinecone_namespace}")
        
        return len(docs)
        
    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise


def query_service(query: str, model: str, provider: str = "gemini") -> str:
    """
    Construct a RAG chain with LangChain & Execute a RAG query.
    """
    try:
        # Retriever & LLM
        retriever = vectorstore_service().as_retriever(search_kwargs={"k": 5})
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


def search_documents(query: str) -> list[Document]:
    """
    Helper to perform similarity search and return raw documents.
    """
    try:
        output = vectorstore_service().as_retriever(search_kwargs={"k": 5})
        return output.invoke(query)
        
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise