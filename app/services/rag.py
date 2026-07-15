from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from google import genai
from google.genai import types
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from app.config import build_genai_client, settings
from app.database.models import UploadedFile
from app.rag import GeminiEmbeddings, PineconeStore
from app.services.gemini import SAFETY_SETTINGS, SafetyBlockError, _check_safety_block
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


def vectorstore_service(
    owner_id: str | None = None, embedding_model: str | None = None
) -> PineconeStore:
    """
    Helper to create a Pinecone vectorstore scoped to owner_id.

    :param embedding_model: Overrides `settings.gemini_embedding_model` for this call.
    """
    try:
        vectorstore = PineconeStore(
            index_name=settings.pinecone_index_name,
            embedding=GeminiEmbeddings(model=embedding_model),
            api_key=settings.pinecone_api_key,
            namespace=_build_namespace(owner_id),
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        raise


def ingest_service(
    text: str, owner_id: str | None = None, embedding_model: str | None = None
) -> int:
    """
    Split text into chunks and store in the owner's Pinecone namespace.

    :param embedding_model: Overrides `settings.gemini_embedding_model` for this call.
    """
    try:
        # Text Splitting & Embeddings
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.create_documents([text])
        ns = _build_namespace(owner_id)
        vectorstore_service(owner_id, embedding_model).add_documents(docs)
        logger.info(f"Ingested {len(docs)} chunks into namespace: {ns}")

        return len(docs)

    except Exception as e:
        logger.error(f"Error in ingestion: {e}")
        raise


def query_service(
    query: str, model: str, owner_id: str | None = None, embedding_model: str | None = None
) -> str:
    """
    Construct a RAG chain and execute a query against the owner's namespace.

    When multimodal files (images, audio, video, PDFs) are retrieved,
    they are passed as attachments so the model can read them directly.

    :param embedding_model: Overrides `settings.gemini_embedding_model` for this call.
    """
    try:
        docs = search_documents(query, owner_id, embedding_model)

        text_docs = [d for d in docs if not d.metadata.get("gemini_file_uri")]
        file_docs = [d for d in docs if d.metadata.get("gemini_file_uri")]

        context_parts = [d.page_content for d in text_docs]
        for fd in file_docs:
            context_parts.append(
                f"[File: {fd.metadata.get('display_name', 'file')} "
                f"({fd.metadata.get('mime_type', 'unknown')})]"
            )
        context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."

        if not file_docs:
            prompt_text = RAG_PROMPT_TEMPLATE.format(question=query, context=context)
            llm = build_llm(model)
            return str(llm.invoke(prompt_text))

        logger.info(f"RAG query with {len(file_docs)} multimodal file attachment(s)")
        prompt_text = (
            "You are an expert assistant for question-answering tasks. "
            "Answer the question using the attached files and the retrieved context below.\n\n"
            f"Question: {query}\n\nContext: {context}\n\nAnswer:"
        )
        contents: list[Any] = []
        for fd in file_docs:
            uri = str(fd.metadata["gemini_file_uri"])
            mime_type = str(fd.metadata.get("mime_type", "application/octet-stream"))
            contents.append(types.Part.from_uri(file_uri=uri, mime_type=mime_type))
        contents.append(prompt_text)

        # GCS URIs (gs://) require Vertex AI — Gemini API client can't read them.
        # Gemini Files API URIs work with either client.
        has_gcs = any(str(fd.metadata["gemini_file_uri"]).startswith("gs://") for fd in file_docs)
        if has_gcs and settings.gcp_project_id:
            client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_region,
            )
        else:
            client = build_genai_client()

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(safety_settings=SAFETY_SETTINGS),
        )
        _check_safety_block(response, model)
        return str(response.text or "")

    except SafetyBlockError:
        raise
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        raise


def search_documents(
    query: str, owner_id: str | None = None, embedding_model: str | None = None
) -> list[Document]:
    """
    Similarity search scoped to owner_id (or the rag_owner_id context var).

    Returns Documents that may include multimodal file references.
    Documents with `gemini_file_uri` in metadata represent multimodal file results
    that can be passed as attachments to gemini_service.

    :param embedding_model: Overrides `settings.gemini_embedding_model` for this call.
    """
    try:
        output = vectorstore_service(owner_id, embedding_model).as_retriever(search_kwargs={"k": 5})
        return output.invoke(query)

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise


def ingest_file_service(
    file_ids: list[str], db: Session, owner_id: str, embedding_model: str | None = None
) -> int:
    """Generate embeddings from Gemini file URIs and store in Pinecone.

    Looks up UploadedFile records, generates multimodal embeddings
    from their Gemini file URIs, and stores the vectors alongside
    the file URI metadata so queries can retrieve them.

    :param file_ids: List of UploadedFile DB record UUIDs
    :param db: SQLAlchemy session
    :param owner_id: API key ID for namespace isolation
    :param embedding_model: Overrides `settings.gemini_embedding_model` for this call.
    :return: number of files ingested
    """
    try:
        file_records = []
        for fid in file_ids:
            query = db.query(UploadedFile).filter(UploadedFile.id == fid)
            if owner_id != "master":
                query = query.filter(UploadedFile.owner_id == owner_id)
            file_rec = query.first()
            if file_rec:
                file_records.append(file_rec)
            else:
                logger.warning(f"File {fid!r} not found or not owned by {owner_id!r}; skipping")

        if not file_records:
            raise ValueError("No valid file records found for the provided file_ids")

        docs = []
        for fr in file_records:
            display: str = str(fr.display_name or "uploaded file")
            docs.append(
                Document(
                    page_content=display,
                    metadata={
                        "source": "file",
                        "gemini_file_uri": str(fr.gemini_file_uri),
                        "mime_type": str(fr.mime_type),
                        "display_name": str(fr.display_name),
                        "file_id": str(fr.id),
                        "source_type": "multimodal",
                    },
                )
            )

        vectorstore = vectorstore_service(owner_id, embedding_model)
        vectorstore.add_file_documents(docs)

        ns = _build_namespace(owner_id)
        logger.info(f"Ingested {len(docs)} multimodal files into namespace: {ns}")
        return len(docs)

    except Exception as e:
        logger.error(f"Error in multimodal file ingestion: {e}")
        raise
