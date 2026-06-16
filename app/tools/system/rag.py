from __future__ import annotations

import logging

from app.services.rag import rag_owner_id, search_documents
from app.tools import register

logger = logging.getLogger(__name__)


@register
def search_knowledge_base(query: str) -> str:
    """
    Search the private knowledge base for relevant information using vector similarity.

    Returns both text chunks and references to uploaded multimodal files
    (images, audio, video) that match the query. Multimodal file results
    include the file URI which can be passed as an attachment for grounded answers.

    :param query: The search query to look up in the knowledge base.
    """
    try:
        logger.info(f"Tool calling RAG search for query: {query}")
        docs = search_documents(query, rag_owner_id.get())

        if not docs:
            return "No relevant information found in the knowledge base."

        formatted_results = []
        file_refs: list[str] = []
        for i, doc in enumerate(docs):
            if doc.metadata.get("gemini_file_uri"):
                file_refs.append(
                    f"- File: {doc.metadata.get('display_name', 'file')} "
                    f"({doc.metadata.get('mime_type', 'unknown')})\n"
                    f"  URI: {doc.metadata['gemini_file_uri']}"
                )
            formatted_results.append(f"Result {i + 1}:\n{doc.page_content}")

        output = "\n\n---\n\n".join(formatted_results)
        if file_refs:
            output += "\n\n**Retrieved File Attachments:**\n" + "\n".join(file_refs)
        return output

    except Exception as e:
        logger.error(f"Error in search_knowledge_base tool: {e}")
        return f"Error searching knowledge base: {str(e)}"
