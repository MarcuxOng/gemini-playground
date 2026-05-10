from __future__ import annotations

import logging

from app.services.rag import search_documents
from app.tools import register

logger = logging.getLogger(__name__)


@register
def search_knowledge_base(query: str) -> str:
    """
    Search the private knowledge base for relevant information using vector similarity.
    
    :param query: The search query to look up in the knowledge base.
    """
    try:
        logger.info(f"Tool calling RAG search for query: {query}")
        docs = search_documents(query)
        
        if not docs:
            return "No relevant information found in the knowledge base."
            
        formatted_results = []
        for i, doc in enumerate(docs):
            formatted_results.append(f"Result {i+1}:\n{doc.page_content}")
            
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error in search_knowledge_base tool: {e}")
        return f"Error searching knowledge base: {str(e)}"
