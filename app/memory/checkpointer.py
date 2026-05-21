from __future__ import annotations

import logging
import threading
from contextlib import _GeneratorContextManager
from typing import Any

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import settings

logger = logging.getLogger(__name__)

# Global instances to be reused and kept alive
_CHECKPOINTER: Any = None
_CHECKPOINTER_CTX: (
    _GeneratorContextManager[SqliteSaver] | _GeneratorContextManager[PostgresSaver] | None
) = None
_CHECKPOINTER_LOCK = threading.Lock()


def get_checkpointer() -> Any:
    """
    Returns the appropriate LangGraph checkpointer based on DATABASE_URL.
    Handles the context manager returned by from_conn_string and ensures it stays alive.
    """
    global _CHECKPOINTER, _CHECKPOINTER_CTX
    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    with _CHECKPOINTER_LOCK:
        if _CHECKPOINTER is not None:
            return _CHECKPOINTER

        db_url = settings.database_url
        if db_url is None:
            raise RuntimeError("DATABASE_URL is required to initialise the checkpointer")
        # Clean logging for security
        logged_url = db_url.split("@")[-1] if "@" in db_url else db_url
        logger.info(f"Initializing checkpointer with database: {logged_url}")

        try:
            ctx: _GeneratorContextManager[SqliteSaver] | _GeneratorContextManager[PostgresSaver]
            if db_url.startswith("sqlite"):
                db_path = db_url.replace("sqlite:///", "")
                # SqliteSaver.from_conn_string is a context manager
                ctx = SqliteSaver.from_conn_string(db_path)
                _CHECKPOINTER_CTX = ctx
                _CHECKPOINTER = ctx.__enter__()
            else:
                # PostgresSaver.from_conn_string is also a context manager
                ctx = PostgresSaver.from_conn_string(db_url)
                _CHECKPOINTER_CTX = ctx
                _CHECKPOINTER = ctx.__enter__()

            return _CHECKPOINTER
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            # Reset globals on failure
            _CHECKPOINTER = None
            _CHECKPOINTER_CTX = None
            raise


def close_checkpointer() -> None:
    """Close the checkpointer context manager from FastAPI lifespan shutdown."""
    global _CHECKPOINTER, _CHECKPOINTER_CTX
    with _CHECKPOINTER_LOCK:
        ctx = _CHECKPOINTER_CTX
        try:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        finally:
            _CHECKPOINTER = None
            _CHECKPOINTER_CTX = None
