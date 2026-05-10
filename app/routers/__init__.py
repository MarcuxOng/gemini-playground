from __future__ import annotations

from app.routers.agents import router as agents_router
from app.routers.auth import router as auth_router
from app.routers.gemini import router as gemini_router
from app.routers.mcp_server import router as mcp_server_router
from app.routers.rag import router as rag_router
from app.routers.threads import router as threads_router

all_routers = [
    auth_router,
    threads_router,
    gemini_router,
    agents_router,
    rag_router,
    mcp_server_router,
]