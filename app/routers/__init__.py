from __future__ import annotations

from app.routers.agents import router as agents_router
from app.routers.auth import router as auth_router
from app.routers.caches import router as caches_router
from app.routers.evals import router as evals_router
from app.routers.files import router as files_router
from app.routers.gemini import router as gemini_router
from app.routers.imagen import router as imagen_router
from app.routers.live import router as live_router
from app.routers.mcp_server import router as mcp_server_router
from app.routers.multi_agent import router as multi_agent_router
from app.routers.rag import router as rag_router
from app.routers.threads import router as threads_router

all_routers = [
    auth_router,
    threads_router,
    files_router,
    caches_router,
    gemini_router,
    agents_router,
    multi_agent_router,
    rag_router,
    imagen_router,
    live_router,
    evals_router,
    mcp_server_router,
]
