from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()


port = int(os.environ.get("PORT", 8000))
uvicorn.run(
    "app.app:app",
    host="0.0.0.0",
    port=port,
    reload=False,
    log_level="info",
)