from __future__ import annotations

import json
import logging
import os


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        })


class AlignedFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # Save original levelname to restore it later
        orig_levelname = record.levelname
        # Pad levelname + colon to 10 characters (e.g., "INFO:     ")
        record.levelname = f"{orig_levelname}:".ljust(10)
        result = super().format(record)
        record.levelname = orig_levelname
        return result


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO")
    handler = logging.StreamHandler()
    if os.getenv("ENV") == "production":
        handler.setFormatter(JsonFormatter())
    else:
        # Default local development format with aligned levels
        # No space between levelname and name because levelname is already padded
        formatter = AlignedFormatter(
            "%(levelname)s%(name)s - %(message)s"
        )
        handler.setFormatter(formatter)

    logging.basicConfig(
        level=level, 
        handlers=[handler],
        force=True
    )