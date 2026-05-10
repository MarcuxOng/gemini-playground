"""
File tool — reads and writes local files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from app.tools import register

logger = logging.getLogger(__name__)
WORKSPACE_ROOT = Path.cwd().resolve()


def _resolve_workspace_path(path: str) -> Path:
    candidate = (WORKSPACE_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Path '{path}' is outside workspace root.") from exc
    return candidate


@register
def read_file(path: str, max_chars: int = 4000) -> str:
    """
    Read the content of a local file.
    
    :param path: The path to the file to read (e.g., 'config.json').
    :param max_chars: Maximum characters to return (default 4000).
    """
    try:
        logger.info(f"Reading file: {path}")
        safe_path = _resolve_workspace_path(path)
        if not safe_path.is_file():
            return f"Error: File '{path}' not found."
            
        if max_chars <= 0:
            raise ValueError("max_chars must be positive.")

        with open(safe_path, encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
            
        if len(content) > max_chars:
            return content[:max_chars] + f"\n\n[...Truncated to {max_chars} chars...]"
        return content
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        return f"Error: {str(e)}"


@register
def write_file(path: str, content: str) -> str:
    """
    Write content to a local file.
    
    :param path: The path where to write the file.
    :param content: The text content to write.
    """
    try:
        logger.info(f"Writing file: {path}")
        safe_path = _resolve_workspace_path(path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(str(safe_path)) or ".", exist_ok=True)
        with open(safe_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Successfully wrote {len(content)} characters to '{path}'."
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        return f"Error: {str(e)}"
