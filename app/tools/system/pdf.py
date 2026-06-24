"""
PDF Text Extractor — extracts raw text from PDFs for LLM processing.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from app.tools import register

logger = logging.getLogger(__name__)
WORKSPACE_ROOT = Path.cwd().resolve()


def _resolve_pdf_path(path: str) -> Path:
    candidate = (
        (WORKSPACE_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    )
    try:
        candidate.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise PermissionError(f"Path '{path}' is outside workspace root.") from exc
    return candidate


@register
def extract_pdf_text(
    path: str,
    start_page: int = 1,
    end_page: int = 0,
    max_chars: int = 10000,
) -> str:
    """
    Extract raw text from a PDF file for summarization or Q&A.

    :param path: The path to the PDF file.
    :param start_page: First page to extract (1-indexed, default 1).
    :param end_page: Last page to extract (0 = all remaining pages).
    :param max_chars: Maximum characters to return (default 10000).
    """
    try:
        logger.info(f"Extracting PDF text: {path}")
        pdf_path = _resolve_pdf_path(path)

        if not pdf_path.is_file():
            return f"Error: File '{path}' not found."
        if pdf_path.suffix.lower() != ".pdf":
            return f"Error: File '{path}' is not a PDF."

        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        if start_page < 1 or start_page > total_pages:
            return f"Error: start_page must be between 1 and {total_pages}."
        if end_page == 0:
            end_page = total_pages
        if end_page < start_page or end_page > total_pages:
            return f"Error: end_page must be between {start_page} and {total_pages}."

        if max_chars <= 0:
            return "Error: max_chars must be positive."

        pages_text: list[str] = []
        char_count = 0
        truncated = False
        last_page_index = start_page - 1

        for i in range(start_page - 1, end_page):
            last_page_index = i
            page_text = reader.pages[i].extract_text() or ""
            remaining = max_chars - char_count
            if len(page_text) > remaining:
                pages_text.append(page_text[:remaining])
                truncated = True
                break
            pages_text.append(page_text)
            char_count += len(page_text)

        extracted = "\n\n".join(pages_text)

        if not extracted.strip():
            return f"Warning: No extractable text found in pages {start_page}–{end_page} of '{path}'. The PDF may be image-based."

        result = f"[PDF: {path} | Pages {start_page}–{last_page_index + 1} of {total_pages}]\n\n{extracted}"

        if truncated:
            result += f"\n\n[...Truncated to {max_chars} chars...]"

        return result

    except PermissionError:
        return f"Error: Path '{path}' is outside workspace root."
    except Exception as e:
        logger.error(f"Error extracting PDF text {path}: {e}")
        return f"Error: {str(e)}"
