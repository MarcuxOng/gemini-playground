"""
Code Runner tool — executes code in multiple languages using the Wandbox API (sandboxed).
"""

from __future__ import annotations

import logging
import httpx

from app.config import settings
from app.tools import register

logger = logging.getLogger(__name__)

# Language to compiler mapping for Wandbox
LANGUAGE_TO_COMPILER = {
    "python": "cpython-3.12.7",
    "python3": "cpython-3.12.7",
    "js": "nodejs-20.17.0",
    "javascript": "nodejs-20.17.0",
    "node": "nodejs-20.17.0",
    "cpp": "gcc-13.2.0",
    "c": "gcc-13.2.0-c",
    "go": "go-1.23.2",
    "rust": "rust-1.80.1",
    "ruby": "ruby-3.4.1",
    "php": "php-8.3.12",
    "java": "openjdk-jdk-22+36",
}


@register
def execute_code(code: str, language: str = "python") -> str:
    """
    Execute code in a sandboxed environment using the Wandbox API.
    Supports Python, JavaScript, C++, Go, Rust, and more.

    :param code: The full code snippet to execute.
    :param language: Programming language. Supported values: 'python', 'javascript', 'cpp', 'c', 'go', 'rust', 'ruby', 'php', 'java'. 
    Defaults to 'python'
    """
    if not settings.enable_execute_code:
        return "Error: Code execution is disabled by server configuration."

    lang_lower = language.lower()
    compiler = LANGUAGE_TO_COMPILER.get(lang_lower)

    if not compiler:
        return f"Error: Language '{language}' is not supported. Supported: {list(LANGUAGE_TO_COMPILER.keys())}"

    logger.info(f"Executing {language} code via Wandbox API (compiler: {compiler})...")

    payload = {
        "compiler": compiler,
        "code": code,
        "save": False
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(settings.wandbox_base_url, json=payload)
            response.raise_for_status()
            result = response.json()

        # Wandbox returns status, program_output, program_error, etc.
        status = result.get("status")
        stdout = result.get("program_output", "")
        stderr = result.get("program_error", "")
        
        # Also check for compiler errors/output for compiled languages
        compiler_output = result.get("compiler_output", "")
        compiler_error = result.get("compiler_error", "")

        output_parts = []
        
        if compiler_output or compiler_error:
            output_parts.append("--- COMPILER ---")
            if compiler_output:
                output_parts.append(compiler_output)
            if compiler_error:
                output_parts.append(compiler_error)

        if stdout or stderr:
            output_parts.append("--- PROGRAM ---")
            if stdout:
                output_parts.append(stdout)
            if stderr:
                output_parts.append(stderr)

        if not output_parts:
            if status == "0":
                return "Code executed successfully with no output."
            else:
                return f"Code execution finished with status {status} and no output."

        return "\n".join(output_parts)

    except httpx.HTTPStatusError as e:
        logger.error(f"Wandbox API error: {e.response.text}")
        return f"Error from code execution service: {e.response.status_code}"
    except httpx.RequestError as e:
        logger.error(f"Wandbox API request error: {e}")
        return f"Could not reach code execution service: {e!s}"
    except Exception as e:
        logger.error(f"Unexpected error in execute_code: {e}")
        return f"Error during execution: {e!s}"
