"""
Tools registry.

Usage:
    from app.tools import register, get_tools

    @register
    def my_tool(...): ...

    tools = get_tools()          # list of tool dicts (OpenAI/Gemini schema)
    tools = get_tools("my_tool") # fetch a specific tool by name

Tools imports are at the bottom to avoid circular imports. 
Each tool should import `register` from this module and use it as a decorator.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


_REGISTRY: dict[str, dict[str, Any]] = {}  # name → {fn, schema}


def _python_type_to_json(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema type dict."""
    mapping: dict[Any, dict[str, Any]] = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
    }
    return mapping.get(annotation, {"type": "string"})


def register(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that adds a function to the tool registry.

    The function's docstring becomes the tool description.
    Parameters are inferred from type annotations + default values.

    Example:
        @register
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b
    """
    sig = inspect.signature(fn)
    doc = (fn.__doc__ or "").strip()

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else str
        prop = _python_type_to_json(annotation)

        # Pull per-parameter description from docstring (":param name: desc" style)
        for line in doc.splitlines():
            line = line.strip()
            if line.startswith(f":param {name}:"):
                prop["description"] = line.split(":", 2)[-1].strip()
                break

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema = {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": doc.splitlines()[0] if doc else fn.__name__,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }

    _REGISTRY[fn.__name__] = {"fn": fn, "schema": schema}
    return fn


def get_tools(name: str | None = None) -> list[dict[str, Any]] | dict[str, Any] | None:
    """
    Return tool schemas for passing to an LLM provider.

    Args:
        name: If given, return the single schema for that tool (or None).
              If omitted, return all schemas as a list.
    """
    if name:
        entry = _REGISTRY.get(name)
        return entry["schema"] if entry else None
    return [entry["schema"] for entry in _REGISTRY.values()]


def call_tool(name: str, **kwargs: Any) -> Any:
    """Execute a registered tool by name with the given keyword arguments."""
    entry = _REGISTRY.get(name)
    if not entry:
        raise KeyError(f"Tool '{name}' is not registered. Available: {list(_REGISTRY)}")
    return entry["fn"](**kwargs)


# Import submodules to trigger registration decorators
from app.tools.finance import finance_tools # noqa: E402, F401
from app.tools.system import system_tools # noqa: E402, F401
from app.tools.web import web_tools # noqa: E402, F401