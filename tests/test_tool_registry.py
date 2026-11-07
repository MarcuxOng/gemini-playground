"""The registry derives each tool's JSON schema from its Python signature.

Every tool module starts with ``from __future__ import annotations`` (PEP 563),
which leaves annotations as *strings*. ``inspect.signature`` returned "int"
rather than ``int``, the type map missed, and every parameter of every tool was
published as ``{"type": "string"}`` — including on GET /api/v1/agents/tools.
"""

from app.tools import get_registry, list_tool_names


def _params(tool_name: str) -> dict:
    return get_registry()[tool_name]["schema"]["function"]["parameters"]["properties"]


def test_int_parameter_is_declared_as_integer():
    assert _params("count_tokens")["max_tokens"]["type"] == "integer"


def test_str_parameter_is_still_a_string():
    assert _params("count_tokens")["text"]["type"] == "string"


def test_no_tool_silently_declares_everything_as_string():
    """A registry where every parameter is a string is the signature of the bug."""
    typed = {
        name
        for name in list_tool_names()
        for prop in _params(name).values()
        if prop["type"] != "string"
    }
    assert typed, "expected at least one non-string parameter across the registry"


def test_required_parameters_exclude_defaults():
    schema = get_registry()["count_tokens"]["schema"]["function"]["parameters"]
    assert "text" in schema["required"]
    assert "max_tokens" not in schema["required"]
