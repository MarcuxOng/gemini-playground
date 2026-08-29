"""Tests for the LangChain-path regional -> global endpoint fallback (finding F9).

The raw `genai` path retries a 404 on the global endpoint; before this, the
LangChain path pinned itself to `settings.gcp_region` with no equivalent, so a
global-only model worked on one path and 404'd on the other.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from google.api_core.exceptions import NotFound
from langchain_core.runnables import Runnable, RunnableWithFallbacks

from app.config import settings
from app.services.llm import build_llm_with_region_fallback


class FakeModel(Runnable):  # type: ignore[type-arg]
    """Minimal real Runnable — RunnableWithFallbacks pydantic-validates its members."""

    def __init__(
        self, location: str | None, result: str = "", error: BaseException | None = None
    ) -> None:
        self.location = location
        self.result = result
        self.error = error
        self.calls = 0

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_composes_regional_primary_with_global_fallback():
    regional = FakeModel(settings.gcp_region)
    global_model = FakeModel("global")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]) as mock_build:
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    assert isinstance(composed, RunnableWithFallbacks)
    # The second build must explicitly ask for the global endpoint.
    assert mock_build.call_args_list[1].kwargs["location"] == "global"


def test_falls_back_to_global_when_regional_raises_not_found():
    regional = FakeModel(settings.gcp_region, error=NotFound("model not found in region"))
    global_model = FakeModel("global", result="answer from global")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]):
        composed = build_llm_with_region_fallback("gemini-embedding-2")

    assert composed.invoke("hi") == "answer from global"
    assert regional.calls == 1
    assert global_model.calls == 1


def test_does_not_fall_back_on_non_404_errors():
    regional = FakeModel(settings.gcp_region, error=RuntimeError("quota exhausted"))
    global_model = FakeModel("global", result="should not be reached")

    with patch("app.services.llm.build_llm", side_effect=[regional, global_model]):
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    with pytest.raises(RuntimeError, match="quota exhausted"):
        composed.invoke("hi")

    assert global_model.calls == 0


def test_ai_studio_path_returns_bare_model():
    """The dev path has no regions, so there is nothing to fall back to."""
    dev_model = FakeModel(None)

    with patch("app.services.llm.build_llm", side_effect=[dev_model]) as mock_build:
        composed = build_llm_with_region_fallback("gemini-2.5-flash")

    assert composed is dev_model
    assert mock_build.call_count == 1
