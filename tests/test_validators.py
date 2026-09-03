"""Model-name allowlist — guards the T14 prefix removal.

`_ALLOWED_MODEL_PREFIXES` had no test at all, so nothing would have caught the
`image-` / `nano-banana-` prefixes being dropped, or quietly re-added later.
"""

import pytest
from pydantic import TypeAdapter, ValidationError

from app.utils.validators import ModelName


def _validate(name: str) -> str:
    return TypeAdapter(ModelName).validate_python(name)


@pytest.mark.parametrize("name", ["gemini-3.5-flash", "veo-3", "text-embedding-004"])
def test_allows_live_model_families(name: str):
    assert _validate(name) == name


@pytest.mark.parametrize("name", ["image-001", "nano-banana-pro"])
def test_rejects_prefixes_left_over_from_the_pruned_image_feature(name: str):
    """Image generation was cut on 2026-08-29; T14 removed the prefixes it left behind."""
    with pytest.raises(ValidationError):
        _validate(name)


def test_rejects_a_bare_prefix_with_nothing_after_it():
    with pytest.raises(ValidationError):
        _validate("gemini-")
