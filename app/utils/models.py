"""Shared base model for request/input schemas.

Provides a model_validator that normalizes empty strings to None on any
optional (union-with-None) field, so clients can send `""` for unused
fields without triggering validation conflicts (e.g. Gemini's
cached_content + tools rejection when an empty cache_id string is passed).
"""

from __future__ import annotations

import types as py_types
from typing import Any, get_args, get_origin

from pydantic import BaseModel, model_validator


def _accepts_none(annotation: Any) -> bool:
    """Return True if the annotation allows None (Optional / T | None)."""
    origin = get_origin(annotation)
    if origin in {py_types.UnionType, type(int | None)}:  # py3.10+ UnionType / Union
        return type(None) in get_args(annotation)
    return False


class BaseRequestModel(BaseModel):
    """Base class for all request input models.

    Automatically converts empty strings to None on optional fields.
    """

    @model_validator(mode="before")
    @classmethod
    def _empty_str_to_none(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for field_name, field_info in cls.model_fields.items():
            if (
                field_name in data
                and data[field_name] == ""
                and _accepts_none(field_info.annotation)
            ):
                data[field_name] = None
        return data
