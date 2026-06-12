from typing import Annotated

from pydantic import AfterValidator

_ALLOWED_MODEL_PREFIXES = (
    "gemini-",
    "imagen-",
    "veo-",
    "text-embedding-",
    "antigravity-",
    "deep-research-",
    "nano-banana-",
    "image-",
    "text-multilingual-",
)


def _validate_model_name(v: str) -> str:
    for p in _ALLOWED_MODEL_PREFIXES:
        if v.startswith(p) and len(v) > len(p):
            return v
    raise ValueError(
        f"Model must start with one of {_ALLOWED_MODEL_PREFIXES} and include characters after the prefix, got: {v!r}"
    )


# Drop-in replacement for `str` on any Pydantic model field that accepts a model name.
ModelName = Annotated[str, AfterValidator(_validate_model_name)]
