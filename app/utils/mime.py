from __future__ import annotations

import magic

# Allowed MIME types for file uploads (Gemini Files API supported types).
# Prefixes ending in '/' match any subtype (e.g. 'image/' matches 'image/jpeg').
ALLOWED_MIME_PREFIXES: frozenset[str] = frozenset(
    ["image/", "audio/", "video/", "application/pdf", "text/plain"]
)


def _mime_allowed(mime_type: str) -> bool:
    return any(
        mime_type.startswith(p) if p.endswith("/") else mime_type == p
        for p in ALLOWED_MIME_PREFIXES
    )


def validate_upload(content: bytes, declared_mime: str) -> None:
    """Raise ValueError if the upload should be rejected.

    Detects the true MIME type from file bytes via libmagic and cross-checks
    against the declared type and the allowed-types allowlist.
    """
    detected = magic.from_buffer(content, mime=True)

    if not _mime_allowed(detected):
        raise ValueError(f"Unsupported file type: {detected!r}")

    # Cross-check: declared and detected type families must agree.
    declared_family = declared_mime.split("/")[0]
    detected_family = detected.split("/")[0]
    if declared_family != detected_family:
        raise ValueError(
            f"File content ({detected!r}) does not match declared type ({declared_mime!r})"
        )
