"""Keep API keys out of error messages."""

from __future__ import annotations

REDACTED = "***"


def redact(text: object, secret: str | None) -> str:
    """``str(text)`` with every occurrence of ``secret`` (and its stripped form) replaced."""
    out = str(text)
    if not secret:
        return out
    for variant in (secret.strip(), secret):  # stripped first: deterministic and thorough
        if variant:
            out = out.replace(variant, REDACTED)
    return out
