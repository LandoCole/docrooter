"""``RouterConfig``: which engine or backend runs each task (spec §7.3)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from docrooter.classify.classifier import DEFAULT_BACKEND
from docrooter.errors import DocrooterError
from docrooter.rooter.profiles import DEFAULT_PROFILE, profile_defaults

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

__all__ = ["RouterConfig"]

ProfileName = Literal["fast", "balanced", "accurate"]


class RouterConfig(BaseModel):
    """The task → engine mapping of one :func:`~docrooter.process` run.

    Every task the caller leaves out takes the ``profile``'s value (``balanced`` by default;
    see :data:`docrooter.rooter.PROFILES`), so ``RouterConfig(profile="accurate")`` is the whole
    ``accurate`` configuration. Unknown keys are errors.

    - ``text``: the PDF text engine (``auto``, ``pymupdf``, ``pdfplumber``, ``both``).
    - ``ocr``: ``none``, ``auto`` (the ``tesseract`` binary if on PATH, else ``ocrmac`` on macOS,
      else ``rapidocr``, else none with a route warning) or an engine name registered with
      :func:`~docrooter.register_ocr_engine`.
    - ``layout``: ``none`` (no elements: the ``fast`` path), ``heuristic`` (the element rules)
      or a layout engine name (``heron``, extra ``layout``, or one registered with
      :func:`~docrooter.register_layout_engine`).
    - ``tables``: ``lines`` (find ruled tables) or ``none`` (skip table finding, for speed).
    - ``doc_type``, ``sections``: a backend spec (``keyword``, ``embedding:…``, ``nli:…``,
      ``openai:…``, ``jev:…``, ``trained:<bundle>``) or ``none``.
    - ``summaries``: ``none``, ``lead`` or ``centroid``.
    - ``embeddings``: ``none`` or an embedding backend spec (``embedding:…``) whose vectors for
      every element go in ``Analysis.embeddings``; needs element mode (a ``layout`` other than
      ``none``).
    - ``escalate``: ``none`` or a backend spec that re-labels results whose confidence is below
      ``escalate_below`` (spec §7.5).
    - ``max_pages``, ``ocr_lang``, ``ocr_dpi``: page cap, OCR language and OCR render DPI
      (72 to 600; a page over ``docrooter.ocr.render.MAX_RENDER_PIXELS`` renders at a lower
      dpi, recorded as ``dpi_capped`` in its route metrics).

    Engine and backend names are checked when :func:`~docrooter.process` resolves them, so
    engines registered after the configuration was built still work.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: ProfileName = DEFAULT_PROFILE
    text: Literal["auto", "pymupdf", "pdfplumber", "both"] = "auto"
    ocr: str = Field("auto", min_length=1)
    layout: str = Field("heuristic", min_length=1)
    tables: Literal["lines", "none"] = "lines"
    doc_type: str = Field(DEFAULT_BACKEND, min_length=1)
    sections: str = Field(DEFAULT_BACKEND, min_length=1)
    summaries: Literal["none", "lead", "centroid"] = "centroid"
    embeddings: str = Field("none", min_length=1)
    escalate: str = Field("none", min_length=1)
    escalate_below: float = Field(0.5, ge=0.0, le=1.0)
    max_pages: int | None = Field(None, ge=1)
    ocr_lang: str = Field("en", min_length=1)
    ocr_dpi: int = Field(200, ge=72, le=600)

    @model_validator(mode="before")
    @classmethod
    def _fill_from_profile(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        name = data.get("profile")
        name = DEFAULT_PROFILE if name is None else name
        if not isinstance(name, str):
            raise ValueError(f"profile must be a string, got {type(name).__name__}")
        return {**profile_defaults(name), **data, "profile": name}

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> RouterConfig:
        """Read a configuration from a ``.toml`` or ``.json`` file of top-level keys (the
        field names above). Keys it leaves out take its ``profile``'s values."""
        return cls.model_validate(_read_file(path))

    @classmethod
    def resolve(
        cls,
        profile: str | None = None,
        file: str | os.PathLike[str] | None = None,
        **overrides: Any,
    ) -> RouterConfig:
        """The configuration from, in order of precedence: ``overrides`` (call arguments),
        then ``file``, then the profile's defaults. The profile is ``profile`` if given, else
        the file's ``profile``, else ``balanced``."""
        data: dict[str, Any] = _read_file(file) if file is not None else {}
        data.update(overrides)
        if profile is not None:
            data["profile"] = profile
        return cls.model_validate(data)


def _read_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """The top-level table of a TOML or JSON configuration file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in (".toml", ".json"):
        raise ValueError(f"router config must be a .toml or .json file, got {path.name!r}")
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8")) if suffix == ".toml" else json.loads(raw)
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:  # TOMLDecodeError is a ValueError
        raise DocrooterError(f"cannot read router config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise DocrooterError(f"router config {path} must hold an object of settings")
    return data
