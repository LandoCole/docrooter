"""An append-only JSONL log of LLM decisions, with the provenance each one needs.

Each line is one decision: the input level and view hash, the options offered, the chosen
label and probabilities, the model, endpoint host and prompt hash, the tokens spent, a
``trainable`` flag and a UTC timestamp. Whether a row may train a model is for the provider's
terms to decide; Jev rows never may (TypeSafe MCA §2.3(b)), so any row whose model, backend
or endpoint names Jev or TypeSafe is written with ``trainable: false``.

The log never holds API keys. Before a row is written:
- every URL, in any string, is cut to its host, which drops the scheme, credentials, path,
  query and fragment;
- fields named like URLs, endpoints or hosts are cut to the host even without a scheme;
- fields named like secrets (``*key*``, ``*token*``, ``*secret*``, ``*password*``, ``*auth*``,
  ``*credential*``) are replaced with ``***``, at any depth, except the numeric token counts.
  The label fields (``options``, ``label``, ``probs``) keep their own keys, which are labels,
  but a mapping nested inside them is named data again;
- in free text, the value after a secret-named key (``api_key=...``, ``"token": "..."``,
  ``password: ...``) is replaced with ``***`` (a number, such as a token count, is kept);
- the keys in the environment and the ``secrets`` given (pass :func:`secrets_of` of the
  backend), when at least :data:`MIN_SECRET_LENGTH` characters long, and any string shaped like
  a key are replaced with ``***``. A shorter literal is not scrubbed: replacing it everywhere
  would corrupt the log and hide nothing a key of that length could protect.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from docrooter.backends._secrets import REDACTED, redact

ROW_FIELDS: tuple[str, ...] = (
    "level",
    "view_hash",
    "options",
    "label",
    "probs",
    "model",
    "endpoint_host",
    "prompt_hash",
    "tokens",
    "trainable",
    "timestamp",
)

MIN_SECRET_LENGTH = 8
"""The shortest literal secret the log scrubs wherever it appears."""

# Environment variables whose values (API keys) must never reach the log.
KEY_ENV_VARS: tuple[str, ...] = (
    "DOCROOTER_LLM_API_KEY",
    "OPENAI_API_KEY",
    "TYPESAFE_API_KEY",
    "HF_TOKEN",
)

# Fields whose provenance decides whether the row comes from Jev.
_PROVENANCE_FIELDS = ("model", "backend", "spec", "endpoint", "endpoint_host")
_UNTRAINABLE = re.compile(r"jev|typesafe", re.IGNORECASE)
_SECRET_FIELD = re.compile(
    r"key|token|secret|passw|auth|credential|cookie|signature|bearer", re.IGNORECASE
)
# Fields named like secrets that hold token *counts*: kept when their value is not a string.
TOKEN_COUNT_FIELDS = frozenset(
    {"tokens", "prompt_tokens", "completion_tokens", "total_tokens", "max_tokens"}
)
# Fields whose mapping keys are labels, not field names: name rules do not apply inside them.
LABEL_FIELDS = frozenset({"options", "label", "probs"})
_URL_FIELD = re.compile(r"url|endpoint|host", re.IGNORECASE)
_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s\"'<>]+", re.IGNORECASE)
# "sk-..." style keys and bearer credentials.
_KEY_SHAPED = re.compile(r"\b[sprt]k[-_][A-Za-z0-9_\-]{12,}|\bBearer\s+\S+", re.IGNORECASE)
# Long runs that mix lower case, upper case and digits look like random keys; hex hashes
# (lower case only) do not.
_LONG_RUN = re.compile(r"[A-Za-z0-9_\-]{32,}")
# "name=value" or "name: value" in free text, the name secret-like ("author" is not "auth").
_KEY_VALUE = re.compile(
    r"""(?P<name>[\w.\-]*(?:key|token|secret|passw|auth(?!or)|credential|cookie|signature)"""
    r"""[\w.\-]*["']?\s*[:=]\s*["']?)(?P<value>[^\s"'&,;}\]]+)""",
    re.IGNORECASE,
)


def text_hash(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8): the view and prompt hashes of a row."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_random(run: str) -> bool:
    return (
        any(c.islower() for c in run)
        and any(c.isupper() for c in run)
        and any(c.isdigit() for c in run)
    )


# One lock per log file, shared by every ResultsLog in the process.
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(str(path.resolve()), threading.Lock())


def secrets_of(backend: object) -> tuple[str, ...]:
    """The literal secrets a log must scrub for ``backend``: its API key, if it has one."""
    values = getattr(backend, "secret_values", None)
    if callable(values):
        return tuple(v for v in values() if isinstance(v, str) and v.strip())
    key = getattr(backend, "_api_key", None)
    return (key,) if isinstance(key, str) and key.strip() else ()


class ResultsLog:
    """Append one JSON object per :meth:`write` call to ``path`` (parents are created).

    ``secrets`` adds literal keys to scrub beyond those in :data:`KEY_ENV_VARS`. Pass
    ``secrets_of(backend)`` so a key given as ``api_key=`` is scrubbed too. Each row is written
    as one ``O_APPEND`` write under a per-file lock, so rows from several writers never
    interleave.
    """

    def __init__(self, path: str | Path, *, secrets: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self._secrets = [s for s in secrets if _scrubbable(s)]

    def write(self, record: Mapping[str, Any]) -> None:
        """Append ``record`` as one row: redacted, with every field of :data:`ROW_FIELDS`.

        ``trainable`` must be a bool when given. Jev and TypeSafe rows are always written
        untrainable.
        """
        row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
        row.update(record)
        trainable = row["trainable"]
        if trainable is not None and not isinstance(trainable, bool):
            raise TypeError(f"trainable must be a bool, got {trainable!r}")
        row["trainable"] = trainable is not False and not _names_jev(row)
        if row.get("timestamp") is None:
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
        env = [v for n in KEY_ENV_VARS if _scrubbable(v := os.environ.get(n, "").strip())]
        secrets = self._secrets + env
        data = json.dumps(_clean(row, secrets), ensure_ascii=False, allow_nan=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _lock_for(self.path):
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, data.encode("utf-8"))
            finally:
                os.close(fd)


def _scrubbable(secret: object) -> bool:
    return isinstance(secret, str) and len(secret.strip()) >= MIN_SECRET_LENGTH


def _names_jev(row: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row.get(name), str) and _UNTRAINABLE.search(row[name])
        for name in _PROVENANCE_FIELDS
    )


def _host(endpoint: str) -> str:
    """The host of an endpoint, with or without a scheme: no path, query or credentials."""
    try:
        host = urlsplit(endpoint if "://" in endpoint else "//" + endpoint).hostname
    except ValueError:  # e.g. a malformed port or IPv6 literal
        host = None
    return host or REDACTED


def _redact_text(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = redact(text, secret)
    text = _URL.sub(lambda m: _host(m.group()), text)
    text = _KEY_SHAPED.sub(REDACTED, text)
    text = _KEY_VALUE.sub(
        lambda m: m.group() if m["value"].isdigit() else m["name"] + REDACTED, text
    )
    return _LONG_RUN.sub(lambda m: REDACTED if _looks_random(m.group()) else m.group(), text)


def _clean(value: Any, secrets: list[str], *, by_name: bool = True) -> Any:
    """``value`` as plain JSON data, with every string redacted.

    With ``by_name``, mapping keys are field names and :func:`_field` applies its name rules.
    Without it (the top level of a label field such as ``probs``), keys are only redacted as
    text, and anything nested below them is cleaned by name again.
    """
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, float):
        return value if math.isfinite(value) else None  # NaN and inf are not JSON
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, np.ndarray):
        return _clean(value.tolist(), secrets, by_name=by_name)
    if isinstance(value, np.generic):
        return _clean(value.item(), secrets)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _clean(dataclasses.asdict(value), secrets, by_name=by_name)
    if hasattr(value, "model_dump"):  # a pydantic model
        return _clean(value.model_dump(), secrets, by_name=by_name)
    if isinstance(value, Mapping):
        return {
            _redact_text(str(k), secrets): (
                _field(str(k), v, secrets) if by_name else _clean(v, secrets)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v, secrets) for v in value]
    raise TypeError(f"cannot write a {type(value).__name__} to the results log")


def _field(name: str, value: Any, secrets: list[str]) -> Any:
    """One field's value, cleaned: secret-named fields hidden, URL-named fields cut to a host."""
    if (
        name not in LABEL_FIELDS
        and _SECRET_FIELD.search(name)
        and not (name in TOKEN_COUNT_FIELDS and not isinstance(value, str))
    ):
        return REDACTED
    if _URL_FIELD.search(name) and isinstance(value, str):
        return _host(value)
    return _clean(value, secrets, by_name=name not in LABEL_FIELDS)
