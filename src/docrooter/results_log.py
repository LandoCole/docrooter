"""An append-only JSONL log of LLM decisions, with the provenance each one needs.

Each line is one decision: the input level and view hash, the options offered, the chosen
label and probabilities, the model, endpoint host and prompt hash, the tokens spent, a
``trainable`` flag and a UTC timestamp. Whether a row may train a model is for the provider's
terms to decide; Jev rows never may (TypeSafe MCA §2.3(b)), so any row whose model, backend
or endpoint names Jev or TypeSafe is written with ``trainable: false``.

The log never holds API keys: values of key-like fields, the keys in the environment and any
string shaped like a key are replaced with ``***`` before the row is written.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
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
_SECRET_FIELD = re.compile(r"api[_-]?key|authorization|secret|password|bearer", re.IGNORECASE)
# "sk-..." style keys and bearer credentials.
_KEY_SHAPED = re.compile(r"\b[sprt]k[-_][A-Za-z0-9_\-]{12,}|\bBearer\s+\S+", re.IGNORECASE)
# Long runs that mix lower case, upper case and digits look like random keys; hex hashes
# (lower case only) do not.
_LONG_RUN = re.compile(r"[A-Za-z0-9_\-]{32,}")


def text_hash(text: str) -> str:
    """SHA-256 hex digest of ``text`` (UTF-8): the view and prompt hashes of a row."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _looks_random(run: str) -> bool:
    return (
        any(c.islower() for c in run)
        and any(c.isupper() for c in run)
        and any(c.isdigit() for c in run)
    )


class ResultsLog:
    """Append one JSON object per :meth:`write` call to ``path`` (parents are created).

    ``secrets`` adds keys to redact beyond those found in :data:`KEY_ENV_VARS`.
    """

    def __init__(self, path: str | Path, *, secrets: Iterable[str] = ()) -> None:
        self.path = Path(path)
        self._secrets = [s for s in secrets if s and s.strip()]
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        """Append ``record`` as one row: redacted, with every field of :data:`ROW_FIELDS`."""
        row: dict[str, Any] = dict.fromkeys(ROW_FIELDS)
        row.update(record)
        row["trainable"] = row.get("trainable") is not False and not _names_jev(row)
        if row.get("timestamp") is None:
            row["timestamp"] = datetime.now(timezone.utc).isoformat()
        if isinstance(row.get("endpoint_host"), str):
            row["endpoint_host"] = _host(row["endpoint_host"])
        secrets = self._secrets + [v for n in KEY_ENV_VARS if (v := os.environ.get(n, "").strip())]
        line = json.dumps(_clean(row, secrets), ensure_ascii=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _names_jev(row: Mapping[str, Any]) -> bool:
    return any(
        isinstance(row.get(name), str) and _UNTRAINABLE.search(row[name])
        for name in _PROVENANCE_FIELDS
    )


def _host(endpoint: str) -> str:
    """The host of an endpoint URL, so no path, query or credentials reach the log."""
    if "://" not in endpoint:
        return endpoint
    return urlsplit(endpoint).hostname or ""


def _redact_text(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        text = redact(text, secret)
    text = _KEY_SHAPED.sub(REDACTED, text)
    return _LONG_RUN.sub(lambda m: REDACTED if _looks_random(m.group()) else m.group(), text)


def _clean(value: Any, secrets: list[str]) -> Any:
    """``value`` as plain JSON data, with every string redacted."""
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.ndarray):
        return _clean(value.tolist(), secrets)
    if isinstance(value, np.generic):
        return _clean(value.item(), secrets)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _clean(dataclasses.asdict(value), secrets)
    if hasattr(value, "model_dump"):  # a pydantic model
        return _clean(value.model_dump(), secrets)
    if isinstance(value, Mapping):
        return {
            _redact_text(str(k), secrets): (
                REDACTED if _SECRET_FIELD.search(str(k)) else _clean(v, secrets)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v, secrets) for v in value]
    raise TypeError(f"cannot write a {type(value).__name__} to the results log")
