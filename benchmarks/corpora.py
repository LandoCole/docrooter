"""The benchmark PDF corpora: the dev corpus and the frozen held-out corpus.

Every PDF harness takes ``--corpus dev|heldout`` (or a directory path). A corpus directory holds
``manifest.jsonl`` (one record per PDF; the held-out manifest starts with a ``{"_meta": ...}``
record, which is skipped) and the PDFs under the paths its records name. Results of a run on a
corpus other than ``dev`` go to ``<corpus name>-<file>`` (``heldout-pdf.json``), so a held-out run
never overwrites the dev results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCHMARKS = Path(__file__).parent
CORPORA: dict[str, Path] = {
    "dev": BENCHMARKS / "corpus",
    "heldout": BENCHMARKS / "heldout",
}
DEV = CORPORA["dev"]
HELDOUT = CORPORA["heldout"]


def corpus_path(value: str | Path) -> Path:
    """``"dev"`` and ``"heldout"`` name the built-in corpora; anything else is a directory."""
    if isinstance(value, str) and value in CORPORA:
        return CORPORA[value]
    return Path(value)


def is_dev(corpus: Path) -> bool:
    return corpus.resolve() == DEV.resolve()


def relative(corpus: Path) -> str:
    """``corpus`` relative to the repository root (``benchmarks/heldout``), else absolute."""
    root = BENCHMARKS.parent.resolve()
    path = corpus.resolve()
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def results_name(corpus: Path, filename: str) -> str:
    """``filename`` for the dev corpus, ``"<corpus dir name>-<filename>"`` for any other."""
    return filename if is_dev(corpus) else f"{corpus.name}-{filename}"


def read_manifest(corpus: Path) -> list[dict[str, Any]]:
    """The document records of ``corpus/manifest.jsonl``, in file order, without ``_meta``."""
    records = []
    with (corpus / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            if "_meta" in record:
                continue
            records.append(record)
    return records
