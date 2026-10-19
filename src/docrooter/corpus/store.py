"""The corpus store (spec §6): a directory of JSON files holding a document corpus's elements
and the labels given to them.

Layout::

    <dir>/manifest.jsonl       one ManifestRecord per document, in the order they were added
    <dir>/docs/<id>.json       the document's id, sha256, hints, pages read and elements
    <dir>/labels/<name>.jsonl  one LabelRow per line, as the labeller wrote them
    <dir>/stats.json           written by :meth:`Corpus.stats`
    <dir>/attempting.json      the document a build is reading now (see ``start_attempt``)
    <dir>/failed.jsonl         documents a build could not read: id, reason, timestamp

Crash safety (Review Focus 5). A doc file is written to a temporary file in ``docs/`` and moved
into place with ``os.replace``, so it is either whole or absent. Its manifest line is appended
only after that, and the manifest is the only index readers use: a document whose line is
missing (the process died in between) is invisible, and adding it again rewrites the doc file
and appends the line. A line counts once its newline is written. A torn last line, left by a
process that died mid-append, is ignored by readers and cut off by the next append; label files
behave the same way. Appends are flushed and fsync'ed before returning.

Corpus documents are untrusted data: their text is stored and returned, never interpreted.
Document ids and labeller names become file names, so they are restricted to letters, digits,
``.``, ``_`` and ``-`` (starting with a letter or digit).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, NamedTuple, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from docrooter.elements.models import ElementKind
from docrooter.errors import DocrooterError

__all__ = [
    "CALIB_HALVES",
    "SPLITS",
    "Corpus",
    "CorpusElement",
    "LabelKey",
    "LabelRow",
    "ManifestRecord",
    "calib_half",
    "is_calib_crc",
    "read_manifest",
    "text_sha1",
]

Split = Literal["train", "calib", "gold"]
SPLITS: tuple[str, ...] = get_args(Split)
CalibHalf = Literal["fit", "crc"]
CALIB_HALVES: tuple[str, ...] = get_args(CalibHalf)
_CALIB_SALT = "calibsplit:"
_KINDS = frozenset(get_args(ElementKind))
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_ELEMENT_KEYS = frozenset({"id", "kind", "text", "heading_path", "section_title", "page"})


def text_sha1(text: str) -> str:
    """The hex SHA-1 of ``text`` (UTF-8): the element-text part of a label key."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def calib_half(doc_id: str) -> CalibHalf:
    """Which half of the ``calib`` split a document belongs to: ``"fit"`` (calib-fit) when
    ``sha256("calibsplit:" + doc_id)`` (UTF-8), its first 8 bytes read as a big-endian
    integer over 2**64, is below 0.5, else ``"crc"`` (calib-crc).

    calib-fit is what v0.5 fits its temperatures on. calib-crc is never fitted by v0.5 (no
    training, active learning, term mining or temperatures): it is reserved for the v0.6
    conformal calibration, which reuses this exact partition. The answer depends on the id
    only, so it is the same on every run and machine."""
    digest = hashlib.sha256((_CALIB_SALT + doc_id).encode("utf-8")).digest()
    return "fit" if int.from_bytes(digest[:8], "big") / 2**64 < 0.5 else "crc"


def is_calib_crc(record: ManifestRecord) -> bool:
    """Whether a corpus document is in calib-crc: the ``calib`` split's half that no v0.5
    fitting may read (:func:`calib_half`)."""
    return record.split == "calib" and calib_half(record.id) == "crc"


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"invalid {what} {value!r}: use 1-128 letters, digits, '.', '_' or '-', starting "
            "with a letter or digit"
        )
    return value


class ManifestRecord(BaseModel):
    """One corpus document, as the fetcher's manifest describes it (spec §5).

    ``split`` is ``train``, ``calib`` or ``gold``. Fields the fetcher adds beyond these (a
    title, a publisher) are kept as they are.
    """

    model_config = ConfigDict(extra="allow")

    id: str
    url: str = ""
    source: str = ""
    licence: str = ""
    sha256: str
    bytes: int = Field(default=0, ge=0)
    pages: int | None = Field(default=None, ge=0)
    fetched_at: str = ""
    doc_type_hint: str | None = None
    family_hint: str | None = None
    split: Split = "train"

    @field_validator("id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return _check_name(value, "document id")

    @field_validator("sha256")
    @classmethod
    def _hex_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("sha256 must be 64 lowercase hex digits")
        return value


class CorpusElement(BaseModel):
    """One stored element. ``heading_path`` runs from the root heading down to the element's
    section (the preamble has none); ``section_title`` is that section's title (``None`` in the
    preamble); ``page`` is the 1-based page of its first box. ``text_sha1`` is computed from
    ``text`` when not given, and must match it when given. ``table_markdown`` holds a table
    element's cells as a Markdown table (``None`` for other kinds)."""

    doc_id: str
    element_id: str
    kind: ElementKind
    text: str
    text_sha1: str = ""
    heading_path: list[str]
    section_title: str | None
    page: int | None
    table_markdown: str | None = None

    @model_validator(mode="after")
    def _text_hash(self) -> CorpusElement:
        expected = text_sha1(self.text)
        if not self.text_sha1:
            self.text_sha1 = expected
        elif self.text_sha1 != expected:
            raise ValueError("text_sha1 does not match the element text")
        return self


class LabelKey(NamedTuple):
    """What a label row is about: an element (by document, id and text hash) and a facet. A
    changed element text gives another key, so labels of the old text no longer apply."""

    doc_id: str
    element_id: str
    text_sha1: str
    facet: str


class LabelRow(BaseModel):
    """One labeller's distribution for one element and facet, and where it came from
    (``provenance``: engine, model, revision, prompt hash, taxonomy hash, timestamp)."""

    doc_id: str
    element_id: str
    text_sha1: str
    facet: str = Field(min_length=1)
    probs: dict[str, float]
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text_sha1")
    @classmethod
    def _hex_sha1(cls, value: str) -> str:
        if not _SHA1.fullmatch(value):
            raise ValueError("text_sha1 must be 40 lowercase hex digits")
        return value

    @field_validator("probs")
    @classmethod
    def _probabilities(cls, value: dict[str, float]) -> dict[str, float]:
        for label, p in value.items():
            if not (math.isfinite(p) and 0.0 <= p <= 1.0):
                raise ValueError(f"probability of {label!r} must be in [0, 1], got {p}")
        return value

    @property
    def key(self) -> LabelKey:
        return LabelKey(self.doc_id, self.element_id, self.text_sha1, self.facet)


# --- files -----------------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    """Make a rename in ``path`` durable (best effort: not every platform opens directories)."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file in the same directory and
    ``os.replace``: ``path`` holds either its old content or all of ``data``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def _append_lines(path: Path, lines: list[str]) -> None:
    """Append ``lines`` (each without its newline) to a JSON-lines file, after cutting off a
    torn last line, and fsync. ``lines`` go in one write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(line + "\n" for line in lines).encode("utf-8")
    with open(path, "a+b") as f:
        _cut_torn_tail(f)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _cut_torn_tail(f: Any) -> None:
    """Truncate an open file after its last newline (to empty if it has none)."""
    size = f.seek(0, os.SEEK_END)
    if size == 0:
        return
    f.seek(size - 1)
    if f.read(1) == b"\n":
        return
    end = size
    keep = 0
    while end > 0:
        start = max(0, end - 65536)
        f.seek(start)
        found = f.read(end - start).rfind(b"\n")
        if found >= 0:
            keep = start + found + 1
            break
        end = start
    f.truncate(keep)
    f.seek(keep)


def _complete_lines(path: Path) -> Iterator[tuple[int, str]]:
    """The newline-terminated, non-blank lines of ``path`` with their 1-based numbers, read one
    at a time; a torn last line is skipped. A missing file has none."""
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return
    with f:
        for number, line in enumerate(f, 1):  # binary files split on b"\n" only
            if not line.endswith(b"\n"):
                return  # torn: the writer died before the newline
            if line.strip():
                yield number, line.decode("utf-8")


def read_manifest(path: str | os.PathLike[str]) -> list[ManifestRecord]:
    """The records of a manifest file (JSON lines; blank lines skipped). Any invalid line is a
    :class:`~docrooter.errors.DocrooterError` naming its line number."""
    path = Path(path)
    records: list[ManifestRecord] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocrooterError(f"cannot read manifest {path}: {exc}") from exc
    for number, line in enumerate(text.split("\n"), 1):  # not splitlines: U+2028 is text
        if not line.strip():
            continue
        try:
            records.append(ManifestRecord.model_validate_json(line))
        except ValidationError as exc:
            raise DocrooterError(f"{path} line {number}: invalid manifest record: {exc}") from exc
    return records


# --- the store -------------------------------------------------------------------------------


class Corpus:
    """The corpus store in directory ``dir`` (created on the first write).

    One process writes a store at a time. The manifest is read once, when first needed, and
    kept up to date by :meth:`add_document`.
    """

    def __init__(self, dir: str | os.PathLike[str]) -> None:
        self.dir = Path(dir)
        self._records: dict[str, ManifestRecord] | None = None
        self._by_sha: dict[str, str] = {}
        self._folded: dict[str, str] = {}

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.jsonl"

    def _doc_path(self, doc_id: str) -> Path:
        return self.dir / "docs" / f"{doc_id}.json"

    def _labels_path(self, labeler: str) -> Path:
        return self.dir / "labels" / f"{_check_name(labeler, 'labeler name')}.jsonl"

    def _index(self) -> dict[str, ManifestRecord]:
        if self._records is None:
            records: dict[str, ManifestRecord] = {}
            for number, line in _complete_lines(self.manifest_path):
                try:
                    record = ManifestRecord.model_validate_json(line)
                except ValidationError as exc:
                    raise DocrooterError(
                        f"{self.manifest_path} line {number}: invalid manifest record: {exc}"
                    ) from exc
                if record.id not in records and record.sha256 not in self._by_sha:
                    self._remember(records, record)
            self._records = records
        return self._records

    def _remember(self, records: dict[str, ManifestRecord], record: ManifestRecord) -> None:
        records[record.id] = record
        self._by_sha[record.sha256] = record.id
        self._folded[record.id.casefold()] = record.id

    def manifest(self) -> list[ManifestRecord]:
        """The documents in the corpus, in the order they were added."""
        return [record.model_copy() for record in self._index().values()]

    def has(self, doc_id: str) -> bool:
        """Whether document ``doc_id`` is in the corpus (its manifest line is written)."""
        return doc_id in self._index()

    def has_sha256(self, sha256: str) -> bool:
        """Whether a document with these bytes is in the corpus, under any id."""
        self._index()
        return sha256 in self._by_sha

    def id_conflict(self, record: ManifestRecord) -> str | None:
        """Why ``record`` cannot be added under its id (``None`` if it can, or if its bytes are
        already in the corpus): the id is in use for other bytes, or differs only in case from
        one in use."""
        records = self._index()
        if record.sha256 in self._by_sha:
            return None
        if record.id in records:
            return f"document id {record.id!r} is already in the corpus with another sha256"
        clash = self._folded.get(record.id.casefold())
        if clash is not None:
            return (
                f"document id {record.id!r} differs only in case from {clash!r}, which is in the "
                "corpus (their files would collide on a case-insensitive file system)"
            )
        return None

    def add_document(
        self, record: ManifestRecord, elements: Iterable[CorpusElement], pages_read: Iterable[int]
    ) -> bool:
        """Store a document's elements and add its record to the manifest.

        Idempotent by sha256: a document whose bytes are already in the corpus (under this id
        or another) is left as it is and ``False`` is returned. An id already used for other
        bytes, or differing only in case from one in use, is a ``ValueError``. Every element
        must belong to ``record.id``, with unique element ids; ``pages_read`` are the 1-based
        pages the elements were read from. Returns ``True`` when the document was added.
        """
        records = self._index()
        if record.sha256 in self._by_sha:
            return False
        conflict = self.id_conflict(record)
        if conflict is not None:
            raise ValueError(conflict)
        stored = list(elements)
        seen: set[str] = set()
        for element in stored:
            if element.doc_id != record.id:
                raise ValueError(
                    f"element {element.element_id!r} has doc_id {element.doc_id!r}, "
                    f"not {record.id!r}"
                )
            if element.element_id in seen:
                raise ValueError(f"duplicate element id {element.element_id!r} in {record.id!r}")
            seen.add(element.element_id)
        pages = sorted(set(pages_read))
        if any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in pages):
            raise ValueError("pages_read must be 1-based page numbers")
        doc = {
            "id": record.id,
            "sha256": record.sha256,
            "doc_type_hint": record.doc_type_hint,
            "family_hint": record.family_hint,
            "pages_read": pages,
            "elements": [_element_json(element) for element in stored],
        }
        data = json.dumps(doc, ensure_ascii=False) + "\n"
        _write_atomic(self._doc_path(record.id), data.encode("utf-8"))
        _append_lines(self.manifest_path, [record.model_dump_json()])  # the commit point
        self._remember(records, record)
        return True

    def iter_elements(
        self,
        split: str | Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> Iterator[CorpusElement]:
        """The stored elements, document by document in manifest order, then in reading order.

        ``split`` keeps the documents of one split or several (``train``, ``calib``, ``gold``);
        ``kinds`` keeps the elements of those kinds. Unknown names are a ``ValueError``.
        """
        splits = _names(split, SPLITS, "split")
        wanted = _names(kinds, _KINDS, "kind")
        records = [r for r in self._index().values() if splits is None or r.split in splits]
        return self._elements(records, wanted)

    def iter_fit_elements(
        self,
        split: str | Iterable[str] | None = None,
        kinds: Iterable[str] | None = None,
    ) -> Iterator[CorpusElement]:
        """:meth:`iter_elements` without the calib-crc documents (:func:`is_calib_crc`): the
        elements any v0.5 fitting may read (training, active learning, term mining,
        temperatures). ``split="calib"`` gives the calib-fit half."""
        splits = _names(split, SPLITS, "split")
        wanted = _names(kinds, _KINDS, "kind")
        records = [
            r
            for r in self._index().values()
            if (splits is None or r.split in splits) and not is_calib_crc(r)
        ]
        return self._elements(records, wanted)

    def _elements(
        self, records: list[ManifestRecord], kinds: frozenset[str] | None
    ) -> Iterator[CorpusElement]:
        for record in records:
            doc = self._read_doc(record)
            for item in doc["elements"]:
                if kinds is not None and item["kind"] not in kinds:
                    continue
                try:
                    element = CorpusElement(
                        doc_id=record.id,
                        element_id=item["id"],
                        kind=item["kind"],
                        text=item["text"],
                        heading_path=item["heading_path"],
                        section_title=item["section_title"],
                        page=item["page"],
                        table_markdown=item.get("table_markdown"),
                    )
                except ValidationError as exc:
                    raise DocrooterError(
                        f"corpus document {self._doc_path(record.id)} holds an invalid element: "
                        f"{exc}"
                    ) from exc
                yield element

    def _read_doc(self, record: ManifestRecord) -> dict[str, Any]:
        path = self._doc_path(record.id)
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise DocrooterError(f"cannot read corpus document {path}: {exc}") from exc
        if (
            not isinstance(doc, dict)
            or doc.get("id") != record.id
            or doc.get("sha256") != record.sha256
        ):
            raise DocrooterError(f"corpus document {path} does not match its manifest record")
        elements, pages = doc.get("elements"), doc.get("pages_read")
        if (
            not isinstance(elements, list)
            or not isinstance(pages, list)
            or not all(_is_element_item(item) for item in elements)
        ):
            raise DocrooterError(
                f"corpus document {path} is malformed: it needs 'pages_read' and 'elements', and "
                f"every element needs {', '.join(sorted(_ELEMENT_KEYS))} and a known kind"
            )
        return doc

    def pages_read(self, doc_id: str) -> list[int]:
        """The 1-based pages document ``doc_id``'s elements were read from (``KeyError`` when it
        is not in the corpus)."""
        record = self._index().get(doc_id)
        if record is None:
            raise KeyError(f"document {doc_id!r} is not in the corpus")
        return [int(p) for p in self._read_doc(record)["pages_read"]]

    def write_labels(self, labeler: str, rows: Iterable[LabelRow]) -> int:
        """Append ``rows`` to ``labels/<labeler>.jsonl`` in one write; returns how many. A row
        for a key that already has one supersedes it (see :meth:`labels`)."""
        path = self._labels_path(labeler)
        lines = [row.model_dump_json() for row in rows]
        if lines:
            _append_lines(path, lines)
        return len(lines)

    def labels(self, labeler: str) -> dict[LabelKey, LabelRow]:
        """``labeler``'s rows by key (document, element, text sha1, facet); the last row written
        for a key wins. A labeller with no file has none."""
        path = self._labels_path(labeler)
        out: dict[LabelKey, LabelRow] = {}
        for number, line in _complete_lines(path):
            try:
                row = LabelRow.model_validate_json(line)
            except ValidationError as exc:
                raise DocrooterError(f"{path} line {number}: invalid label row: {exc}") from exc
            out[row.key] = row
        return out

    def labelers(self) -> list[str]:
        """The names of the labellers with a label file, sorted."""
        folder = self.dir / "labels"
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.jsonl") if _SAFE_NAME.fullmatch(p.stem))

    def _count_label_keys(self, labeler: str) -> int:
        """The distinct keys of ``labeler``'s rows, counted line by line (no row validation)."""
        path = self._labels_path(labeler)
        keys: set[tuple[Any, ...]] = set()
        for number, line in _complete_lines(path):
            try:
                row = json.loads(line)
                keys.add((row["doc_id"], row["element_id"], row["text_sha1"], row["facet"]))
            except (ValueError, KeyError, TypeError) as exc:
                raise DocrooterError(f"{path} line {number}: invalid label row: {exc}") from exc
        return len(keys)

    # --- build bookkeeping: the document being read, and the ones that could not be ----------

    @property
    def _attempt_path(self) -> Path:
        return self.dir / "attempting.json"

    @property
    def _failed_path(self) -> Path:
        return self.dir / "failed.jsonl"

    def start_attempt(self, doc_id: str) -> None:
        """Durably note that ``doc_id`` is being read, before reading it. If the process dies
        while reading (a native crash, the OOM killer), the note survives it:
        :meth:`pending_attempt` names the document on the next run."""
        _check_name(doc_id, "document id")
        data = json.dumps({"doc_id": doc_id, "started_at": _now()}) + "\n"
        _write_atomic(self._attempt_path, data.encode("utf-8"))

    def finish_attempt(self) -> None:
        """Clear the note of :meth:`start_attempt` (the document was added, its failure
        recorded, or the run stopped for a reason that is not the document's)."""
        with contextlib.suppress(FileNotFoundError):
            self._attempt_path.unlink()
            _fsync_dir(self.dir)

    def pending_attempt(self) -> str | None:
        """The document an earlier run was reading when it died, if any."""
        try:
            data = json.loads(self._attempt_path.read_text(encoding="utf-8"))
            return _check_name(data["doc_id"], "document id")
        except FileNotFoundError:
            return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DocrooterError(f"cannot read {self._attempt_path}: {exc}") from exc

    def record_failure(self, doc_id: str, reason: str) -> None:
        """Append ``doc_id`` and why it could not be read to ``failed.jsonl``."""
        _check_name(doc_id, "document id")
        line = json.dumps({"doc_id": doc_id, "reason": reason, "timestamp": _now()})
        _append_lines(self._failed_path, [line])

    def failures(self) -> dict[str, dict[str, str]]:
        """The recorded failures by document id (the latest for each), in first-seen order:
        ``{"reason": ..., "timestamp": ...}``."""
        out: dict[str, dict[str, str]] = {}
        for number, line in _complete_lines(self._failed_path):
            try:
                entry = json.loads(line)
                out[str(entry["doc_id"])] = {
                    "reason": str(entry["reason"]),
                    "timestamp": str(entry["timestamp"]),
                }
            except (ValueError, KeyError, TypeError) as exc:
                raise DocrooterError(f"{self._failed_path} line {number}: invalid entry") from exc
        return out

    def stats(self) -> dict[str, Any]:
        """Counts of documents, elements and pages read (in all, by split, by source, by kind),
        of label keys by labeller and of documents that failed and are not in the corpus; also
        written to ``stats.json``."""
        records = list(self._index().values())
        by_split = {name: {"documents": 0, "elements": 0} for name in SPLITS}
        by_source: Counter[str] = Counter()
        by_kind: Counter[str] = Counter()
        pages = 0
        for record in records:
            doc = self._read_doc(record)
            by_split[record.split]["documents"] += 1
            by_split[record.split]["elements"] += len(doc["elements"])
            by_source[record.source] += 1
            by_kind.update(item["kind"] for item in doc["elements"])
            pages += len(doc["pages_read"])
        stats: dict[str, Any] = {
            "documents": len(records),
            "elements": sum(by_kind.values()),
            "pages_read": pages,
            "by_split": by_split,
            "by_source": dict(sorted(by_source.items())),
            "by_kind": dict(sorted(by_kind.items())),
            "labels": {name: self._count_label_keys(name) for name in self.labelers()},
            "failed": len([d for d in self.failures() if d not in self._index()]),
        }
        if records or self.dir.exists():
            data = json.dumps(stats, indent=2, sort_keys=True) + "\n"
            _write_atomic(self.dir / "stats.json", data.encode("utf-8"))
        return stats


def _is_element_item(item: object) -> bool:
    return isinstance(item, dict) and _ELEMENT_KEYS <= item.keys() and item["kind"] in _KINDS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _element_json(element: CorpusElement) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": element.element_id,
        "kind": element.kind,
        "text": element.text,
        "heading_path": element.heading_path,
        "section_title": element.section_title,
        "page": element.page,
    }
    if element.table_markdown is not None:
        item["table_markdown"] = element.table_markdown
    return item


def _names(
    value: str | Iterable[str] | None, known: Iterable[str], what: str
) -> frozenset[str] | None:
    if value is None:
        return None
    names = frozenset([value] if isinstance(value, str) else value)
    unknown = sorted(names - set(known))
    if unknown:
        raise ValueError(
            f"unknown {what} {', '.join(map(repr, unknown))}; use {', '.join(sorted(known))}"
        )
    return names
