"""Labelled training examples: loading, saving, label resolution, splitting."""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from docrooter.errors import DatasetError
from docrooter.taxonomy import Taxonomy

if TYPE_CHECKING:
    from docrooter.classify.results import DocumentResult
    from docrooter.document import Document
    from docrooter.views import SectionView

PATH_SEP = " > "


class Example(BaseModel):
    """One labelled text. ``path`` holds node ids or labels, from the top level down."""

    text: str
    path: list[str]
    doc_id: str | None = None
    section_id: str | None = None


def _split_path(value: str) -> list[str]:
    return [part.strip() for part in value.split(">") if part.strip()]


class Dataset:
    """A list of :class:`Example` with I/O, taxonomy resolution and stratified splitting.

    Files are JSONL (one ``{"text", "path", ...}`` object per line, ``path`` a list) or CSV with
    ``text,path`` columns (``path`` joined by ``" > "``). Paths may mix node ids and labels;
    :meth:`resolve` turns them into ids. Error messages name the example's line in its source
    file, counting non-blank lines (so a CSV header is line 1).
    """

    def __init__(self, examples: Sequence[Example] = (), *, _lines: Sequence[int] | None = None):
        self.examples: list[Example] = list(examples)
        self._lines: list[int] = (
            list(_lines) if _lines is not None else list(range(1, len(self.examples) + 1))
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[Example]:
        return iter(self.examples)

    def __repr__(self) -> str:
        return f"Dataset({len(self)} examples)"

    def _subset(self, indices: Sequence[int]) -> Dataset:
        return Dataset(
            [self.examples[i] for i in indices], _lines=[self._lines[i] for i in indices]
        )

    # --- I/O ------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """Read a ``.csv`` file, or JSONL for any other extension."""
        p = Path(path)
        try:
            raw = p.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            raise DatasetError(f"dataset file not found: {p}") from None
        except (OSError, UnicodeDecodeError) as exc:
            raise DatasetError(f"cannot read dataset {p}: {exc}") from exc
        records = _csv_records(raw, p) if p.suffix.lower() == ".csv" else _jsonl_records(raw, p)
        examples, lines = [], []
        for line, record in records:
            examples.append(_example_from(record, line, p))
            lines.append(line)
        return cls(examples, _lines=lines)

    def save(self, path: str | Path) -> None:
        """Write ``.csv`` (paths joined by ``" > "``), or JSONL for any other extension."""
        p = Path(path)
        if p.suffix.lower() == ".csv":
            extra = [k for k in ("doc_id", "section_id") if any(getattr(e, k) for e in self)]
            with p.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["text", "path", *extra])
                for e in self:
                    writer.writerow(
                        [e.text, PATH_SEP.join(e.path), *(getattr(e, k) or "" for k in extra)]
                    )
        else:
            with p.open("w", encoding="utf-8") as f:
                for e in self:
                    f.write(json.dumps(e.model_dump(exclude_none=True), ensure_ascii=False) + "\n")

    # --- taxonomy -------------------------------------------------------------------------

    def resolve(self, taxonomy: Taxonomy) -> Dataset:
        """A copy with every path turned into node ids (given as ids or labels, per level)."""
        resolved = []
        for example, line in zip(self.examples, self._lines, strict=True):
            if not example.path:
                raise DatasetError(f"line {line}: empty path")
            try:
                ids = taxonomy.resolve_path(example.path)
            except KeyError as exc:
                raise DatasetError(f"line {line}: {exc.args[0]}") from exc
            resolved.append(example.model_copy(update={"path": ids}))
        return Dataset(resolved, _lines=self._lines)

    def node_examples(self, taxonomy: Taxonomy, node_id: str) -> tuple[list[str], list[str]]:
        """Texts and the id of the child each leads to, for examples passing through ``node_id``.

        ``node_id`` may be ``Taxonomy.ROOT_ID``. Examples whose path ends at the node itself
        carry no decision for it and are left out.
        """
        texts: list[str] = []
        children: list[str] = []
        for example in self.resolve(taxonomy):
            path = example.path
            if node_id == Taxonomy.ROOT_ID:
                nxt = 0
            elif node_id in path:
                nxt = path.index(node_id) + 1
            else:
                continue
            if nxt < len(path):
                texts.append(example.text)
                children.append(path[nxt])
        return texts, children

    def split(
        self, test_size: float = 0.2, stratify_depth: int = 1, seed: int = 0
    ) -> tuple[Dataset, Dataset]:
        """``(train, test)`` stratified on ``path[:stratify_depth]``, deterministic for a seed.

        Each class puts ``round(n * test_size)`` examples in ``test``, at least one and at most
        ``n - 1``. A class with a single example goes entirely to ``train``.
        """
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        groups: dict[tuple[str, ...], list[int]] = {}
        for i, example in enumerate(self.examples):
            groups.setdefault(tuple(example.path[:stratify_depth]), []).append(i)
        rng = random.Random(seed)
        test_idx: set[int] = set()
        for key in sorted(groups):
            members = groups[key]
            if len(members) < 2:
                continue
            shuffled = members[:]
            rng.shuffle(shuffled)
            n_test = max(1, min(len(members) - 1, round(len(members) * test_size)))
            test_idx.update(shuffled[:n_test])
        train = [i for i in range(len(self)) if i not in test_idx]
        return self._subset(train), self._subset(sorted(test_idx))

    # --- from classifier output -----------------------------------------------------------

    @classmethod
    def from_results(cls, result: DocumentResult, doc: Document, view: SectionView) -> Dataset:
        """Pseudo-labelled examples from a classification: rendered section text -> best path.

        Sections with no best path, or an empty one, are skipped.
        """
        ids = [sid for sid, r in result.sections.items() if r.best is not None and r.best.node_ids]
        unknown = [sid for sid in ids if sid not in doc.sections]
        if unknown:
            raise DatasetError(
                f"sections {unknown} of the result are not in document {doc.source!r}"
            )
        rendered = view.render_all(doc, ids)
        examples = [
            Example(
                text=rendered[sid].text,
                path=list(result.sections[sid].best.node_ids),  # type: ignore[union-attr]
                doc_id=doc.source,
                section_id=sid,
            )
            for sid in ids
        ]
        return cls(examples)


def _jsonl_records(raw: str, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    line = 0
    for content in raw.splitlines():
        if not content.strip():
            continue
        line += 1
        try:
            record = json.loads(content)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}: line {line}: invalid JSON ({exc.msg})") from exc
        if not isinstance(record, dict):
            raise DatasetError(f"{path}: line {line}: expected a JSON object")
        yield line, record


def _csv_records(raw: str, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    rows = (row for row in csv.reader(raw.splitlines(keepends=True)) if any(c.strip() for c in row))
    header = next(rows, None)
    if header is None:
        return
    names = [h.strip().lower() for h in header]
    missing = [c for c in ("text", "path") if c not in names]
    if missing:
        raise DatasetError(
            f"{path}: line 1: CSV needs a 'text' and a 'path' column; missing {missing}"
        )
    for line, row in enumerate(rows, start=2):
        record: dict[str, Any] = dict(zip(names, row, strict=False))
        record["path"] = _split_path(record.get("path", ""))
        yield line, record


def _example_from(record: dict[str, Any], line: int, path: Path) -> Example:
    text, parts = record.get("text"), record.get("path")
    if not isinstance(text, str) or not text.strip():
        raise DatasetError(f"{path}: line {line}: missing or empty 'text'")
    if isinstance(parts, str):
        parts = _split_path(parts)
    if (
        not isinstance(parts, list)
        or not parts
        or not all(isinstance(p, str) and p.strip() for p in parts)
    ):
        raise DatasetError(f"{path}: line {line}: 'path' must be a non-empty list of labels or ids")
    return Example(
        text=text,
        path=[p.strip() for p in parts],
        doc_id=record.get("doc_id") or None,
        section_id=record.get("section_id") or None,
    )
