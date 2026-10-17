"""Read taxonomies from nested/flat JSON and adjacency/levels/path CSV."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

from pydantic import ValidationError

from docrooter.errors import TaxonomyError
from docrooter.taxonomy.models import Taxonomy, TaxonomyNode, check_root
from docrooter.taxonomy.validate import TaxonomyIssue

logger = logging.getLogger("docrooter.taxonomy")


class ParsedNodes(NamedTuple):
    """Nodes read from a file, where each came from, and optional root-level settings."""

    nodes: list[TaxonomyNode]
    locations: list[str | None]
    root: dict[str, Any] | None = None


_LEVEL_COLUMN = re.compile(r"level_(\d+)")
_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def slug(label: str) -> str:
    """Lowercase, runs of non-alphanumerics become ``-``, stripped. Empty becomes ``untitled``."""
    return re.sub(r"[\W_]+", "-", label.lower()).strip("-") or "untitled"


def _fail(code: str, message: str, location: str | None = None) -> NoReturn:
    raise TaxonomyError(message, [TaxonomyIssue("error", code, message, location)])


def _build(parsed: ParsedNodes) -> Taxonomy:
    taxonomy = Taxonomy(parsed.nodes, locations=parsed.locations, root=parsed.root)
    for warning in taxonomy.warnings:
        where = f" ({warning.location})" if warning.location else ""
        logger.warning("taxonomy %s%s: %s", warning.code, where, warning.message)
    return taxonomy


# ---------------------------------------------------------------- JSON


def nodes_from_dict(data: Any) -> ParsedNodes:
    """Parse nested and/or flat JSON data into nodes plus JSON-pointer locations.

    A top-level ``"root"`` object (dict form only) holds root-level decision settings.
    Items may carry ``children`` as node objects (nested form; the parent comes from the nesting).
    Child entries that are plain strings are ids from a flat dump and are ignored, since
    ``parent_id`` already carries the link. Missing ids are generated from the label path.
    """
    root: dict[str, Any] | None = None
    if isinstance(data, dict):
        items, base = data.get("nodes"), "/nodes"
        if not isinstance(items, list):
            _fail("invalid_structure", "expected a top-level 'nodes' list", "/nodes")
        if data.get("root") is not None:
            root = check_root(data["root"], location="/root")
    elif isinstance(data, list):
        items, base = data, ""
    else:
        _fail("invalid_structure", "expected an object with 'nodes' or a list of nodes", "")

    nodes: list[TaxonomyNode] = []
    locations: list[str | None] = []
    stack: list[tuple[Any, str, str | None, tuple[str, ...]]] = [
        (item, f"{base}/{i}", None, ()) for i, item in reversed(list(enumerate(items)))
    ]
    while stack:
        raw, pointer, nesting_parent, label_path = stack.pop()
        if not isinstance(raw, dict):
            _fail("invalid_node", "node must be an object", pointer)
        fields = dict(raw)
        child_items = fields.pop("children", [])
        if not isinstance(child_items, list):
            _fail("invalid_node", "'children' must be a list", f"{pointer}/children")

        label = fields.get("label")
        if isinstance(label, str):
            label_path = (*label_path, label)
            if fields.get("id") is None:
                fields["id"] = ".".join(slug(part) for part in label_path)
        if nesting_parent is not None:
            stated = fields.get("parent_id")
            if stated is not None and stated != nesting_parent:
                _fail(
                    "parent_mismatch",
                    f"parent_id {stated!r} disagrees with the enclosing node {nesting_parent!r}",
                    pointer,
                )
            fields["parent_id"] = nesting_parent
        try:
            node = TaxonomyNode(**fields)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            _fail("invalid_node", detail, pointer)
        nodes.append(node)
        locations.append(pointer)
        for j in range(len(child_items) - 1, -1, -1):
            if not isinstance(child_items[j], str):
                stack.append((child_items[j], f"{pointer}/children/{j}", node.id, label_path))
    return ParsedNodes(nodes, locations, root)


def from_dict(data: Any) -> Taxonomy:
    return _build(nodes_from_dict(data))


# ---------------------------------------------------------------- CSV


def _csv_text(path_or_text: str | Path) -> str:
    if isinstance(path_or_text, Path) or _is_file(path_or_text):
        return Path(path_or_text).read_text(encoding="utf-8-sig")
    return path_or_text.lstrip("﻿")


def _is_file(candidate: str) -> bool:
    if "\n" in candidate or "\r" in candidate:
        return False
    try:
        return Path(candidate).is_file()
    except OSError:
        return False


def _row_attrs(cells: dict[str, str], location: str) -> dict[str, Any]:
    """Optional per-node columns shared by all three CSV layouts."""
    attrs: dict[str, Any] = {}
    if cells.get("description"):
        attrs["description"] = cells["description"]
    if cells.get("examples"):
        attrs["examples"] = [e.strip() for e in cells["examples"].split("|") if e.strip()]
    if cells.get("backend"):
        attrs["backend"] = cells["backend"]
    if cells.get("threshold"):
        try:
            attrs["threshold"] = float(cells["threshold"])
        except ValueError:
            _fail("invalid_value", f"threshold {cells['threshold']!r} is not a number", location)
    if cells.get("multi_label"):
        flag = cells["multi_label"].lower()
        if flag not in _TRUE | _FALSE:
            _fail(
                "invalid_value", f"multi_label {cells['multi_label']!r} is not a boolean", location
            )
        attrs["multi_label"] = flag in _TRUE
    return attrs


def nodes_from_csv(text: str, *, sep: str = ">") -> ParsedNodes:
    """Parse CSV text into nodes plus ``"row N"`` locations (header is row 1)."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿"), newline=""))
    if not reader.fieldnames:
        _fail("unknown_csv_format", "CSV file is empty", "row 1")
    reader.fieldnames = [(name or "").strip().lower() for name in reader.fieldnames]
    header = set(reader.fieldnames)

    rows: list[tuple[str, dict[str, str]]] = []
    for row_no, raw in enumerate(reader, start=2):
        cells = {k: (v or "").strip() for k, v in raw.items() if k and isinstance(v, str | None)}
        if any(cells.values()):
            rows.append((f"row {row_no}", cells))

    if "parent_id" in header:
        if "label" not in header:
            _fail("unknown_csv_format", "adjacency CSV needs a 'label' column", "row 1")
        return _adjacency(rows)
    if "level_1" in header:
        columns = sorted(
            (c for c in reader.fieldnames if _LEVEL_COLUMN.fullmatch(c)),
            key=lambda c: int(c.split("_")[1]),
        )
        return _from_paths(_level_entries(rows, columns))
    if "path" in header:
        return _from_paths(_path_entries(rows, sep))
    _fail(
        "unknown_csv_format",
        "cannot tell the CSV layout from its header "
        f"({', '.join(reader.fieldnames)}); expected 'parent_id', 'level_1' or 'path'",
        "row 1",
    )


def _adjacency(rows: list[tuple[str, dict[str, str]]]) -> ParsedNodes:
    nodes: list[TaxonomyNode] = []
    locations: list[str | None] = []
    for location, cells in rows:
        label = cells.get("label", "")
        parent = cells.get("parent_id") or None
        node_id = cells.get("id") or (f"{parent}." if parent else "") + slug(label)
        nodes.append(
            TaxonomyNode(id=node_id, label=label, parent_id=parent, **_row_attrs(cells, location))
        )
        locations.append(location)
    return ParsedNodes(nodes, locations)


_PathEntry = tuple[list[str], dict[str, Any], str]


def _level_entries(rows: list[tuple[str, dict[str, str]]], columns: list[str]) -> list[_PathEntry]:
    entries: list[_PathEntry] = []
    for location, cells in rows:
        labels = [cells.get(c, "") for c in columns]
        while labels and not labels[-1]:
            labels.pop()
        if not labels:
            _fail("empty_path", "row has no level_N value", location)
        if not all(labels):
            _fail("empty_path_segment", "empty cell between filled levels", location)
        entries.append((labels, _row_attrs(cells, location), location))
    return entries


def _path_entries(rows: list[tuple[str, dict[str, str]]], sep: str) -> list[_PathEntry]:
    entries: list[_PathEntry] = []
    for location, cells in rows:
        parts = [p.strip() for p in cells.get("path", "").split(sep)]
        if parts == [""]:
            _fail("empty_path", "row has an empty path", location)
        if not all(parts):
            _fail("empty_path_segment", f"empty segment in path {cells['path']!r}", location)
        entries.append((parts, _row_attrs(cells, location), location))
    return entries


def _from_paths(entries: list[_PathEntry]) -> ParsedNodes:
    """Build nodes from one-path-per-row entries; ancestors are created implicitly."""
    nodes: list[TaxonomyNode] = []
    locations: list[str | None] = []
    index: dict[tuple[str, ...], int] = {}
    declared: set[tuple[str, ...]] = set()
    for labels, attrs, location in entries:
        for depth in range(1, len(labels) + 1):
            key = tuple(labels[:depth])
            if key in index:
                continue
            index[key] = len(nodes)
            parent = nodes[index[key[:-1]]].id if depth > 1 else None
            node_id = ".".join(slug(part) for part in key)
            nodes.append(TaxonomyNode(id=node_id, label=key[-1], parent_id=parent))
            locations.append(location)
        key = tuple(labels)
        if key in declared:
            # Repeat the node so validation reports the duplicate at this row.
            nodes.append(nodes[index[key]].model_copy(update=attrs))
            locations.append(location)
            continue
        declared.add(key)
        nodes[index[key]] = nodes[index[key]].model_copy(update=attrs)
        locations[index[key]] = location
    return ParsedNodes(nodes, locations)


def from_csv(path_or_text: str | Path, *, sep: str = ">") -> Taxonomy:
    """Build a taxonomy from a CSV file path or CSV text. Layout is detected from the header."""
    return _build(nodes_from_csv(_csv_text(path_or_text), sep=sep))


# ---------------------------------------------------------------- files


def read_nodes(path: str | Path, *, sep: str = ">") -> ParsedNodes:
    """Parse a JSON or CSV taxonomy file without validating the tree."""
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    is_json = suffix == ".json" or (suffix != ".csv" and text.lstrip()[:1] in ("{", "["))
    if not is_json:
        return nodes_from_csv(text, sep=sep)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        _fail("invalid_json", f"invalid JSON: {exc.msg}", f"line {exc.lineno}, column {exc.colno}")
    return nodes_from_dict(data)


def load(path: str | Path, *, sep: str = ">") -> Taxonomy:
    return _build(read_nodes(path, sep=sep))
