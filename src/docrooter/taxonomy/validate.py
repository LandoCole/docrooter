"""Structural validation of taxonomy nodes."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from docrooter.taxonomy.models import TaxonomyNode

ROOT_ID = "__root__"


@dataclass(frozen=True)
class TaxonomyIssue:
    """One problem found in a taxonomy. ``location`` is a CSV row or JSON pointer."""

    severity: Literal["error", "warning"]
    code: str
    message: str
    location: str | None = None


def normalise_label(label: str) -> str:
    """Normalise a label for sibling-duplicate comparison."""
    return " ".join(unicodedata.normalize("NFKC", label).split()).casefold()


def validate_nodes(
    nodes: Sequence[TaxonomyNode], *, locations: Sequence[str | None] | None = None
) -> list[TaxonomyIssue]:
    """Check ``nodes`` and return every issue found.

    ``locations`` is parallel to ``nodes`` and says where each one came from in the source file.
    Codes: ``duplicate_id``, ``reserved_id``, ``missing_parent``, ``cycle``, ``empty_label``,
    ``duplicate_sibling_label``, ``empty_taxonomy`` (errors) and ``single_child`` (warning).
    """
    locs: list[str | None] = list(locations) if locations is not None else []
    locs += [None] * (len(nodes) - len(locs))
    issues: list[TaxonomyIssue] = []

    def error(code: str, message: str, index: int) -> None:
        issues.append(TaxonomyIssue("error", code, message, locs[index]))

    if not nodes:
        return [TaxonomyIssue("error", "empty_taxonomy", "taxonomy has no nodes", None)]

    first: dict[str, int] = {}
    for i, node in enumerate(nodes):
        if node.id == ROOT_ID:
            error("reserved_id", f"id {ROOT_ID!r} is reserved for the virtual root", i)
        if node.id in first:
            where = locs[first[node.id]]
            suffix = f" (first defined at {where})" if where else ""
            error("duplicate_id", f"duplicate id {node.id!r}{suffix}", i)
        else:
            first[node.id] = i
        if not node.label.strip():
            error("empty_label", f"node {node.id!r} has an empty label", i)

    parent_of: dict[str, str | None] = {}
    for i, node in enumerate(nodes):
        parent = None if node.parent_id == ROOT_ID else node.parent_id
        if first[node.id] == i:
            parent_of[node.id] = parent
        if parent is not None and parent not in first:
            error("missing_parent", f"node {node.id!r} refers to missing parent {parent!r}", i)

    issues.extend(_find_cycles(parent_of, first, locs))

    seen_labels: dict[tuple[str | None, str], int] = {}
    child_count: dict[str, int] = {}
    for i, node in enumerate(nodes):
        parent = None if node.parent_id == ROOT_ID else node.parent_id
        if parent in first:
            child_count[parent] = child_count.get(parent, 0) + 1
        normalised = normalise_label(node.label)
        if not normalised:
            continue
        key = (parent, normalised)
        if key in seen_labels:
            other = nodes[seen_labels[key]]
            error(
                "duplicate_sibling_label",
                f"label {node.label!r} duplicates sibling {other.id!r} after normalisation",
                i,
            )
        else:
            seen_labels[key] = i

    for parent in first:
        if child_count.get(parent) == 1:
            issues.append(
                TaxonomyIssue(
                    "warning",
                    "single_child",
                    f"node {parent!r} has exactly one child, which adds a step with no decision",
                    locs[first[parent]],
                )
            )
    return issues


def _find_cycles(
    parent_of: dict[str, str | None], first: dict[str, int], locs: list[str | None]
) -> list[TaxonomyIssue]:
    """Walk parent pointers iteratively; report each cycle once, at its earliest member."""
    state: dict[str, int] = {}  # 1 = on the current walk, 2 = finished
    issues: list[TaxonomyIssue] = []
    for start in parent_of:
        walk: list[str] = []
        current: str | None = start
        while current is not None and current in parent_of and state.get(current, 0) == 0:
            state[current] = 1
            walk.append(current)
            current = parent_of[current]
        if current is not None and state.get(current) == 1:
            members = walk[walk.index(current) :]
            anchor = min(members, key=lambda m: first[m])
            chain = " -> ".join([*members, members[0]])
            issues.append(
                TaxonomyIssue("error", "cycle", f"parent cycle: {chain}", locs[first[anchor]])
            )
        for node_id in walk:
            state[node_id] = 2
    return issues
