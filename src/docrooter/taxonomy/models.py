"""Taxonomy data model: a flat id -> node map with iterative traversal."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    model_serializer,
)

from docrooter.errors import TaxonomyError
from docrooter.rules import DocTypeRules
from docrooter.taxonomy.validate import ROOT_ID, TaxonomyIssue, validate_nodes


class TaxonomyNode(BaseModel):
    """One category. In a built :class:`Taxonomy`, ``children`` holds child ids.

    ``keywords`` (optional) holds class terms, such as those mined from a corpus. An empty list is
    left out of every dump, so a taxonomy without keywords dumps and hashes exactly as before the
    field existed (trained bundles store that hash).

    ``sections`` (optional, doc-type leaves) holds the section rules that define the type; see
    :mod:`docrooter.rules`. It is left out of a dump when unset, and out of
    :meth:`Taxonomy.content_hash` always: rules decide among the labels, they do not change them.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    description: str | None = None
    examples: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    multi_label: bool = False
    threshold: float | None = None
    backend: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)
    sections: DocTypeRules | None = None

    @model_serializer(mode="wrap")
    def _omit_unset_extras(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.keywords:
            data.pop("keywords", None)
        if self.sections is None:
            data.pop("sections", None)
        return data


ROOT_SETTINGS = ("multi_label", "threshold", "backend", "description")
_ROOT_DEFAULTS: dict[str, Any] = {
    "multi_label": False,
    "threshold": None,
    "backend": None,
    "description": None,
}


def check_root(root: Any, *, location: str | None = None) -> dict[str, Any]:
    """Validate root-level settings and return the non-default ones.

    Accepted keys are ``multi_label``, ``threshold``, ``backend`` and ``description``.
    Anything else, or a bad value, raises ``TaxonomyError`` with code ``invalid_root``.
    """

    def fail(message: str) -> TaxonomyError:
        return TaxonomyError(message, [TaxonomyIssue("error", "invalid_root", message, location)])

    if not isinstance(root, dict):
        raise fail("root settings must be an object")
    unknown = sorted(str(k) for k in root if k not in ROOT_SETTINGS)
    if unknown:
        raise fail(
            f"unknown root setting(s) {', '.join(unknown)}; allowed: {', '.join(ROOT_SETTINGS)}"
        )
    try:
        probe = TaxonomyNode(id=ROOT_ID, label="(root)", **root)
    except ValidationError as exc:
        detail = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        raise fail(f"invalid root settings: {detail}") from exc
    return {k: v for k in ROOT_SETTINGS if (v := getattr(probe, k)) != _ROOT_DEFAULTS[k]}


class Taxonomy:
    """A validated hierarchy of categories.

    ``nodes`` maps id to node for the real nodes only; the virtual root is never stored there.
    ``node(ROOT_ID)`` returns a synthetic root node (label ``"(root)"``, ``children`` = top-level
    ids in input order) carrying the optional ``root`` decision settings, and
    ``children(ROOT_ID)`` (the default) returns the top-level nodes. Child lists and parent links
    are derived from ``parent_id`` in input order. All traversal is iterative.
    """

    ROOT_ID = ROOT_ID

    def __init__(
        self,
        nodes: Sequence[TaxonomyNode],
        *,
        locations: Sequence[str | None] | None = None,
        root: dict[str, Any] | None = None,
    ):
        root_settings = check_root(root) if root is not None else {}
        nodes = list(nodes)
        issues = validate_nodes(nodes, locations=locations)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            shown = "; ".join(_describe(i) for i in errors[:3])
            more = f" (+{len(errors) - 3} more)" if len(errors) > 3 else ""
            raise TaxonomyError(f"invalid taxonomy, {len(errors)} error(s): {shown}{more}", issues)
        self.warnings: list[TaxonomyIssue] = [i for i in issues if i.severity == "warning"]

        child_ids: dict[str, list[str]] = {ROOT_ID: [], **{n.id: [] for n in nodes}}
        for n in nodes:
            child_ids[ROOT_ID if n.parent_id in (None, ROOT_ID) else n.parent_id].append(n.id)
        self.nodes: dict[str, TaxonomyNode] = {
            n.id: n.model_copy(
                update={
                    "children": child_ids[n.id],
                    "parent_id": None if n.parent_id == ROOT_ID else n.parent_id,
                }
            )
            for n in nodes
        }
        self._children = child_ids
        self._root_settings = root_settings
        self._root = TaxonomyNode(
            id=ROOT_ID, label="(root)", children=list(child_ids[ROOT_ID]), **root_settings
        )
        self._depth: dict[str, int] = {ROOT_ID: 0}
        queue = deque([ROOT_ID])
        while queue:
            current = queue.popleft()
            for child in child_ids[current]:
                self._depth[child] = self._depth[current] + 1
                queue.append(child)

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def node(self, id: str) -> TaxonomyNode:
        return self._root if id == ROOT_ID else self.nodes[id]

    def children(self, node_id: str = ROOT_ID) -> list[TaxonomyNode]:
        return [self.nodes[c] for c in self._children[node_id]]

    def path(self, node_id: str) -> list[TaxonomyNode]:
        """Nodes from the top level down to ``node_id`` (root excluded); ``[]`` for the root."""
        if node_id == ROOT_ID:
            return []
        chain = [self.nodes[node_id]]
        while chain[-1].parent_id is not None:
            chain.append(self.nodes[chain[-1].parent_id])
        return chain[::-1]

    def node_depth(self, node_id: str) -> int:
        """Depth of a node; the virtual root is 0 and top-level nodes are 1."""
        return self._depth[node_id]

    def depth(self) -> int:
        """Number of nodes on the longest root-to-leaf path."""
        return max(self._depth.values())

    def leaves(self) -> list[TaxonomyNode]:
        return [n for n in self.nodes.values() if not n.children]

    def validate(self) -> list[TaxonomyIssue]:
        return validate_nodes(list(self.nodes.values()))

    def resolve_path(self, parts: list[str]) -> list[str]:
        """Resolve ids or case-insensitive labels, one per level, to a list of node ids."""
        resolved: list[str] = []
        current = ROOT_ID
        for part in parts:
            options = self.children(current)
            match = next((n for n in options if n.id == part), None)
            if match is None:
                wanted = part.strip().casefold()
                match = next((n for n in options if n.label.strip().casefold() == wanted), None)
            if match is None:
                valid = ", ".join(f"{n.id} ({n.label})" for n in options) or "none (leaf node)"
                where = "the top level" if current == ROOT_ID else f"node {current!r}"
                raise KeyError(f"{part!r} not found under {where}; valid options: {valid}")
            resolved.append(match.id)
            current = match.id
        return resolved

    def to_dict(self) -> dict[str, Any]:
        """Flat representation that :meth:`from_dict` reads back."""
        data: dict[str, Any] = {"nodes": [n.model_dump() for n in self.nodes.values()]}
        if self._root_settings:
            data["root"] = dict(self._root_settings)
        return data

    def content_hash(self) -> str:
        """sha256 of the canonical JSON; equal for the same tree whatever format it came from.

        Covers every node, the root settings and the order of the top-level nodes. Doc-type
        ``sections`` rules are left out, so adding or editing them keeps trained bundles valid.
        """
        dumps = [n.model_dump() for n in self.nodes.values()]
        for dump in dumps:
            dump.pop("sections", None)
        ordered = sorted(dumps, key=lambda d: d["id"])
        payload = {"nodes": ordered, "root": self._root_settings, "top": self._children[ROOT_ID]}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Any) -> Taxonomy:
        from docrooter.taxonomy.io import from_dict

        return from_dict(data)

    @classmethod
    def load(cls, path: str | Path, *, sep: str = ">") -> Taxonomy:
        """Load a JSON or CSV taxonomy file (format auto-detected). Raises on errors."""
        from docrooter.taxonomy.io import load

        return load(path, sep=sep)


def _describe(issue: TaxonomyIssue) -> str:
    where = f" at {issue.location}" if issue.location else ""
    return f"{issue.code}{where}: {issue.message}"
