"""Fields: named, query-independent needs tied to element topic and function tags (spec §4).

A field says which parts of a document a reader needs, such as an annual report's risk factors
or a fund's fees. Its ``topics`` are ``element_topics`` leaf ids, all in one family, and define
relevance: an element is relevant to a field when its topic is one of them. Its ``functions``
are ``element_functions`` tags that only strengthen the ranking. ``query`` is the field as a
question, for the embedding term.

Built-in sets ship as package data in ``docrooter/fields/<family>.json``. A field set file is
``{"family": ..., "fields": [...]}`` or a bare list of fields.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from docrooter.errors import FieldError
from docrooter.taxonomy import Taxonomy

__all__ = [
    "Field",
    "builtin_fields",
    "calibration_of",
    "field_fingerprint",
    "list_builtin_field_sets",
    "load_fields",
    "validate_fields",
]


class Field(BaseModel):
    """A named need: the element topics that carry it and the functions that strengthen it."""

    model_config = ConfigDict(extra="forbid")

    id: str  # "<family>.<name>", e.g. "annual_report.risk_factors"
    label: str
    description: str
    topics: list[str]  # element_topics leaf ids, one family
    functions: list[str] = []  # element_functions tags, e.g. "risk_disclosure"
    query: str | None = None  # the natural-language form, for the embedding term

    @field_validator("id", "label", "description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("topics")
    @classmethod
    def _some_topic(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("a field needs at least one topic")
        return value

    @property
    def family(self) -> str:
        """The element topic family: the first segment of the topic ids."""
        return self.topics[0].split(".", 1)[0]


def field_fingerprint(field: Field) -> str:
    """A short fingerprint of what makes a field a field to the selector: its sorted topics and
    functions, and its need text (its ``query``, else its ``description``). The dial is
    calibrated on relevance, which topics define, on the ranking the functions strengthen, and
    on ``sim``, which compares elements with the need text; the label does not enter it, nor
    the description of a field that has a query."""
    canonical = json.dumps(
        {
            "topics": sorted(field.topics),
            "functions": sorted(field.functions),
            "need": field.query or field.description,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def calibration_of(fields: Sequence[Field]) -> dict[str, dict[str, str]]:
    """The calibrated-fields record of a field set: family -> {field id: fingerprint}."""
    out: dict[str, dict[str, str]] = {}
    for field in fields:
        out.setdefault(field.family, {})[field.id] = field_fingerprint(field)
    return out


def _parse(data: Any, source: str) -> list[Field]:
    family = None
    if isinstance(data, dict):
        unknown = sorted(set(data) - {"family", "fields"})
        if unknown:
            raise FieldError(f"{source}: unknown keys {unknown}; expected 'family' and 'fields'")
        family, data = data.get("family"), data.get("fields")
    if not isinstance(data, list):
        raise FieldError(f"{source}: expected a list of fields or an object with 'fields'")
    try:
        fields = [Field.model_validate(item) for item in data]
    except ValidationError as exc:
        raise FieldError(f"{source}: invalid field: {exc}", [str(exc)]) from exc
    stray = [f.id for f in fields if family is not None and f.family != family]
    if stray:
        raise FieldError(f"{source}: fields outside family {family!r}: {stray}", stray)
    return fields


def load_fields(path: str | Path) -> list[Field]:
    """Read a field set from a JSON file. It is not checked against the taxonomies: call
    :func:`validate_fields` for that."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FieldError(f"{path}: cannot read a field set: {exc}") from exc
    return _parse(data, str(path))


def list_builtin_field_sets() -> list[str]:
    """The families with a built-in field set, sorted."""
    root = resources.files("docrooter").joinpath("fields")
    return sorted(p.name[: -len(".json")] for p in root.iterdir() if p.name.endswith(".json"))


def builtin_fields(family: str) -> list[Field]:
    """The built-in field set of an element topic family (``annual_report``, ``fund_document``
    or ``contract``). Every call returns new objects."""
    available = list_builtin_field_sets()
    if family not in available:
        raise FieldError(f"no built-in field set for {family!r}; available: {', '.join(available)}")
    resource = resources.files("docrooter").joinpath("fields").joinpath(f"{family}.json")
    return _parse(json.loads(resource.read_text(encoding="utf-8")), f"built-in {family}")


def validate_fields(
    fields: Sequence[Field], topics_taxonomy: Taxonomy, functions_taxonomy: Taxonomy
) -> None:
    """Check a field set against the topic and function taxonomies.

    Every topic must be a leaf of ``topics_taxonomy``, and every function a node of
    ``functions_taxonomy``. A field's topics, and all the fields of a set, must belong to one
    family (a top-level node of the topic tree), and each id must be ``"<family>.<name>"``.
    The set must not be empty, ids must be unique, and a field may not list a topic or function
    twice. A field may include its family's catch-all ``<family>.other`` leaf, but not as its
    only topic: "everything that fits nowhere else" is not a need. Raises
    :class:`~docrooter.errors.FieldError` listing every problem.
    """
    problems: list[str] = []
    if not fields:
        problems.append("empty field set: a set needs at least one field")
    set_family: str | None = None
    seen: set[str] = set()
    for field in fields:
        if field.id in seen:
            problems.append(f"{field.id}: duplicate field id")
        seen.add(field.id)
        for kind, values in (("topic", field.topics), ("function", field.functions)):
            repeated = sorted({v for v in values if values.count(v) > 1})
            if repeated:
                problems.append(f"{field.id}: duplicate {kind}s {repeated}")
        families: set[str] = set()
        other_only = True
        for topic in field.topics:
            if topic not in topics_taxonomy:
                problems.append(f"{field.id}: unknown topic {topic!r}")
                other_only = False
                continue
            if topics_taxonomy.children(topic):
                problems.append(f"{field.id}: topic {topic!r} is not a leaf")
            top = topics_taxonomy.path(topic)[0].id
            families.add(top)
            other_only = other_only and topic == f"{top}.other"
        if families and other_only:
            problems.append(f"{field.id}: its only topics are catch-all '.other' leaves")
        if len(families) > 1:
            problems.append(f"{field.id}: topics are in more than one family: {sorted(families)}")
        elif families:
            family = families.pop()
            if not field.id.startswith(f"{family}."):
                problems.append(f"{field.id}: id must start with its family, {family + '.'!r}")
            if set_family is None:
                set_family = family
            elif family != set_family:
                problems.append(
                    f"{field.id}: family {family!r} differs from the set's {set_family!r}"
                )
        for function in field.functions:
            if function not in functions_taxonomy:
                problems.append(f"{field.id}: unknown function {function!r}")
    if problems:
        shown = "; ".join(problems[:3])
        more = f" (+{len(problems) - 3} more)" if len(problems) > 3 else ""
        raise FieldError(f"invalid field set, {len(problems)} problem(s): {shown}{more}", problems)
