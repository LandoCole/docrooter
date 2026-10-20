"""Doc-type section rules: decide, or narrow, a document's type from its section tags.

Each doc-type leaf in ``doc_types.json`` carries a ``sections`` block (:class:`DocTypeRules`)
whose ids come from the section catalogue (``builtin_taxonomy("section_catalogue")``). An id is
a catalogue leaf (``"financial.balance_sheet"``) or a group (``"financial"``), which matches
any of its leaves.

:func:`match_rules` scores every doc type against a document's tags:

- **candidates** meet every ``requires`` group;
- **near misses** meet all but one of their groups, and at least one;
- an ``excludes`` hit removes a type, unless that would leave none in its list;
- candidates and near misses are ordered by how many ``typical`` ids they match, most first,
  then by rule order. ``typical`` orders and never gates;
- a single candidate is **decided** when every tag supporting it has confidence at least
  ``decide_conf``.

What a flow does with the result (asking a model among the candidates, adding near misses, or
falling back to every type) belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from docrooter.taxonomy import Taxonomy


class DocTypeRules(BaseModel):
    """The sections that define a doc type.

    ``requires`` is a list of groups of catalogue ids: a document must have a section tagged
    with at least one id from each group. ``typical`` ids raise a type's rank without being
    needed, and an ``excludes`` hit argues against the type.
    """

    model_config = ConfigDict(extra="forbid")

    requires: list[list[str]] = Field(min_length=1)
    typical: list[str] = Field(default_factory=list)
    excludes: list[str] = Field(default_factory=list)

    @field_validator("requires")
    @classmethod
    def _groups_not_empty(cls, groups: list[list[str]]) -> list[list[str]]:
        for i, group in enumerate(groups):
            if not group:
                raise ValueError(f"requires group {i} is empty, so it could never be met")
        return groups

    @field_validator("requires", "typical", "excludes")
    @classmethod
    def _ids_not_blank(cls, value: list) -> list:
        flat = [i for item in value for i in (item if isinstance(item, list) else [item])]
        if any(not i.strip() for i in flat):
            raise ValueError("section ids must be non-empty")
        return value


class RuleMatch(BaseModel):
    """The doc types a document's section tags support.

    ``support`` maps each candidate and near miss to the section ids that met its required
    groups, one per group (the group's strongest tag), with their confidences. ``excluded``
    lists the types an ``excludes`` hit removed.
    """

    candidates: list[str] = Field(default_factory=list)
    near_misses: list[str] = Field(default_factory=list)
    decided: str | None = None
    support: dict[str, dict[str, float]] = Field(default_factory=dict)
    excluded: list[str] = Field(default_factory=list)


def covers(rule_id: str, section_id: str) -> bool:
    """Whether a rule id (a catalogue leaf or group) matches a tagged section id."""
    return section_id == rule_id or section_id.startswith(f"{rule_id}.")


def doc_type_rules(taxonomy: Taxonomy) -> dict[str, DocTypeRules]:
    """The ``sections`` block of every leaf that has one, in taxonomy order (copies)."""
    return {
        leaf.id: leaf.sections.model_copy(deep=True)
        for leaf in taxonomy.leaves()
        if leaf.sections is not None
    }


def match_rules(
    rules: Mapping[str, DocTypeRules],
    tags: Mapping[str, float],
    *,
    min_conf: float = 0.5,
    decide_conf: float = 0.7,
) -> RuleMatch:
    """Score every doc type in ``rules`` against a document's section ``tags``.

    ``tags`` maps a catalogue id to the highest confidence it was tagged with anywhere in the
    document; a tag below ``min_conf`` counts for nothing. Never raises on missing or unknown
    tags: no evidence gives an empty match.
    """
    present = sorted((sid, conf) for sid, conf in tags.items() if conf >= min_conf)
    candidates: list[str] = []
    near_misses: list[str] = []
    support: dict[str, dict[str, float]] = {}
    typical_hits: dict[str, int] = {}
    barred: set[str] = set()

    for doc_type, rule in rules.items():
        met: dict[str, float] = {}
        missing = 0
        for group in rule.requires:
            hits = [(sid, conf) for sid, conf in present if any(covers(r, sid) for r in group)]
            if not hits:
                missing += 1
                continue
            sid, conf = max(hits, key=lambda hit: hit[1])
            met[sid] = conf
        if not met:
            continue
        support[doc_type] = met
        typical_hits[doc_type] = sum(
            any(covers(t, sid) for sid, _ in present) for t in rule.typical
        )
        if any(covers(e, sid) for e in rule.excludes for sid, _ in present):
            barred.add(doc_type)
        if missing == 0:
            candidates.append(doc_type)
        elif missing == 1:
            near_misses.append(doc_type)

    excluded: list[str] = []

    def keep(ids: list[str]) -> list[str]:
        kept = [i for i in ids if i not in barred]
        if kept:
            excluded.extend(i for i in ids if i in barred)
        else:
            kept = ids
        return sorted(kept, key=lambda i: -typical_hits[i])  # stable: ties keep rule order

    candidates = keep(candidates)
    near_misses = keep(near_misses)
    decided = None
    if len(candidates) == 1 and all(c >= decide_conf for c in support[candidates[0]].values()):
        decided = candidates[0]
    return RuleMatch(
        candidates=candidates,
        near_misses=near_misses,
        decided=decided,
        support={k: v for k, v in support.items() if k in candidates or k in near_misses},
        excluded=excluded,
    )


__all__ = ["DocTypeRules", "RuleMatch", "covers", "doc_type_rules", "match_rules"]
