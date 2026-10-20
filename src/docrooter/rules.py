"""Doc-type section rules: decide, or narrow, a document's type from its section tags.

Each doc-type leaf in ``doc_types.json`` carries a ``sections`` block (:class:`DocTypeRules`)
whose ids come from the section catalogue (``builtin_taxonomy("section_catalogue")``). An id is
a catalogue leaf (``"financial.balance_sheet"``) or a group (``"financial"``), which matches
any of its leaves.

:func:`match_rules` scores every doc type against a document's tags:

- **candidates** meet every ``requires`` group;
- **near misses** meet all but one of their groups, and at least one;
- an ``excludes`` hit removes a candidate, unless that would leave no candidate; then the
  hit is waived and flagged, and a waived candidate is never decided. An excludes hit always
  removes a near miss;
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

from pydantic import BaseModel, Field

from docrooter.taxonomy.models import DocTypeRules

if TYPE_CHECKING:
    from docrooter.taxonomy import Taxonomy


class RuleMatch(BaseModel):
    """The doc types a document's section tags support.

    ``support`` maps each candidate and near miss to the section ids that met its required
    groups, one per group (the group's strongest tag), with their confidences. ``excluded``
    lists the types an ``excludes`` hit removed.

    When every candidate hit an exclude, the candidates are kept rather than emptied: they are
    in ``waived`` and ``excludes_waived`` is true. The rules' own evidence argued against them,
    so a waived candidate is never ``decided``, and the caller should offer the waived
    candidates to the model; :func:`docrooter.classify` adds the near misses only when a single
    waived candidate is left, as for any lone undecided candidate.
    """

    candidates: list[str] = Field(default_factory=list)
    near_misses: list[str] = Field(default_factory=list)
    decided: str | None = None
    support: dict[str, dict[str, float]] = Field(default_factory=dict)
    excluded: list[str] = Field(default_factory=list)
    waived: list[str] = Field(default_factory=list)
    excludes_waived: bool = False


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

    def ranked(ids: list[str]) -> list[str]:
        return sorted(ids, key=lambda i: -typical_hits[i])  # stable: ties keep rule order

    excluded = [i for i in candidates + near_misses if i in barred]
    waived: list[str] = []
    if candidates and all(i in barred for i in candidates):
        # every candidate hit an exclude: keep them, never an empty list, but flag it
        waived = list(candidates)
        excluded = [i for i in excluded if i not in waived]
    else:
        candidates = [i for i in candidates if i not in barred]
    candidates = ranked(candidates)
    near_misses = ranked([i for i in near_misses if i not in barred])
    decided = None
    if (
        len(candidates) == 1
        and candidates[0] not in waived
        and all(c >= decide_conf for c in support[candidates[0]].values())
    ):
        decided = candidates[0]
    return RuleMatch(
        candidates=candidates,
        near_misses=near_misses,
        decided=decided,
        support={k: v for k, v in support.items() if k in candidates or k in near_misses},
        excluded=excluded,
        waived=waived,
        excludes_waived=bool(waived),
    )


def unknown_section_ids(
    rules: Mapping[str, DocTypeRules], catalogue: Taxonomy
) -> dict[str, list[str]]:
    """The ids each rule block names that are not in ``catalogue``, such as typos in a custom
    doc-type taxonomy. A rule id that matches nothing never fires, so check it at load time.
    Doc types with no unknown ids are left out."""
    unknown = {
        doc_type: [i for i in dict.fromkeys(rule.ids()) if i not in catalogue]
        for doc_type, rule in rules.items()
    }
    return {doc_type: ids for doc_type, ids in unknown.items() if ids}


__all__ = [
    "DocTypeRules",
    "RuleMatch",
    "covers",
    "doc_type_rules",
    "match_rules",
    "unknown_section_ids",
]
