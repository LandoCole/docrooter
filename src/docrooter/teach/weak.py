"""Structural weak labels (spec §7.3): topic evidence from an element's headings alone.

Two sources, each recorded with its name and never mixed into the teacher's labels:

- ``section_norm`` (source ``heading_synonym``): the element's heading path, deepest heading
  first, matched against the ``label`` and heading synonyms (``examples``) of its family's
  topic leaves. An exact match after normalisation (case, numbering such as "Item 1A." or
  "3.2", punctuation, stop words, plural "s") has confidence 1.0; otherwise the confidence is
  the word-set overlap (Jaccard). The first heading whose best leaf reaches 0.5, and is not
  tied with another leaf, gives the label. The spec's source is tier-1 section normalisation;
  the corpus store holds no section labels (the build switches classification off), so this
  is its model-free stand-in.
- ``edgar_item``: a Form 10-K heading "Item N" in an ``annual_report``-family document,
  mapped to its topic leaf by :data:`EDGAR_10K_ITEMS`. Quarterly reports (10-Q items number
  differently) and items with no clear leaf give none.
"""

from __future__ import annotations

import re
import weakref
from typing import Any

from docrooter.corpus import CorpusElement, ManifestRecord
from docrooter.taxonomy import Taxonomy
from docrooter.teach.shortlist import _topics, doc_family, family_leaves

__all__ = [
    "EDGAR_10K_ITEMS",
    "MIN_SECTION_CONFIDENCE",
    "heading_tokens",
    "weak_labels",
]

MIN_SECTION_CONFIDENCE = 0.5

# Form 10-K item -> annual_report leaf. Items whose content has no clear leaf (1B unresolved
# staff comments, 1C cybersecurity, 4 mine safety, 6 [reserved], 9 changes in accountants,
# 9B other information, 9C foreign jurisdictions, 16 summary) are left out. Part III items 12-14
# (ownership, relationships and independence, accountant fees) are governance disclosures.
EDGAR_10K_ITEMS: dict[str, str] = {
    "1": "business_overview",
    "1A": "risk_factors",
    "2": "properties",
    "3": "legal_proceedings",
    "5": "market_for_equity",
    "7": "management_discussion",
    "7A": "management_discussion",
    "8": "financial_statements",
    "9A": "controls_and_procedures",
    "10": "corporate_governance",
    "11": "executive_compensation",
    "12": "corporate_governance",
    "13": "corporate_governance",
    "14": "corporate_governance",
    "15": "exhibits",
}
_EDGAR_FAMILY = "annual_report"
_QUARTERLY = "financial.quarterly_report"

_ITEM = re.compile(r"^\s*item\s+(\d{1,2}[a-c]?)(?![0-9a-z])", re.IGNORECASE)
# Leading numbering: "Item 1A.", "Note 3:", "Part II", "Article 4 -", "3.2", "IV.", "A)".
_NUMBERING = re.compile(
    r"^\s*(?:(?:item|note|part|section|chapter|article)\s+[0-9ivxlc]+[a-z]?(?![0-9a-z])"
    r"|\d+(?:\.\d+)*\.?(?=\s)|[ivxlc]+[.)]|[a-z][.)])\s*[.:)\-–—]*\s*",
    re.IGNORECASE,
)
_STOP = frozenset("a an and as at by for from in into of on or our the their to with its".split())


def heading_tokens(heading: str) -> frozenset[str]:
    """A heading's words for matching: numbering stripped, case folded, ``&`` read as "and",
    stop words and one-letter words dropped, a plural "s" cut."""
    text = heading
    for _ in range(3):  # "Part II Item 7." has two numberings
        stripped = _NUMBERING.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    words = re.findall(r"[a-z0-9]+", text.casefold().replace("&", " and "))
    return frozenset(_stem(w) for w in words if len(w) > 1 and w not in _STOP)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


_Synonyms = tuple[tuple[str, str, frozenset[str]], ...]
# Per taxonomy object (weakly held) and family: the synonyms' tokens, computed once.
_SYNONYMS: weakref.WeakKeyDictionary[Taxonomy, dict[str, _Synonyms]] = weakref.WeakKeyDictionary()


def _synonyms(taxonomy: Taxonomy, family: str) -> _Synonyms:
    """``(leaf id, synonym, its tokens)`` for every leaf label and example of ``family``."""
    per_family = _SYNONYMS.setdefault(taxonomy, {})
    if family not in per_family:
        out = []
        for leaf in family_leaves(family, taxonomy):
            for synonym in [leaf.label, *leaf.examples]:
                tokens = heading_tokens(synonym)
                if tokens:
                    out.append((leaf.id, synonym, tokens))
        per_family[family] = tuple(out)
    return per_family[family]


def _match(heading: str, synonyms: _Synonyms) -> tuple[str, str, float] | None:
    """``(leaf id, synonym, confidence)`` of the best unambiguous match of ``heading``."""
    tokens = heading_tokens(heading)
    if not tokens:
        return None
    best: dict[str, tuple[float, str]] = {}
    for leaf_id, synonym, syn_tokens in synonyms:
        score = len(tokens & syn_tokens) / len(tokens | syn_tokens)
        if score > best.get(leaf_id, (0.0, ""))[0]:
            best[leaf_id] = (score, synonym)
    if not best:
        return None
    top = max(score for score, _ in best.values())
    leaders = [leaf for leaf, (score, _) in best.items() if score == top]
    if top < MIN_SECTION_CONFIDENCE or len(leaders) > 1:
        return None
    return leaders[0], best[leaders[0]][1], top


def _section_norm(headings: list[str], family: str, taxonomy: Taxonomy) -> dict[str, Any] | None:
    synonyms = _synonyms(taxonomy, family)
    for heading in reversed(headings):
        found = _match(heading, synonyms)
        if found is not None:
            leaf_id, synonym, confidence = found
            return {
                "label": leaf_id,
                "confidence": round(confidence, 4),
                "source": "heading_synonym",
                "heading": heading,
                "matched": synonym,
            }
    return None


def _edgar_item(headings: list[str], family: str) -> dict[str, Any] | None:
    for heading in reversed(headings):
        match = _ITEM.match(heading)
        if match is None:
            continue
        item = match.group(1).upper()
        leaf = EDGAR_10K_ITEMS.get(item)
        if leaf is None:
            return None  # the nearest item heading has no topic; an outer one is not its topic
        return {
            "label": f"{family}.{leaf}",
            "item": item,
            "source": "edgar_item",
            "heading": heading,
        }
    return None


def weak_labels(
    element: CorpusElement, doc: ManifestRecord, *, taxonomy: Taxonomy | None = None
) -> dict[str, dict[str, Any]]:
    """The weak topic labels of ``element`` of document ``doc``, by kind:

    - ``"section_norm"``: ``{"label", "confidence" (>= 0.5), "source": "heading_synonym",
      "heading", "matched"}``;
    - ``"edgar_item"``: ``{"label", "item", "source": "edgar_item", "heading"}``.

    A kind is absent when its source gives nothing. ``taxonomy`` is the topic taxonomy
    (default: the built-in ``element_topics``).
    """
    taxonomy = taxonomy if taxonomy is not None else _topics()
    family = doc_family(doc)
    headings = [h for h in element.heading_path if h.strip()]
    if not headings and element.section_title:
        headings = [element.section_title]
    out: dict[str, dict[str, Any]] = {}
    if not headings:
        return out
    norm = _section_norm(headings, family, taxonomy)
    if norm is not None:
        out["section_norm"] = norm
    if family == _EDGAR_FAMILY and doc.doc_type_hint != _QUARTERLY:
        item = _edgar_item(headings, family)
        if item is not None and item["label"] in taxonomy:
            out["edgar_item"] = item
    return out
