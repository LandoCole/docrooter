"""Built-in taxonomies shipped as package data (JSON), loaded with ``importlib.resources``."""

from __future__ import annotations

import json
from importlib import resources

from docrooter.taxonomy import Taxonomy

_SECTION_PREFIX = "section_types/"

_SECTION_TAXONOMIES = _SECTION_PREFIX.rstrip("/")


def _resource(name: str):
    parts = name.split("/")
    if (
        not all(parts)
        or len(parts) > 2
        or (len(parts) == 2 and parts[0] != _SECTION_TAXONOMIES)
        or ".." in parts
    ):
        raise KeyError(_unknown(name))
    node = resources.files(__package__)
    for part in parts[:-1]:
        node = node.joinpath(part)
    return node.joinpath(f"{parts[-1]}.json")


def _unknown(name: str) -> str:
    return f"unknown built-in taxonomy {name!r}; available: {', '.join(list_builtin_taxonomies())}"


def list_builtin_taxonomies() -> list[str]:
    """Names accepted by :func:`builtin_taxonomy`, sorted."""
    root = resources.files(__package__)
    names = [p.name[: -len(".json")] for p in root.iterdir() if p.name.endswith(".json")]
    section_dir = root.joinpath("section_types")
    names += [
        _SECTION_PREFIX + p.name[: -len(".json")]
        for p in section_dir.iterdir()
        if p.name.endswith(".json")
    ]
    return sorted(names)


def builtin_taxonomy(name: str) -> Taxonomy:
    """Load a built-in taxonomy: ``"doc_types"`` or ``"section_types/<name>"``."""
    resource = _resource(name)
    if not resource.is_file():
        raise KeyError(_unknown(name))
    return Taxonomy.from_dict(json.loads(resource.read_text(encoding="utf-8")))


# Leaf doc-type id -> built-in section taxonomy name (a ``builtin_taxonomy`` argument).
# Static so that importing docrooter does not parse any taxonomy; tests check it covers every leaf.
DOC_TYPE_TO_SECTION_TAXONOMY: dict[str, str] = {
    "financial.annual_report": "section_types/annual_report",
    "financial.quarterly_report": "section_types/annual_report",
    "financial.shareholder_letter": "section_types/generic",
    "financial.earnings_release": "section_types/generic",
    "financial.prospectus": "section_types/generic",
    "financial.proxy_statement": "section_types/annual_report",
    "legal.contract": "section_types/contract",
    "legal.terms_policy": "section_types/generic",
    "legal.court_opinion": "section_types/generic",
    "legal.legislation": "section_types/generic",
    "research.research_paper": "section_types/research_paper",
    "research.technical_standard": "section_types/generic",
    "research.white_paper": "section_types/generic",
    "technical.user_manual": "section_types/technical_document",
    "technical.datasheet": "section_types/technical_document",
    "government.government_report": "section_types/generic",
    "corporate.sustainability_report": "section_types/generic",
    "corporate.investor_presentation": "section_types/generic",
}

__all__ = [
    "DOC_TYPE_TO_SECTION_TAXONOMY",
    "builtin_taxonomy",
    "list_builtin_taxonomies",
]
