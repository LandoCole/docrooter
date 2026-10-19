"""Generate ``src/docrooter/taxonomies/element_topics.json`` from the section taxonomies.

The element topic tree is root -> family -> leaf. Each family is a section taxonomy
(``ELEMENT_FAMILIES``); its leaves are that taxonomy's nodes, reused unchanged under the ids
``f"{family}.{section_leaf}"``, plus ``f"{family}.other"`` when the section taxonomy has no
``other`` leaf of its own. The family descriptions and the added ``other`` leaves are written
here; the document types of each family come from ``DOC_TYPE_TO_SECTION_TAXONOMY``.

    PYTHONPATH=src .venv/bin/python scripts/build_element_topics.py          # write the file
    PYTHONPATH=src .venv/bin/python scripts/build_element_topics.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from docrooter.taxonomies import (
    DOC_TYPE_TO_SECTION_TAXONOMY,
    ELEMENT_FAMILIES,
    builtin_taxonomy,
)

OUTPUT = Path(__file__).resolve().parents[1] / "src/docrooter/taxonomies/element_topics.json"

ROOT_DESCRIPTION = (
    "What a document element is about. Hierarchical and single-label: the family is the kind of "
    "document, given by its document type, and the leaf is the part of that document the element "
    "belongs to, or other."
)

FAMILY_LABELS = {
    "annual_report": "Annual report",
    "fund_document": "Fund document",
    "contract": "Contract",
    "research_paper": "Research paper",
    "technical_document": "Technical document",
    "generic": "Generic document",
}

FAMILY_DESCRIPTIONS = {
    "annual_report": (
        "Elements of a company's periodic reports to shareholders and of its proxy statements, "
        "whose topics are the parts of such a report, from the shareholder letter and "
        "management's discussion to the financial statements and governance."
    ),
    "fund_document": (
        "Elements of fund and securities-offering documents such as prospectuses and fact "
        "sheets, whose topics are the parts of a fund disclosure, from the investment objective "
        "and fees to performance and holdings."
    ),
    "contract": (
        "Elements of a legally binding agreement between named parties, whose topics are its "
        "clauses, from the parties and recitals to the signatures and exhibits."
    ),
    "research_paper": (
        "Elements of a scholarly research paper, whose topics are the parts of a paper, from "
        "the abstract and method to the results and references."
    ),
    "technical_document": (
        "Elements of product and technical documentation such as user manuals and datasheets, "
        "whose topics range from installation and configuration to specifications and "
        "troubleshooting."
    ),
    "generic": (
        "Elements of any other kind of document, whose topics are the structural parts of a "
        "document: front matter, summary, introduction, body, conclusion, references, and "
        "appendix."
    ),
}

# The ``other`` leaf of each family whose section taxonomy has none, de-overlapped against
# that taxonomy's leaves.
OTHER_LEAVES = {
    "contract": (
        "Content of a contract that fits none of the other categories, such as the cover page, "
        "table of contents, version or draft markings, and running headers and footers.",
        ["Table of Contents", "Contents", "Cover Page", "Execution Version"],
    ),
    "research_paper": (
        "Content of a research paper that fits none of the other categories, such as the title, "
        "author names and affiliations, keywords, correspondence details, and running headers "
        "and footers.",
        ["Keywords", "Author Information", "Affiliations", "Correspondence"],
    ),
    "technical_document": (
        "Content of a technical document that fits none of the other categories, such as cover "
        "pages, tables of contents, safety warnings, contact details, and running headers and "
        "footers.",
        [
            "Table of Contents",
            "Contents",
            "Safety Information",
            "Important Safety Instructions",
            "Contact Information",
        ],
    ),
    "generic": (
        "Content that is not part of the document's substance and fits none of the other "
        "categories, such as distribution lists, document-control blocks, standalone "
        "disclaimers, contact details, and running headers and footers.",
        ["Distribution List", "Document Control", "Disclaimer", "Contact Information"],
    ),
}


def _doc_type_labels(family: str) -> list[str]:
    doc_types = builtin_taxonomy("doc_types")
    wanted = f"section_types/{family}"
    return [
        doc_types.node(doc_type).label
        for doc_type, name in DOC_TYPE_TO_SECTION_TAXONOMY.items()
        if name == wanted
    ]


def build() -> dict:
    """The element topic tree in the nested JSON format of the other built-in taxonomies."""
    families = []
    for family in ELEMENT_FAMILIES:
        section = builtin_taxonomy(f"section_types/{family}")
        leaves = [
            {
                "id": f"{family}.{node.id}",
                "label": node.label,
                "description": node.description,
                "examples": list(node.examples),
                "children": [],
            }
            for node in section.nodes.values()
        ]
        if "other" not in section.nodes:
            description, examples = OTHER_LEAVES[family]
            leaves.append(
                {
                    "id": f"{family}.other",
                    "label": "Other",
                    "description": description,
                    "examples": examples,
                    "children": [],
                }
            )
        doc_types = ", ".join(_doc_type_labels(family))
        families.append(
            {
                "id": family,
                "label": FAMILY_LABELS[family],
                "description": f"{FAMILY_DESCRIPTIONS[family]} Document types: {doc_types}.",
                "children": leaves,
            }
        )
    return {"root": {"description": ROOT_DESCRIPTION}, "nodes": families}


def render() -> str:
    return json.dumps(build(), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if current != text:
            print(f"{args.output} is stale; re-run scripts/build_element_topics.py")
            return 1
        return 0
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
