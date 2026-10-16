"""Classify the sections of a Markdown document. Needs no models: uses the keyword backend."""

from docrooter import Classifier, Taxonomy, load

MARKDOWN = """\
# Report

## Income tax
The tax rate rose.

## Auditor's report
An audit opinion.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "description": "Tax rates, audit opinions and financial results",
                "children": [
                    {"id": "tax", "label": "Tax", "description": "Income tax and the tax rate"},
                    {"id": "audit", "label": "Audit", "description": "The audit opinion"},
                ],
            },
            {"id": "legal", "label": "Legal", "description": "Contracts, courts and disputes"},
        ]
    }
)

doc = load(MARKDOWN)
classifier = Classifier(taxonomy, "keyword", threshold=0.0)  # the default backend is a model
for section in classifier.classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)}")
