"""Classify sections with the default backend, UltraBERT embeddings (needs docrooter[models]).

The first run downloads the model (about 600 MB) and runs its pinned remote code; see the README.
"""

from docrooter import Beam, Classifier, Taxonomy, load

MARKDOWN = """\
# Annual report

## Income taxes
The effective tax rate was 21.4 percent, down from 23.1 percent a year earlier.

## Independent auditor's opinion
We audited the consolidated statements and issued an unqualified opinion.

## Lease agreements
The master services agreement renews automatically unless either party gives notice.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "description": "Accounting, taxes and audits",
                "children": [
                    {"id": "tax", "label": "Tax", "description": "Income tax and deferred tax"},
                    {"id": "audit", "label": "Audit", "description": "Audit reports and opinions"},
                ],
            },
            {"id": "legal", "label": "Legal", "description": "Contracts and legal proceedings"},
        ]
    }
)

# With no backend argument, Classifier uses "embedding:dmis-lab/UltraBERT-Embed-150M".
# Beam(3) and threshold 0.35 are the defaults, spelled out here.
classifier = Classifier(taxonomy, strategy=Beam(3), threshold=0.35)
result = classifier.classify(load(MARKDOWN))
for section in result.sections.values():
    best = section.best
    print(f"{section.title}: {' > '.join(best.labels)} ({best.score:.2f})")
