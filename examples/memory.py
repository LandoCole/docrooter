"""Bounded memory: process-wide limits, then a classifier that releases its model when done.

Needs docrooter[models]; the first run downloads UltraBERT-Embed-150M (see the README).
"""

import docrooter
from docrooter import Taxonomy, load

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "tax", "label": "Tax", "description": "Income tax and deferred tax"},
            {"id": "audit", "label": "Audit", "description": "Audit reports and opinions"},
        ]
    }
)
doc = load("# Income taxes\n\nThe effective tax rate was 21.4 percent.\n")

# or set DOCROOTER_MEMORY_BUDGET_GB=8 and DOCROOTER_THREADS=4
docrooter.configure(memory_budget_gb=8, threads=4)

with docrooter.Classifier(taxonomy) as clf:  # the model is released when the block ends
    results = clf.classify(doc)

for section in results.sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels) if section.best else '-'}")
