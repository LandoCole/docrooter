"""A custom backend: any object with these members works; there is nothing to subclass."""

import numpy as np

from docrooter import Classifier, Taxonomy
from docrooter.backends import Scores


class OverlapBackend:
    name = "overlap"  # keys calibrated temperatures; keep it unique per backend
    max_candidates = None  # or a limit: larger candidate sets go to the next backend

    def score(self, texts, candidates, *, instruction=None):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            rows.append([len(words & set(c.text.lower().split())) for c in candidates])
        return Scores(np.array(rows, dtype=float), "similarity")

    def supports(self, node_id):
        return True

    def tokenizer(self):
        return None  # token counts are estimated


taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "tax", "label": "Tax", "description": "income tax rate"},
            {"id": "audit", "label": "Audit", "description": "auditor opinion"},
        ]
    }
)

classifier = Classifier(taxonomy, OverlapBackend(), threshold=0.0)
for result in classifier.classify_texts(["The income tax rate rose.", "The auditor opinion."]):
    print(" > ".join(result.best.labels))
