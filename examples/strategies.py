"""Strategies and results: how a section walks the taxonomy, and what comes back.

Needs no models (keyword backend), so the scores are word-overlap scores, not real
probabilities of a model.
"""

from docrooter import Beam, Classifier, Flat, Greedy, Taxonomy, load

MARKDOWN = """\
# Report

## Income tax
The income tax rate rose and deferred tax grew.

## Supplier contract
The supply agreement renews each year unless terminated.
"""

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "description": "Tax, audit and financial results",
                "children": [
                    {"id": "tax", "label": "Tax", "description": "Income tax and deferred tax"},
                    {"id": "audit", "label": "Audit", "description": "The audit opinion"},
                ],
            },
            {
                "id": "legal",
                "label": "Legal",
                "description": "Contracts, agreements and disputes",
                "children": [
                    {"id": "contracts", "label": "Contracts", "description": "Supply agreement"},
                    {"id": "litigation", "label": "Litigation", "description": "Court cases"},
                ],
            },
        ]
    }
)
doc = load(MARKDOWN)

# Greedy follows the best child; Beam(n) keeps n paths; Flat scores every leaf at once.
for strategy in (Greedy(), Beam(3), Flat()):
    classifier = Classifier(taxonomy, "keyword", strategy=strategy, threshold=0.0)
    result = classifier.classify(doc)
    paths = [" > ".join(s.best.labels) for s in result.sections.values()]
    print(f"{type(strategy).__name__}: {paths}")

# A result per section: the best path, the decisions behind it, and the rival paths.
result = Classifier(taxonomy, "keyword", threshold=0.0).classify(doc)
section = result.sections["s1.1"]
best = section.best
print(best.node_ids, best.labels, round(best.score, 3), best.stopped_reason)
for step in best.steps:
    print(f"  at {step.node_id}: chose {step.chosen} from {step.probs}")
print("alternatives:", [" > ".join(path.labels) for path in section.alternatives])

# The share of the document (by tokens) in each top-level category, and JSON out.
print(result.summary())
print(result.to_json()[:60])

# A threshold stops a path early when no child is probable enough.
strict = Classifier(taxonomy, "keyword", threshold=0.9).classify(doc)
print(strict.sections["s1.1"].best.stopped_reason)
