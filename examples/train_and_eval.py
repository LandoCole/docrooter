"""Train per-node heads on labelled sections, evaluate them against zero-shot, and calibrate.

Needs docrooter[models]. Fine-tuning the shared body is switched off to keep this quick; drop
``finetune_body=False`` to fine-tune it too (slower, usually better with enough data).
On a toy set this small expect noise, not a gain from training.
The script works in a scratch directory, so the files it writes (labelled.jsonl, bundle/,
pseudo.jsonl) vanish with it.
"""

import os
import tempfile

from docrooter import (
    Beam,
    Classifier,
    Dataset,
    Taxonomy,
    TrainedModel,
    calibrate,
    compare,
    evaluate,
    load,
    train,
)
from docrooter.train import Example

taxonomy = Taxonomy.from_dict(
    {
        "nodes": [
            {
                "id": "financial",
                "label": "Financial",
                "children": [
                    {"id": "financial.tax", "label": "Tax"},
                    {"id": "financial.audit", "label": "Audit"},
                ],
            },
            {
                "id": "legal",
                "label": "Legal",
                "children": [
                    {"id": "legal.contracts", "label": "Contracts"},
                    {"id": "legal.litigation", "label": "Litigation"},
                ],
            },
        ]
    }
)

EXAMPLES = {
    ("financial", "financial.tax"): [
        "Deferred income tax liabilities rose because of temporary differences.",
        "The effective tax rate was 21.4 percent, against 23.1 percent last year.",
        "We recorded a valuation allowance against foreign tax credit carryforwards.",
        "Cash paid for income taxes totalled $41 million in fiscal 2020.",
        "Unrecognized tax benefits increased following the examination in Germany.",
        "The statutory federal income tax rate fell to 21 percent after the reform.",
    ],
    ("financial", "financial.audit"): [
        "The auditor issued an unqualified opinion on the consolidated statements.",
        "Management found a material weakness in internal control over reporting.",
        "Our audit committee reviewed the independent auditor's fees and findings.",
        "The auditor identified a critical audit matter concerning revenue recognition.",
        "The firm has served as the company's auditor since 1998.",
        "The audit report states that the statements present fairly, in all respects.",
    ],
    ("legal", "legal.contracts"): [
        "The master services agreement renews automatically for one-year terms.",
        "Either party may terminate the agreement with 90 days' written notice.",
        "The supply contract obliges us to purchase a minimum annual volume.",
        "The licence agreement grants an exclusive right to the patents in Europe.",
        "This agreement is governed by the laws of the State of New York.",
        "The lease obligates the company to minimum annual payments through 2031.",
    ],
    ("legal", "legal.litigation"): [
        "The court granted the defendants' motion to dismiss the class action.",
        "A shareholder filed a lawsuit alleging misstatements in our filings.",
        "The company settled the patent infringement suit for $12 million.",
        "An appeal of the judgment is pending before the circuit court.",
        "The plaintiffs seek damages in an amount to be determined at trial.",
        "A jury returned a verdict in favour of the company in March.",
    ],
}

os.chdir(tempfile.mkdtemp())  # scratch directory for the files below

# Labelled data lives in a JSONL (or CSV) file: one text and its path through the taxonomy.
Dataset(
    [Example(text=text, path=list(path)) for path, texts in EXAMPLES.items() for text in texts]
).save("labelled.jsonl")

dataset = Dataset.load("labelled.jsonl")
train_set, test_set = dataset.split(test_size=0.2)  # stratified by top-level category
train_set, dev_set = train_set.split(test_size=0.2)  # a development set for calibration

# One head per node whose children each have enough examples; the rest fall back to zero-shot.
model = train(taxonomy, train_set, finetune_body=False, min_examples_per_child=2)
model.save("bundle/")

trained = Classifier(taxonomy, TrainedModel.load("bundle/"))  # or backend "trained:bundle/"
compare({"zero-shot": Classifier(taxonomy), "trained": trained}, test_set).print()
evaluate(trained, test_set).print()  # adds per-node accuracy and sibling confusions

# Tune the backend temperatures on the development set so probabilities match accuracy.
calibrated = calibrate(trained, dev_set)
print(calibrated.calibration_report)

# The defaults of Classifier(taxonomy), spelled out.
classifier = Classifier(taxonomy, strategy=Beam(3), threshold=0.35)

# Bootstrap more labels: classify a document zero-shot, keep the rendered text and best paths,
# review them (in a JSONL or CSV file) and train again on the corrected set.
doc = load("# Notes\n\n## Tax\nThe tax rate fell.\n\n## Lawsuit\nThe court dismissed the case.\n")
classifier = Classifier(taxonomy)  # zero-shot
result = classifier.classify(doc)
pseudo = Dataset.from_results(result, doc, classifier.view)
pseudo.save("pseudo.jsonl")  # review and correct, then load it back
for example in pseudo:
    print(example.section_id, "->", example.path)
