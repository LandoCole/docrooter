"""Train per-node heads on a few labelled sections, then compare them with zero-shot.

Needs docrooter[models]. Fine-tuning the shared body is switched off to keep this quick; drop
``finetune_body=False`` to fine-tune it too (slower, usually better with enough data).
"""

import tempfile
from pathlib import Path

from docrooter import Classifier, Dataset, Taxonomy, TrainedModel, compare, load, train
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
    ],
    ("financial", "financial.audit"): [
        "The auditor issued an unqualified opinion on the consolidated statements.",
        "Management found a material weakness in internal control over reporting.",
        "Our audit committee reviewed the independent auditor's fees and findings.",
        "The auditor identified a critical audit matter concerning revenue recognition.",
    ],
    ("legal", "legal.contracts"): [
        "The master services agreement renews automatically for one-year terms.",
        "Either party may terminate the agreement with 90 days' written notice.",
        "The supply contract obliges us to purchase a minimum annual volume.",
        "The licence agreement grants an exclusive right to the patents in Europe.",
    ],
    ("legal", "legal.litigation"): [
        "The court granted the defendants' motion to dismiss the class action.",
        "A shareholder filed a lawsuit alleging misstatements in our filings.",
        "The company settled the patent infringement suit for $12 million.",
        "An appeal of the judgment is pending before the circuit court.",
    ],
}
dataset = Dataset(
    [Example(text=text, path=list(path)) for path, texts in EXAMPLES.items() for text in texts]
)
train_set, test_set = dataset.split(test_size=0.25)  # stratified by top-level category

# One head per node whose children each have enough examples; the rest fall back to zero-shot.
model = train(taxonomy, train_set, finetune_body=False, min_examples_per_child=2)

with tempfile.TemporaryDirectory() as tmp:
    model.save(Path(tmp) / "bundle")
    trained = TrainedModel.load(Path(tmp) / "bundle")  # or backend "trained:<dir>"
    comparison = compare(
        {
            "zero-shot": Classifier(taxonomy),
            "trained": Classifier(taxonomy, trained),
        },
        test_set,
    )
    comparison.print()  # a toy set this small shows the workflow, not a gain from training

# Bootstrap more labels: classify a document zero-shot, keep the rendered text and best paths,
# review them (in a JSONL or CSV file) and train again on the corrected set.
doc = load("# Notes\n\n## Tax\nThe tax rate fell.\n\n## Lawsuit\nThe court dismissed the case.\n")
zero_shot = Classifier(taxonomy)
pseudo = Dataset.from_results(zero_shot.classify(doc), doc, zero_shot.view)
for example in pseudo:
    print(example.section_id, "->", example.path)
