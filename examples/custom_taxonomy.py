"""Your own taxonomy: write it as CSV (or JSON), check it, classify with it, extend a built-in.

Needs no models (keyword backend). The script works in a scratch directory.
"""

import os
import tempfile

from docrooter import Classifier, Taxonomy, builtin_taxonomy, load

os.chdir(tempfile.mkdtemp())  # scratch directory for taxonomy.csv

# A path CSV: one row per category, labels joined by ">". Descriptions and examples are
# what zero-shot backends compare a section against, so write them for the model.
with open("taxonomy.csv", "w", encoding="utf-8") as f:
    f.write(
        "path,description,examples,threshold\n"
        'Financial,"Money, accounting, results",,\n'
        "Financial > Tax,Income tax and deferred tax,The effective tax rate was 21%,\n"
        "Financial > Audit,Audit reports and findings,The auditor issued an opinion,\n"
        "Legal,Contracts and disputes,,0.5\n"
        "Legal > Contracts,Agreements and obligations,This agreement renews yearly,\n"
        "Legal > Litigation,Lawsuits and court proceedings,The court dismissed the case,\n"
    )

taxonomy = Taxonomy.load("taxonomy.csv")  # the layout (path, levels, adjacency, JSON) is detected
for issue in taxonomy.validate():  # warnings only: errors stop the file from loading
    print(issue.severity, issue.code, issue.message)
print(len(taxonomy.nodes), "nodes, depth", taxonomy.depth())
print([node.id for node in taxonomy.leaves()])
print(taxonomy.node("legal").threshold)  # this node's own threshold beats the classifier's

doc = load("# Notes\n\n## Tax\nThe income tax rate fell.\n\n## Lawsuit\nThe court dismissed it.\n")
for section in Classifier(taxonomy, "keyword", threshold=0.0).classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)}")

# Extend a built-in taxonomy: start from its dict form, add a node, build a new one.
data = builtin_taxonomy("doc_types").to_dict()
data["nodes"].append(
    {
        "id": "internal",
        "label": "Internal",
        "description": "Internal company documents such as memos and meeting minutes",
    }
)
extended = Taxonomy.from_dict(data)
print([node.label for node in extended.children()])
