# Custom taxonomies

The built-in taxonomies cover document types and common section types. For anything else
(your product lines, your risk categories, your contract clause library) write your own
taxonomy in JSON or CSV. This guide builds one, checks it, classifies with it, and extends a
built-in one. The script is [`examples/custom_taxonomy.py`](../../../examples/custom_taxonomy.py).

## 1. Write it

A spreadsheet is often the easiest place to start. Export it as CSV with a `path` column (labels
joined by `>`), plus any of `description`, `examples` (separated by `|`), `threshold`,
`multi_label` and `backend`:

```csv
path,description,examples,threshold
Financial,"Money, accounting, results",,
Financial > Tax,Income tax and deferred tax,The effective tax rate was 21%,
Financial > Audit,Audit reports and findings,The auditor issued an opinion,
Legal,Contracts and disputes,,0.5
Legal > Contracts,Agreements and obligations,This agreement renews yearly,
Legal > Litigation,Lawsuits and court proceedings,The court dismissed the case,
```

The same tree can be written as nested JSON, flat JSON with `parent_id`, an adjacency CSV
(`id,parent_id,label`) or a levels CSV (`level_1,level_2,...`). The
[taxonomy formats reference](../reference/taxonomy-formats.md) shows all five side by side.

## 2. Check it

```bash
docrooter taxonomy validate taxonomy.csv
```

```text
valid: 6 nodes, depth 2
```

Errors (a duplicate id, a cycle, a missing parent, two siblings with the same label, an
unreadable row) are reported with the CSV row or JSON pointer and exit status 1. Warnings, such
as a node with a single child, are reported but allowed.

## 3. Load and use it

```python
taxonomy = Taxonomy.load("taxonomy.csv")  # the layout (path, levels, adjacency, JSON) is detected
for issue in taxonomy.validate():  # warnings only: errors stop the file from loading
    print(issue.severity, issue.code, issue.message)
print(len(taxonomy.nodes), "nodes, depth", taxonomy.depth())
print([node.id for node in taxonomy.leaves()])
print(taxonomy.node("legal").threshold)  # this node's own threshold beats the classifier's

doc = load("# Notes\n\n## Tax\nThe income tax rate fell.\n\n## Lawsuit\nThe court dismissed it.\n")
for section in Classifier(taxonomy, "keyword", threshold=0.0).classify(doc).sections.values():
    print(f"{section.title}: {' > '.join(section.best.labels)}")
```

```text
6 nodes, depth 2
['financial.tax', 'financial.audit', 'legal.contracts', 'legal.litigation']
0.5
Tax: Financial > Tax
Lawsuit: Legal > Litigation
```

Ids come from the slugged labels (`Legal > Litigation` becomes `legal.litigation`); give an
`id` column (adjacency CSV or JSON) if you need ids that survive a label change. On the CLI, any
command that takes `-t` accepts the file: `docrooter classify report.pdf -t taxonomy.csv`.

## 4. Extend a built-in taxonomy

Start from the built-in's dict form, add or change nodes, and build a new taxonomy:

```python
data = builtin_taxonomy("doc_types").to_dict()
extended = Taxonomy.from_dict(data)
print([node.label for node in extended.children()])
```

```text
['Financial', 'Legal', 'Research', 'Technical', 'Government', 'Corporate', 'Internal']
```

(The script appends an `internal` group between those lines.) Building `doc_types` logs one
warning, because its `government` group has a single child; that is expected.

## Writing nodes that classify well

Zero-shot backends see only the text of each node, so the taxonomy *is* the model's
instructions:

- **Describe the content.** "Income tax expense, deferred tax assets and the effective tax rate"
  works better than "Tax matters".
- **Make siblings distinguishable.** Two children whose descriptions share most of their words
  cannot be separated by any backend.
- **Add real headings as `examples`.** For section-level taxonomies, list the headings you see
  in practice; they help most where headings are formulaic (filings, contracts).
- **Keep the tree shallow.** Each level is another decision that can go wrong; three clear
  levels beat six fuzzy ones.
- **Measure.** Label 50-100 sections, run `docrooter eval`, and change the descriptions where
  the confusions are; see [Training and evaluation](training-evaluation.md). Per-node
  `threshold` and `backend` settings let you be strict, or use a stronger backend, only where
  it matters.
