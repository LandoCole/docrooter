# Taxonomies

A taxonomy is the set of categories docrooter chooses from: a tree of nodes, each with a `label`
and, ideally, a `description` and a few `examples`. Classification walks it from the top: first
"Financial or Legal?", then "Tax or Audit?" under Financial, and so on.

```python
taxonomy = Taxonomy.load("taxonomy.csv")  # the layout (path, levels, adjacency, JSON) is detected
print(len(taxonomy.nodes), "nodes, depth", taxonomy.depth())
print([node.id for node in taxonomy.leaves()])
```

## Nodes

| Field | Meaning |
| --- | --- |
| `id` | Stable identifier, e.g. `financial.tax`. Defaults to the slugged label path. |
| `label` | The name shown in results. |
| `description` | What belongs here. Zero-shot backends compare the text against it, so write it for the model: the words a matching section would use. |
| `examples` | Example texts (or heading synonyms). Backends that take examples use them. |
| `threshold` | Minimum probability to follow a child of this node; overrides the classifier's. |
| `multi_label` | Whether several children may be chosen at this node. |
| `backend` | A backend spec for the decisions among this node's children. |
| `metadata` | Anything you want carried along; docrooter ignores it. |

The [taxonomy formats reference](../reference/taxonomy-formats.md) shows the five file layouts
(nested JSON, flat JSON, and adjacency, levels and path CSV) and every validation check.

## Built-in taxonomies

Three families ship with the package, as JSON package data. `builtin_taxonomy(name)` returns a
fresh copy; `list_builtin_taxonomies()` lists them; on the CLI they are `builtin:<name>`.

```python
doc_types = builtin_taxonomy("doc_types")
print(", ".join(list_builtin_taxonomies()))
```

```text
doc_types, element_functions, element_topics, section_catalogue, section_types/annual_report, section_types/contract, section_types/fund_document, section_types/generic, section_types/research_paper, section_types/technical_document
```

**`doc_types`**: 19 document types in 6 groups. Every node has a definition written for the
model.

| Group | Types |
| --- | --- |
| `financial` | `annual_report`, `quarterly_report`, `shareholder_letter`, `earnings_release`, `prospectus`, `proxy_statement`, `fact_sheet` |
| `legal` | `contract`, `terms_policy`, `court_opinion`, `legislation` |
| `research` | `research_paper`, `technical_standard`, `white_paper` |
| `technical` | `user_manual`, `datasheet` |
| `government` | `government_report` |
| `corporate` | `sustainability_report`, `investor_presentation` |

Leaf ids are prefixed by their group: `financial.annual_report`, `legal.contract`.
`financial.fact_sheet` (new in v0.5) is a short periodic summary of one fund or share class:
objective, fund facts, fees, performance and top holdings as of a date. It is not a full
prospectus and not an annual report. The benchmark corpora were labelled with 18 types, before it
existed, so their numbers do not include it.

**`section_catalogue`**: the section tags `classify` uses, 108 leaves in 12 groups plus `none`,
shared by every kind of document. A tag normalises what a heading means, so the same kind of
section can be found across documents however each one words it: "Statement of financial
position" and "Consolidated balance sheets" are both `financial.balance_sheet`; "Audit opinion"
and "Report of independent registered public accounting firm" are both
`financial.auditors_report`. Generic structure that nobody looks for across documents (a
contents page, an introduction, a summary, background, a conclusion, "Part II") is `none`.
Groups: `research`, `narrative`, `financial`, `disclosure`, `esg`, `fund`, `offering`,
`contract`, `government` (audit and oversight reports: highlights, objectives and methodology,
findings, recommendations, agency comments), `law`, `technical` and `standard`; each has an
`other` leaf for its own domain. To tag structure too (summary, background, conclusion), pass
your own catalogue with `classify(..., catalogue=...)`.

**`section_types/*`**: canonical sections for a family of documents, used by
[section normalisation](../guides/section-normalisation.md). Each document type maps to one:

| Section taxonomy | Used for |
| --- | --- |
| `section_types/annual_report` | annual reports, quarterly reports, proxy statements |
| `section_types/fund_document` | prospectuses and fact sheets: objective, fees, strategies, risks, performance, holdings, management, purchase and sale, tax |
| `section_types/contract` | contracts |
| `section_types/research_paper` | research papers |
| `section_types/technical_document` | user manuals, datasheets |
| `section_types/generic` | every other type, and unknown ones |

**`element_topics`** and **`element_functions`** (new in v0.5) tag elements rather than sections.
`element_topics` is one tree, root, then a document family (the section taxonomy above, plus an
`other` leaf), then the leaf; `element_functions` is a flat list of twelve multi-label tags. Both
are described on [Element tags](element-tags.md).

## Validation

Loading a file validates it. Errors (duplicate ids, cycles, a missing parent, two siblings with
the same label) stop it from loading and name the row or JSON pointer; warnings (a node with a
single child) are logged and listed in `taxonomy.validate()`. Check a file from the shell
before you use it:

```bash
docrooter taxonomy validate taxonomy.csv
```

## Writing good descriptions

Zero-shot backends never see your intentions, only the text of each node. A few rules carry
most of the accuracy:

- Describe the content, not the category: "Income tax expense, deferred tax assets, the
  effective tax rate" beats "Tax matters".
- Keep siblings distinguishable. If two nodes' descriptions share most of their words, the model
  cannot separate them either.
- Add `examples` with the headings you actually see ("Item 1A. Risk Factors", "Principal
  risks"). The built-in section taxonomies list such heading synonyms on every node, which is
  what lets normalisation classify on the heading alone.
- Prefer a shallow tree with clear splits over a deep one: every level is another decision that
  can go wrong.

See [Custom taxonomies](../guides/custom-taxonomies.md) for a worked example.
