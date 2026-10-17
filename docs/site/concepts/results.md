# Results

Every result is a pydantic model: it prints, compares, and serialises to JSON with
`model_dump_json()` (or `to_json()`), and reads back with `model_validate_json()`. Results keep
every decision, so you can see why a label was chosen and how close the alternatives were.

## Section results: `DocumentResult`

`Classifier.classify(doc)` returns a `DocumentResult`: `source` and `sections`, a dict from
section id to `SectionResult`.

```python
result = Classifier(taxonomy, "keyword", threshold=0.0).classify(doc)
section = result.sections["s1.1"]
best = section.best
print(best.node_ids, best.labels, round(best.score, 3), best.stopped_reason)
for step in best.steps:
    print(f"  at {step.node_id}: chose {step.chosen} from {step.probs}")
print("alternatives:", [" > ".join(path.labels) for path in section.alternatives])
```

```text
['financial', 'tax'] ['Financial', 'Tax'] 0.571 None
  at __root__: chose ['financial'] from {'financial': 0.6421700617614096, 'legal': 0.3578299382385904}
  at financial: chose ['tax'] from {'tax': 0.8887890798973598, 'audit': 0.11121092010264023}
alternatives: ['Legal > Contracts', 'Legal > Litigation']
```

| `SectionResult` field | Meaning |
| --- | --- |
| `best` | The highest-scoring `PathResult` (or `None` if nothing was scored) |
| `alternatives` | The rival paths that competed with it |
| `selected` | The paths chosen together at a multi-label node (`[best]` otherwise) |
| `title`, `section_id`, `n_tokens`, `truncated` | The section, the size of its view, and whether the view was cut |

| `PathResult` field | Meaning |
| --- | --- |
| `node_ids`, `labels` | The path from the top level down |
| `score` | The product of the chosen children's probabilities |
| `steps` | One `Step` per decision: the parent `node_id`, the `chosen` child, the top `probs` and the `backend` that decided |
| `stopped_reason` | `None` at a leaf, otherwise why the path stopped (see [Strategies](strategies.md#thresholds-and-early-stops)) |

`result.summary(depth=1)` gives the share of the document, weighted by tokens, in each category
at a depth (`{'Financial': 0.52, 'Legal': 0.48}` above). `result.to_dataframe()` gives one row
per section (needs `pandas`, extra `bench`).

## Document-type results: `DocTypeResult`

`classify_document` and `Classifier.classify_documents` return one per document.

| Field | Meaning |
| --- | --- |
| `best` | The chosen path, e.g. `node_ids == ["financial", "financial.annual_report"]`. Empty with `stopped_reason="empty_text"` for a document with no text, or `"low_confidence"` below the threshold |
| `confidence` | The chosen type's probability: the number to gate a pipeline on |
| `leaf_probs` | The ten most probable types, most probable first |
| `view_text` | The rendered `DocumentView` the backend saw |
| `timings` | `{"classify": seconds}`, amortised over a batch |
| `mode` | `"view"`, `"sections"` or `"hybrid"` |

`classify_document` uses a threshold of 0, so a document with any text always gets a type;
decide yourself what confidence is enough for your pipeline.

## Normalisation results: `NormalizationResult`

`normalize_sections` returns the `doc_type` it used (and `doc_type_confidence` when it classified
the type itself), the section `taxonomy` name, and `sections`: section id to `SectionNorm` with
`title`, `canonical` (the node id, e.g. `risk_factors`), `label` and `score`.

## Evaluation reports

`evaluate()` and `compare()` return reports with exact-path accuracy, hierarchical
precision/recall/F1, leaf macro-F1, the stop rate, selective accuracy and throughput; see
[Training and evaluation](../guides/training-evaluation.md).

The [API reference](../reference/api.md#results) lists every field.
