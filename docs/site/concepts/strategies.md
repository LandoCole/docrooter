# Strategies

A strategy decides how a text walks down the taxonomy. Pass one to `Classifier(strategy=...)`
or `classify_document(strategy=...)`.

| Strategy | What it does | Backend calls per text | Default for |
| --- | --- | --- | --- |
| `Greedy()` | At each node, follow the most probable child. One path. | one per level | |
| `Beam(width)` | Keep the `width` best partial paths, ranked by summed log probability. | up to `width` per level | `Classifier` (`Beam(3)`) |
| `Flat()` | Score every leaf in one call, each described by its full path. `Flat(allow_internal=True)` also scores internal nodes. | one | `classify_document`, `normalize_sections` |

```python
for strategy in (Greedy(), Beam(3), Flat()):
    classifier = Classifier(taxonomy, "keyword", strategy=strategy, threshold=0.0)
    result = classifier.classify(doc)
    paths = [" > ".join(s.best.labels) for s in result.sections.values()]
    print(f"{type(strategy).__name__}: {paths}")
```

```text
Greedy: ['Financial > Tax', 'Financial > Audit']
Beam: ['Financial > Tax', 'Legal > Contracts']
Flat: ['Financial > Tax', 'Legal > Contracts']
```

The second section is a supplier contract. At the top level Greedy committed to *Financial*
and could only choose among its children; Beam kept *Legal* alive long enough to find
*Contracts*. This is the usual trade-off:

- **Greedy** is the cheapest and easiest to read, but an early mistake cannot be undone.
- **Beam** recovers from a near-miss at the top at a few times the cost. It is the default for
  section classification.
- **Flat** makes one decision over all leaves, which suits wide, shallow taxonomies such as the
  18 document types, and the section taxonomies. Trained bundles cannot use it (their heads
  decide per node).

## Thresholds and early stops

`threshold` (default 0.35 in `Classifier`, 0 in `classify_document`) is the minimum probability
to follow a child. When no child reaches it, the path stops at the current node with a
`stopped_reason`:

| `stopped_reason` | Meaning |
| --- | --- |
| `None` | The path reached a leaf |
| `"low_confidence"` | No child reached the threshold |
| `"low_margin"` | The best two children were closer than `min_margin` |
| `"max_depth"` | The path hit the depth limit (64 levels) |
| `"no_backend"` | No backend supports this node or accepts its number of candidates |
| `"empty_text"` | (document type) The document had no text to classify |

```python
strict = Classifier(taxonomy, "keyword", threshold=0.9).classify(doc)
print(strict.sections["s1.1"].best.stopped_reason)
```

```text
low_confidence
```

A stopped path is a feature: "Financial, not sure which kind" is more useful than a confident
wrong leaf. Per-node `threshold` settings in the taxonomy override the classifier's, so you can
be strict where mistakes are expensive. Run [`calibrate`](../guides/training-evaluation.md#5-calibrate)
first if you want a threshold to mean a real probability.

## Multi-label nodes

A node with `multi_label: true` follows every child that reaches the threshold, up to
`max_fanout` (default 3) of them, and backends that can score candidates independently are asked
to. The section's `selected` then holds one path per chosen branch; `best` is still the single
highest-scoring one.

The script is [`examples/strategies.py`](../../../examples/strategies.py).
