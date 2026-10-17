# Training and evaluation

Zero-shot is a good start. When it is not enough and you have labelled sections, docrooter fits
one small classifier head per taxonomy node on top of a shared embedding model (few-shot heads),
and can fine-tune that model first. Nodes without enough examples keep using the zero-shot
model. Every step has a CLI equivalent.

Training needs `docrooter[models]`. The full script is
[`examples/train_and_eval.py`](../../../examples/train_and_eval.py).

## 1. Labelled data

A dataset is JSONL (one `{"text": ..., "path": [...]}` per line) or CSV (`text` and `path`
columns, the path joined by `>`). A path lists node ids or labels from the top down; labels
match case-insensitively.

```json
{"text": "The effective tax rate was 21.4 percent.", "path": ["Financial", "Tax"]}
{"text": "The auditor issued an unqualified opinion.", "path": ["financial", "financial.audit"]}
```

**Bootstrap labels from a zero-shot run.** Classify some documents, keep the rendered text of
each section with its best path, then fix the mistakes by hand:

```python
classifier = Classifier(taxonomy)  # zero-shot
result = classifier.classify(doc)
pseudo = Dataset.from_results(result, doc, classifier.view)
pseudo.save("pseudo.jsonl")  # review and correct, then load it back
```

## 2. Split

```python
dataset = Dataset.load("labelled.jsonl")
train_set, test_set = dataset.split(test_size=0.2)  # stratified by top-level category
train_set, dev_set = train_set.split(test_size=0.2)  # a development set for calibration
```

Keep the test set out of every decision (descriptions, thresholds, which model to use) or its
score stops being an estimate.

## 3. Train

```python
model = train(taxonomy, train_set, finetune_body=False, min_examples_per_child=2)
model.save("bundle/")
```

- **Few-shot heads** (`finetune_body=False`): a logistic-regression head per node on the frozen
  embeddings. Fast (minutes on a laptop CPU), and enough for many taxonomies.
- **Fine-tuning** (the default, `finetune_body=True`): the embedding body is first trained
  contrastively on triplets built from your examples, then the heads are fitted. Slower (about
  18 minutes on Apple MPS and 24 on CPU for 306 examples), usually better with enough data.
- A node gets a head only when each of its children has at least `min_examples_per_child`
  examples (default 4). The others defer to the zero-shot body.
- Memory stays bounded: texts are cut to `max_seq_len=512` tokens and fine-tuning uses gradient
  checkpointing; see [Memory and performance](memory-performance.md).

```bash
docrooter train -t taxonomy.json -d labelled.jsonl -o bundle/ --no-finetune-body --min-examples 2
```

!!! warning "Trained bundles contain pickles"
    A bundle stores its heads as `joblib` files, which are Python pickles, and loading a pickle
    can run arbitrary code. Only load bundles you trained yourself or got from a source you
    trust. The same goes for `trained:<path>` specs and `docrooter eval -b trained:...`.

## 4. Evaluate

```python
trained = Classifier(taxonomy, TrainedModel.load("bundle/"))  # or backend "trained:bundle/"
compare({"zero-shot": Classifier(taxonomy), "trained": trained}, test_set).print()
evaluate(trained, test_set).print()  # adds per-node accuracy and sibling confusions
```

```bash
docrooter eval -t taxonomy.json -d test.jsonl -b trained:bundle/ --format md
```

Reports give:

| Metric | Meaning |
| --- | --- |
| Exact | The predicted path equals the gold path |
| hP / hR / hF1 | Hierarchical precision, recall and F1: partial credit for getting the upper levels right |
| Macro-F1 | Leaf F1 averaged over classes, so small classes count |
| Stop rate | Share of sections that stopped early (below the threshold) |
| Selective accuracy | Accuracy on the sections that did not stop |
| Sections/s | Throughput |

`evaluate` adds per-node decision accuracy and the sibling pairs that are confused most, which
tell you which descriptions or examples to improve. Reports print as tables, or come out with
`to_markdown()` and `to_json()`.

## 5. Calibrate

Model scores are not probabilities until they are calibrated. `calibrate` returns a copy of the
classifier whose backend temperatures are tuned on a development set, so that the
probabilities, and with them the `threshold`, mean what they say:

```python
calibrated = calibrate(trained, dev_set)
print(calibrated.calibration_report)
```

Use a development set that was not used for training, and not the test set.

## When does training help?

From the committed results (`benchmarks/results/2026-10-16/few-shot.json` and
`zero-shot.json`, hierarchical F1 on 1,000 test sections, UltraBERT body):

| Dataset | Zero-shot | 4 per leaf | 16 per leaf | 16 per leaf, fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| B1: 10-K items (EDGAR-CORPUS) | 0.752 | 0.800 | 0.888 | 0.936 |
| B2: DBpedia classes (3 levels) | 0.172 | 0.786 | 0.922 | |
| B3: LEDGAR clauses (100 classes) | 0.381 | 0.660 | 0.719 | |

On these section-level datasets, a few labelled examples per class beat zero-shot by a wide
margin, especially where the categories are not self-describing (DBpedia).

It did **not** help for document types on our PDF corpora: with only 2-4 documents per type, a
head trained on the 51 development documents scored 0.675 on the held-out set, 15 points
*below* zero-shot UltraBERT (0.825). It learned publisher and layout cues that did not transfer.
Train when you have many examples per class from the same distribution you will classify;
otherwise improve the taxonomy's descriptions first.
