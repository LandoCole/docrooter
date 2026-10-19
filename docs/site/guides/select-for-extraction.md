# Select what to send to an LLM

Reading a whole 80-page filing to answer one question about its risk factors costs tokens and
dilutes the answer. `select()` returns only the parts of a document that matter for a **need**,
at a **completeness** you choose, and tells you how many tokens that is.

```text
 report.pdf ──► process() ──► select(need, completeness) ──► Selection.chunks ──► your LLM
                elements,     which elements to read,         the text to put      your prompt,
                tags          how much it costs               in the prompt        your extraction
```

**Who does what.** docrooter selects. Your application extracts: it writes the prompt, calls the
model and reads the answer. docrooter never calls an LLM for this. Read
[The completeness dial](../concepts/completeness-dial.md) first if you want to know exactly what
the number `c` promises, because the answer is more careful than "90%".

## Before you start

- **A tagger and a selector.** Selection scores each element with the [element tags](../concepts/element-tags.md)
  of a [tagger you trained](train-your-tagger.md) and fits its dial on a calibration set. You
  need a bundle directory with a `selector.json` next to it (or anywhere, passed as
  `--selector`). **There is no built-in selector**: without one, only the whole document
  (`c = 1.0`) can be selected, and `select()` below 1.0 says how to fit one. The
  [last section](#refit-on-your-own-corpus) does that.
- **Extras.** `pip install 'docrooter[models,layout]'` for the tagger and the embedding model.
- **One model at a time, within 8 GiB.** Selecting loads the student and the embedder, as
  [element tagging](../concepts/element-tags.md) does.

The Python example [`examples/select_for_extraction.py`](../../../examples/select_for_extraction.py)
needs none of this. It runs `select()` with keyword stand-ins and a hand-set selector, so you can
see what a Selection holds; its dial is **not calibrated**, so its "guarantee" means nothing.

## 1. Pick a field, or write your own

A **field** is a named target such as the risk factors of an annual report. Three sets ship with
docrooter, 15 fields for `annual_report`, 12 for `fund_document` and 15 for `contract`:

```python
from docrooter import builtin_fields

for field in builtin_fields("fund_document"):
    print(field.id, "-", field.label)
```

On the command line, `--need` takes a field id. Repeat it for several fields of one family: they
are read as one need.

```bash
docrooter select report.pdf --need annual_report.risk_factors --tagger bundles/tagger-a
docrooter select report.pdf --need annual_report.risk_factors --need annual_report.legal_proceedings --tagger bundles/tagger-a
```

Any other `--need` value is a **free-text query**, such as `--need "interest rate risk"`. A query
works, but it never carries the guarantee (see [below](#3-read-the-selection)).

**Your own field** is a JSON file. A field names the `element_topics` leaves that carry it, all in
one family; `functions` and `query` are optional:

```json
{
  "family": "annual_report",
  "fields": [
    {
      "id": "annual_report.litigation_and_regulation",
      "label": "Litigation and regulation",
      "description": "Lawsuits, investigations and regulatory actions that involve the company.",
      "topics": ["annual_report.legal_proceedings", "annual_report.risk_factors"],
      "functions": ["risk_disclosure"],
      "query": "What legal and regulatory actions is the company facing?"
    }
  ]
}
```

Check it against the taxonomies before you use it. `validate_fields` lists every problem at once:
an unknown or non-leaf topic, topics in more than one family, an id that does not start with its
family, a field whose only topic is a catch-all `.other` leaf, and duplicates.

```python
from docrooter import builtin_taxonomy, load_fields
from docrooter.select import validate_fields

fields = load_fields("my_fields.json")
topics = builtin_taxonomy("element_topics")
functions = builtin_taxonomy("element_functions")
validate_fields(fields, topics, functions)
```

Then pass a `Field` as the need: `select(analysis, fields[0], ...)`. Read
[Refit on your own corpus](#refit-on-your-own-corpus) before you rely on the guarantee for it.

## 2. Choose a completeness, or a budget

`completeness` is the share of the relevant content you want to keep, on average over documents
like the calibration set.

| `completeness` | Use it when |
| --- | --- |
| `0.8`, Lean | Tokens matter more than the last few relevant passages |
| `0.9`, Balanced | The default |
| `0.98`, Thorough | A miss is costly |
| `1.0`, Whole document | Always everything; no dial, no model needed |

The playground has these four presets, and moving its slider only re-applies the threshold: the
document is tagged and embedded once.

A **budget** (`budget=4000` in Python, `--budget 4000` on the command line) replaces the dial. It
fills the budget with the most relevant elements and reports the recall it **expects** at that
cut, an estimate, never a guarantee. Use it when your context window is the hard limit. It
cannot be combined with `c = 1.0`.

## 3. Read the Selection

```python
from docrooter import select
from docrooter.select.selection import readout

selection = select(analysis, field, completeness=0.9, selector="bundles/tagger-a")
print(readout(selection))
```

```text
≈ 90% of relevant content kept · 3,240 tokens (12%) · guaranteed on calibration-like documents
```

(The numbers are the shape of the line, not a result.) The line is built from the Selection's
fields. These matter before you trust it:

| You see | It means | What to do |
| --- | --- | --- |
| `guarantee=True` | The dial's guarantee applies: on documents like its calibration set, the mean share of **teacher-relevant** elements kept is at least `c` | Rely on it on, and only on, such documents |
| `guarantee=False`, "(estimate)" | The recall is the model's own estimate | Treat as an estimate; raise `c` if a miss is costly |
| `capped=True` | The calibration set was too small to reach this `c`, so the **whole document** is returned | Fine, just costlier; fit with more documents to uncap it |
| `doc_type_mismatch=True` | The document's type maps to another family than the field's, for example an annual-report field on a prospectus. The elements were re-tagged for the field's family; the guarantee is dropped | Check the document type, or accept an estimate |
| `over_budget=True` | The most relevant element alone is over your budget | Raise the budget |
| `expected_recall=None` | No relevant content is expected in this document | Nothing to send; do not read it as 0 or 1 |

A query, a missing tagger or embedder, and a budget also give `guarantee=False`.
[The completeness dial](../concepts/completeness-dial.md#when-there-is-no-guarantee) lists every
case and why.

**What the guarantee is against.** It is calibrated against **teacher relevance**: a local LLM
labelled the topic of each element, and "relevant" means its topic is one of the field's. It is
not calibrated against human labels. It is also an average: at `c = 0.9` some documents keep less
than 90%.

## 4. Pass the chunks to your LLM

`selection.chunks` are the selected elements as [chunks](chunking-for-rag.md), in reading order,
with contiguous runs merged and each starting with its heading path. Put their text in your
prompt:

```python
context = "\n\n".join(chunk.text for chunk in selection.chunks)  # send it to your own model
```

`selection.element_ids` has the elements, in reading order, if you want to cite them or look up
their pages and boxes in `analysis.document.elements`. Expansion adds context around each
selected element (the section's heading and first paragraph, the preceding element, a table's
caption), so `tokens` is what you pay, not the sum of the relevant elements alone.

On the shell the same thing prints the readout and then one line per selected element; `--json`
prints the whole Selection:

```bash
docrooter select report.pdf --need annual_report.risk_factors --completeness 0.9 --tagger bundles/tagger-a --json
```

The output of the example, which needs no model:

```text
≈ 90% of relevant content kept · 41 tokens (43%) · guaranteed on calibration-like documents
guarantee=True capped=False
elements: h1, p2, p3, p4

Risk factors

Interest rate risk could raise our borrowing costs.

A shortage of components is a supply risk for our two plants.

Currency risk follows from sales in six currencies.

≈ 100% of relevant content kept · 95 tokens (100%) · guaranteed on calibration-like documents
```

## In the playground

`docrooter serve --element-tags BUNDLE --selector SELECTOR` (or a `selector.json` inside the
bundle) adds a panel to a run in element mode: a field picker or a free-text query, a
completeness slider with the four presets, and a live readout with badges for "estimate",
"dial capped by calibration size" and "doc type mismatch". The Elements tab highlights the
selected elements, and **Show selected only** hides the rest. See [the playground](playground.md).

## Refit on your own corpus

The shipped fields and a selector fitted elsewhere are calibrated on someone else's documents.
For your own, fit your own. The steps need a corpus and a tagger from
[Train your own element tagger](train-your-tagger.md), and they run one model at a time:

```bash
# 1. The teacher labels the topic of EVERY content element of the calibration documents.
docrooter teach label corpus/ --full --splits calib
# 2. Fit the relevance models on half of them, and the dials on the other half.
docrooter fit-selector corpus/ --bundle bundles/tagger-a --out bundles/tagger-a/selector.json
```

`fit-selector` reads each calibration PDF again exactly as the corpus build read it, so pass the
build's `--profile` and `--max-pages` if they were not the defaults (`accurate`, 30). It refuses a
corpus whose teacher labels are missing or do not match the document text, so it cannot calibrate
on a reading that diverged. Its summary lists the number of units behind each family's dial: if a
family has few, expect `capped=True` at high `c`.

**For a field set of your own:**

1. Write the JSON and check it with `validate_fields` (step 1).
2. Include it when you fit, from Python (`fit-selector` on the shell uses the built-in sets), so the dial is calibrated on units like yours:

    ```python
    from docrooter.corpus import Corpus
    from docrooter.select import fit_selector

    selector = fit_selector(
        Corpus("corpus/"),
        "bundles/tagger-a",
        "embedding:dmis-lab/UltraBERT-Embed-150M",
        fields_by_family={"annual_report": fields},
        pdf_dir="corpus/pdfs",
    )
    selector.save("bundles/tagger-a/selector.json")
    ```

3. `fields_by_family` replaces the built-in sets, so list every field you want in the family.

A field you did not fit with still **works**, since the combiner scores any field from the same
features, but `select()` cannot tell: it reports `guarantee=True` for any field of a family that has a dial.
Whether that is true for your field depends on it resembling the fields the dial was calibrated
on, and on your documents resembling the calibration documents. Check recall on a few documents
you have labelled yourself.

## Measuring it

`benchmarks/selection_bench.py` measures the realised recall against the promised one and the
tokens against baselines, and `benchmarks/evidence_qa.py` scores evidence recall on a public QA
set; neither is part of the wheel. The method and the (pending) results are on
[Selection (completeness dial)](../benchmarks/selection.md). `benchmarks/extraction/` is a demo
harness that plays the application: it asks an LLM your question with the selected context and
with the whole document and compares the answers. It is optional, outside the wheel, and runs
only when you configure an OpenAI-compatible endpoint or a local MLX model; Jev is never used for
it, because Jev is a classifier.

## Next

- [The completeness dial](../concepts/completeness-dial.md): the rule, the guarantee and its limits.
- [Chunking for RAG](chunking-for-rag.md): what the chunks are.
- [Element tags](../concepts/element-tags.md): the topics and functions fields are made of.
