# The completeness dial

A long document costs tokens, and most of it is not what your LLM was asked about. Version 0.6
adds **selection**: for a **need** (the risk factors of an annual report, a fund's fees, or a
question in your own words), docrooter says which elements to read, and a **completeness** `c`
between 0 and 1 sets how much of the relevant content you want to keep.

- `c = 1.0` is the whole document. Nothing is ever missed.
- A lower `c` sends fewer tokens and accepts that some relevant content is left out.

docrooter only **selects**. It does not prompt an LLM or extract anything: your application reads
the chosen chunks and does that, as the [guide](../guides/select-for-extraction.md) shows. This
page is about what the number `c` promises, and where it promises nothing.

## What a need is

| Need | Python | Relevance | The guarantee |
| --- | --- | --- | --- |
| A **field** | `Field`, or a list of fields of one family | Elements whose topic is one of the field's topics | Possible (see below) |
| A **query** | a `str` | The embedding similarity of each element to the text, with a tag prior | Never |

A **field** is a named, query-independent target tied to [element tags](element-tags.md): its
`topics` are `element_topics` leaves, and an element is relevant to the field when its topic is
one of them. Its `functions` (such as `risk_disclosure`) only strengthen the ranking, and its
optional `query` is the field as a question, for the embedding term. Three built-in sets ship in
the package: `annual_report` (15 fields), `fund_document` (12) and `contract` (15). Some fields
are one topic leaf; a few cross cutting ones gather several, such as `contract.core_obligations`
or `fund_document.costs_and_returns`. A list of fields of one family is one need: the union of
their topics.

Each content element (paragraph, list item, table, caption, footnote, formula) gets a
probability P(relevant) from a small logistic regression over a few features:

| Feature | What it is |
| --- | --- |
| `p_topic` | The tagger's probability on the need's topics (0 when the element has no tags) |
| `p_func` | The highest tagger probability over the need's functions |
| `sim` | The cosine between the element's heading path and text and the need text, from an embedding model |
| `section_hit` | Whether the element's section normalises to one of the need's topics |
| `kind`, `log_tokens` | The element's kind and its size |

Headings are read and counted but never scored: they arrive as context.

## The rule: conformal risk control

The dial turns `c` into a threshold on P(relevant): elements scoring at least the threshold are
selected. The threshold comes from **conformal risk control** (Angelopoulos et al., 2022), fitted
once on a calibration set, so that the share of relevant content you keep is what you asked for.

For a calibration unit *d* (a document and a field, with at least one relevant element), the loss
at a threshold λ is the share of its relevant elements that are missed,

L_d(λ) = 1 − |selected_λ ∩ relevant_d| / |relevant_d|.

A lower λ selects more, so L can only rise with λ. For a target `c`, write α = 1 − c and let *n*
be the number of calibration units. The threshold is

λ̂(c) = max { λ : (n/(n+1)) · mean L_d(λ) + 1/(n+1) ≤ α }.

The result is the guarantee

> E[recall] ≥ c on new documents that are exchangeable with the calibration set.

There is one dial for each element **family** (`annual_report`, `fund_document`, `contract`),
calibrated on that family's units. The table of thresholds is precomputed on a grid of `c`
(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99), and a value between grid points is evaluated
exactly from the stored risk curve, so moving the slider never loosens the guarantee.

### What `c` means

`c` is an **average over documents** like the calibration set. It is not a promise about any one
document: at `c = 0.9` some documents will keep less than 90% of their relevant elements and some
will keep more, and the mean is at least 0.9. For a document where a miss is costly, use a higher
`c`, or `c = 1.0`.

### The 1/(n+1) slack

The `1/(n+1)` term is the price of having only *n* calibration units. With 100 units it is 0.01,
so the dial cannot honestly promise a miss rate below that: a target whose α is smaller than
`1/(n+1)` is out of reach, and so is one the calibration units' own miss rate rules out at every
threshold. `select()` then returns the whole document and sets
**`capped=True`**: "dial capped by calibration size". Recall is trivially 1 there, `guarantee`
stays `True`, and `expected_recall` is 1.0 (what is delivered), not `c`. A bigger calibration set
lowers the slack. The fitted `selector.json` records *n* for every family.

### What it is calibrated against

!!! warning "The guarantee is against teacher relevance, not human labels"
    Relevance on the calibration set is "the **teacher's** topic for the element is one of the
    field's topics". The teacher is a local LLM (Qwen3-4B-Instruct-2507 by default) that labels
    the topic of every content element. The dial promises recall **of what the teacher called
    relevant**. It does not promise recall of what a person would have called relevant, and the
    teacher is not always right.

    The 900 gold-annotated elements of the training corpus give a second, independent check:
    recall with relevance taken from the gold topic, reported on the
    [selection benchmark page](../benchmarks/selection.md) beside the main numbers, together
    with the measured disagreement between teacher and gold. Read both before you rely on the
    dial.

### Units from one document are correlated

A unit is a (document, field) pair, and one document yields several units, which share that
document. Units are therefore correlated, not exchangeable in the strict sense the proof assumes.
`n` counts units; the number of distinct documents, which is smaller, is stored with it (`n_docs`)
so the gap is visible. Treat the guarantee as approximate, and check it on your own documents.

## When there is no guarantee

`Selection.guarantee` is `True` only when **all** of these hold. When any fails, `guarantee` is
`False` and the selection is an **estimate only**:

| Case | Why |
| --- | --- |
| The need is a **free-text query** | The dial is calibrated on fields. There is no query-side calibration, whatever you ask |
| There is **no tagger**, or **no embedder** | The full relevance model needs both. Without them docrooter falls back to `sim` and `section_hit` alone, and the dial was not calibrated on those probabilities |
| The **document type** maps to another family than the field's (`doc_type_mismatch=True`) | The dial was calibrated on correctly typed documents. The content elements are re-tagged within the field's family, which gives a usable selection, not a promise |
| You pass a **budget** | A budget replaces the dial, so no threshold is calibrated |
| The selector has **no dial for the family** | Nothing was calibrated for it |

Without the guarantee the cut is the model's own estimate. Elements are taken in order of
P(relevant) until their share of the total probability, ΣP, reaches `c`. `expected_recall` is
that share, shown with "(estimate)", and `threshold` is the probability of the last element
taken. It rests on the model's calibrated probabilities being right, and nothing checks that on
your document.

If a document is expected to hold nothing relevant (ΣP below 0.5), `expected_recall` is `None`,
meaning undefined, and an estimate or budget cut selects nothing. On the guarantee path nothing
changes: the dial's threshold is applied, which usually keeps nothing.

## Endpoints and expansion

- **`c = 1.0` is always the whole document**: every content element and heading. It bypasses the
  dial and needs no selector, tagger or embedder, so it is never approximate. A `budget` cannot
  be combined with it.
- **Expansion** runs after the cut. Each selected element brings the element just before it when
  that is in the same section, the section's heading and its first paragraph, and a table brings
  its caption (and a caption its table), so tables arrive whole. Expansion only adds elements, so
  recall can only rise, and the guarantee is on the dial's cut and holds for what you receive.
  The readout counts the tokens of the expanded selection.

## A budget

`budget` is a token count. Elements are taken by P(relevant), each with its expansion, while the
total stays within it. One that does not fit is skipped and smaller ones are tried. The readout
gives the **estimated** recall at that cut, never a guarantee. If the most relevant element alone
is over the budget, it is returned alone with `over_budget=True`.

## Your own corpus

The dial is fitted to the calibration set of the training corpus you used. On documents unlike
that set (other publishers, languages or layouts), the guarantee is weaker than it reads.
**Refit on your own corpus**:

1. `docrooter teach label CORPUS --full --splits calib` has the teacher label the topic of every content element
   of the calibration documents (see [Train your own element tagger](../guides/train-your-tagger.md)).
2. `docrooter fit-selector CORPUS --bundle BUNDLE --out selector.json` fits the relevance models
   on half of the calibration documents (calib-fit) and the dials on the other half (calib-crc).
   The two halves are split by a hash of the document id, so the dials are never calibrated on
   documents the relevance models were fitted on.

`fit-selector` refuses a corpus where the teacher labels are missing or do not match the
document text, rather than calibrate on a diverged reading. The
[guide](../guides/select-for-extraction.md#refit-on-your-own-corpus) gives the commands.

## What the Selection reports

| Field | Meaning |
| --- | --- |
| `guarantee` | `True` when the dial's guarantee applies, else the selection is an estimate |
| `capped` | The target was unreachable with this many units; the whole document is returned |
| `doc_type_mismatch` | The document's type maps to another family than the field's |
| `expected_recall` | `c` (or 1.0 when capped) under the guarantee, else the model's estimate; `None` when no relevant content is expected |
| `relevant_expected` | ΣP over the content elements: the expected number of relevant elements |
| `over_budget` | The most relevant element alone is over the token budget |
| `threshold` | The relevance cut (`None` at `c = 1.0`, when capped or under a budget) |
| `tokens`, `token_share` | The tokens selected, and their share of the document's content and heading tokens |
| `chunks` | The selected elements as [chunks](../guides/chunking-for-rag.md), contiguous runs merged |

`readout(selection)` turns these into the one line the CLI and the [playground](../guides/playground.md)
print, such as `≈ 90% of relevant content kept · 3,240 tokens (12%) · guaranteed on calibration-like documents`.

## How good is it?

The benchmark page, [Selection (completeness dial)](../benchmarks/selection.md), compares the
recall the dial delivers with the recall it promises, and the tokens it spends with the tokens of
simple baselines. **The results are pending**; until the notice on that page is removed, no
number on it is final.

## Next

- [Select what to send to an LLM](../guides/select-for-extraction.md): pick a field, choose `c`,
  read the Selection.
- [Element tags](element-tags.md): where topics and functions come from.
- [Chunking for RAG](../guides/chunking-for-rag.md): what `Selection.chunks` are.
