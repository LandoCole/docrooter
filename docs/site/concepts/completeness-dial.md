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
| A **field** | `Field`, or a list of fields of one family | Elements whose topic is one of the field's topics | Possible below `c = 1.0` for **one** field the dial was calibrated on (see below); a list is an estimate |
| A **query** | a `str` | The embedding similarity of each element to the text, plus the element's kind and length. Tags are not used | Never below `c = 1.0` |

A **field** is a named, query-independent target tied to [element tags](element-tags.md): its
`topics` are `element_topics` leaves, and an element is relevant to the field when its topic is
one of them. Its `functions` (such as `risk_disclosure`) only strengthen the ranking, and its
optional `query` is the field as a question, for the embedding term. Three built-in sets ship in
the package: `annual_report` (15 fields), `fund_document` (12) and `contract` (15). Some fields
are one topic leaf; a few cross cutting ones gather several, such as `contract.core_obligations`
or `fund_document.costs_and_returns`. A list of fields of one family is one need: the union of
their topics. Below `c = 1.0` a list is an **estimate only** (`multi_field=True`): the dial is
calibrated on units of one field each, and a union of fields is scored and weighted differently,
so for a guarantee call `select()` **once per field**.

Each content element (paragraph, list item, table, caption, footnote, formula) gets a
probability P(relevant) from a small logistic regression over a few features:

| Feature | What it is |
| --- | --- |
| `p_topic` | The tagger's probability on the need's topics (0 when the element has no tags) |
| `p_func` | The highest tagger probability over the need's functions |
| `sim` | The cosine between the element's heading path and text and the need text, from an embedding model |
| `section_hit` | Whether the element's section normalises to one of the need's topics |
| `kind`, `log_tokens` | The element's kind and its size |

For a free-text query the tag features (`p_topic`, `p_func`, `section_hit`) are 0, so only `sim`,
`kind` and `log_tokens` carry it. Headings are read and counted but never scored: they arrive as
context.

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

> E[recall] ≥ c on a new (document, field) unit that is exchangeable with the calibration units.

There is one dial for each element **family** (`annual_report`, `fund_document`, `contract`),
calibrated on that family's units, pooled over **all** of the family's fields. The table of
thresholds is precomputed on a grid of `c`
(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99), and a value between grid points is evaluated
exactly from the stored risk curve, so moving the slider never loosens the guarantee.

### What `c` means

`c` is an **average over documents and fields**: over (document, field) units of documents like
the calibration set, pooled across the family's fields. It is not a promise about any one
document: at `c = 0.9` some documents will keep less than 90% of their relevant elements and some
will keep more, and the mean is at least 0.9. Nor is it a promise about any one field: a hard or
cross-cutting field can sit below `c` on average while the family's mean is at `c`. For a
document where a miss is costly, use a higher `c`, or `c = 1.0`.

### The 1/(n+1) slack

The `1/(n+1)` term is the price of having only *n* calibration units. With 100 units it is 0.01,
so the dial cannot honestly promise a miss rate below that. A target whose α = 1 − c is smaller
than `1/(n+1)`, that is `c > n/(n+1)` (above 0.990 for 100 units), is out of reach, and only then
is the dial capped. `select()` then returns the whole document and sets
**`capped=True`**: "dial capped by calibration size". Recall is trivially 1 there, `guarantee`
stays `True`, and `expected_recall` is 1.0 (what is delivered), not `c`. A bigger calibration set
lowers the slack. The fitted `selector.json` records *n* for every family, and a selection made
with the dial reports it as `calibration_n` (the playground shows it as a badge).

### What it is calibrated against

!!! warning "The guarantee is against teacher relevance, not human labels"
    Relevance on the calibration set is "the **teacher's** topic for the element is one of the
    field's topics". The teacher is a local LLM (Qwen3-4B-Instruct-2507 by default) that labels
    the topic of every content element. The dial promises recall **of what the teacher called
    relevant**. It does not promise recall of what a person would have called relevant, and the
    teacher is not always right.

    The 900 gold-annotated elements of the training corpus give a second, independent check:
    recall with relevance taken from the gold topic. Those labels are **Claude-annotated**, not
    human: the annotators share one underlying model, so the check compares the teacher with
    another labeller, not with people. It is reported on the
    [selection benchmark page](../benchmarks/selection.md) beside the main numbers, together
    with the measured disagreement between teacher and gold. Read both before you rely on the
    dial.

### Units from one document are correlated

A unit is a (document, field) pair, and one document yields several units, which share that
document. Units are therefore correlated, not exchangeable in the strict sense the proof assumes.
`n` counts units; the number of distinct documents, which is smaller, is stored with it (`n_docs`)
so the gap is visible. And because units are pooled over the family's fields, the guarantee is
an average over fields too, as [above](#what-c-means). Treat the guarantee as approximate, and
check it on your own documents.

## The same reading, tagger and embedder

The dial was calibrated on probabilities computed in one particular way, and it promises nothing
about probabilities computed another way. `fit-selector` records that way in the selector's
**provenance** (`Selector.provenance`, saved in `selector.json`):

- the **tagger**: the fingerprint of the tagger bundle (a content hash of its files; a
  `selector.json` kept in the bundle does not count);
- the **embedder**: the embedding spec that gave `sim`;
- the **reading**: the rooter profile, its layout engine (`heron` for `accurate`, the heuristic
  rules for `balanced`) and the sections backend that labelled the sections, on which
  `section_hit` depends. `fit-selector` reads with `accurate` by default, as the corpus build
  does.

`select()` grants the guarantee only when the document was **read and tagged the same way**:
every content element's tags carry the fingerprint of the fit's bundle (tags from a zero-shot
`embedding:` or `teacher:` tagger, or from another bundle, do not), `sim` comes from the same
embedder spec, the analysis was read with the same layout engine, and its sections were
labelled by the same sections backend with **escalation off** (`analysis.config.sections` and
`analysis.config.escalate`; labels from another backend, from an unknown one, none at all, or a
configured escalation, which relabels low-confidence sections with another backend, give
`"sections"`). Backends are compared by their full spec, revision and recipe included
(`EmbeddingBackend.spec`), never by their display name: a backend object without a full spec
does not match. Otherwise `Selection.fit_mismatch` lists
what differs (`"tagger"`, `"embedder"`, `"layout"`, `"sections"`), the selection is an estimate,
and the readout adds "reading differs from calibration (estimate)". A selector saved before provenance was recorded reports
`fit_mismatch=["provenance"]` and guarantees nothing until you refit.

The flow that earns the guarantee reads with the fit's profile and tags with the bundle that
holds the selector:

```python
analysis = process(
    "report.pdf", profile="accurate", element_tags="bundles/tagger-a", escalate="none"
)
selection = select(analysis, risk, completeness=0.9, selector="bundles/tagger-a")
```

The default `balanced` profile reads with the heuristic layout rules, so the same call on
`process("report.pdf", element_tags=...)` reports `fit_mismatch=["layout"]`. `escalate="none"`
matters when `TYPESAFE_API_KEY` is set: `accurate` then escalates sections to Jev, and the
selection reports `fit_mismatch=["sections"]`. From the shell, `docrooter select report.pdf
--need annual_report.risk_factors --profile accurate --tagger bundles/tagger-a` does the same;
it always turns escalation off.

## When there is no guarantee

**Below `c = 1.0`**, `Selection.guarantee` is `True` only when **all** of these hold. When any
fails, `guarantee` is `False` and the selection is an **estimate only**. (At `c = 1.0` every
need, a free-text query and a document-type mismatch included, returns the whole document and
reports `guarantee=True`: nothing is missed, so there is nothing to estimate.)

| Case | Why |
| --- | --- |
| The need is a **free-text query** | The dial is calibrated on fields. There is no query-side calibration, whatever you ask |
| The need is **several fields** (`multi_field=True`) | The dial is calibrated on single-field units; a union of fields is not one of them. Call `select()` once per field |
| The document was **read, tagged or embedded otherwise than the fit** (`fit_mismatch` not empty) | The dial's probabilities came from the fit's tagger bundle, embedder and reading; others give other probabilities. See [the same reading, tagger and embedder](#the-same-reading-tagger-and-embedder) |
| A field of the need is **not one the dial was calibrated on** (`uncalibrated_field=True`) | The selector records the fields it was fitted with, each by id and a fingerprint of its topics, functions and need text, and only those that gave the dial at least one calibration unit. A field of your own, or a built-in one whose topics or query you edited, is not among them. A `selector.json` made before this was recorded calibrated no field, so every field in it reads as uncalibrated |
| There is **no tagger**, or **no embedder** | The full relevance model needs both. Without them docrooter falls back to `sim` and `section_hit` alone, and the dial was not calibrated on those probabilities |
| The **document type** maps to another family than the field's (`doc_type_mismatch=True`) | The dial was calibrated on correctly typed documents. The content elements are re-tagged within the field's family, which gives a usable selection, not a promise |
| You pass a **budget** | A budget replaces the dial, so no threshold is calibrated |
| The selector has **no dial for the family** | Nothing was calibrated for it |

Without the guarantee the cut is the model's own estimate. Elements are taken in order of
P(relevant) until their share of the total probability, ΣP, reaches `c`. `expected_recall` is
that share, shown with "(estimate)", and `threshold` is the probability of the last element
taken. It rests on the model's calibrated probabilities being right, and nothing checks that on
your document.

If a document is expected to hold nothing relevant (ΣP below 0.5), the estimate and budget cuts
select nothing and `expected_recall` is `None`, meaning undefined. This happens only **without**
the guarantee. On the guarantee path nothing changes: the dial's threshold is applied (which
usually keeps nothing) and `expected_recall` is still `c`, or 1.0 when capped, since that is what
is promised.

## Endpoints and expansion

- **`c = 1.0` is always the whole document**: every content element and heading. It bypasses the
  dial and needs no selector, tagger or embedder, so it is never approximate, and it reports
  `guarantee=True` for any need. A `budget` cannot
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
is over the budget, it is returned alone with `over_budget=True`. If it fits alone but nothing fits
together with its expansion, it is returned alone too.

## Your own fields

A field outside the set the dial was fitted with is not covered, whatever its topics. `select()`
checks every field of the need against the selector's record, by id and by a fingerprint of its
sorted topics and functions and its need text, which `sim` compares elements with (its `query`,
else its `description`; the label does not enter it). A field that does not match gets the
estimate path, `guarantee=False` and **`uncalibrated_field=True`**, and the readout says "field
not calibrated (estimate)". A field that gave the dial no calibration unit (no calibration
document had an element relevant to it) is not recorded as calibrated either: the dial has
seen nothing of it. `fit-selector` reports the units per field. To make your own field part of the guarantee,
include it when you fit (see the [guide](../guides/select-for-extraction.md#refit-on-your-own-corpus));
the dial then pools its units with the family's.

## Your own corpus

The dial is fitted to the calibration set of the training corpus you used. On documents unlike
that set (other publishers, languages or layouts), the guarantee is weaker than it reads.
**Refit on your own corpus**:

1. `docrooter teach label CORPUS --full --splits calib` has the teacher label the topic of every
   content element of the calibration documents (see
   [Train your own element tagger](../guides/train-your-tagger.md)).
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
| `guarantee` | `True` when the dial's guarantee applies (or `c = 1.0`), else the selection is an estimate |
| `capped` | The target was unreachable with this many units; the whole document is returned |
| `doc_type_mismatch` | The document's type maps to another family than the field's |
| `uncalibrated_field` | The guarantee would have applied, but a field of the need is not one the dial was calibrated on |
| `multi_field` | The need is several fields, an estimate below `c = 1.0` |
| `fit_mismatch` | What differs from the selector's fit: `tagger`, `embedder`, `layout`, `sections` (no labels, labels from another or an unknown sections backend, or escalation on), or `provenance` for an old selector; empty when nothing does |
| `calibration_n` | The *n* of the dial that was applied (`None` when none was) |
| `expected_recall` | `c` (or 1.0 when capped) under the guarantee, else the model's estimate; `None` only without the guarantee, when no relevant content is expected |
| `relevant_expected` | ΣP over the content elements: the expected number of relevant elements |
| `over_budget` | The most relevant element alone is over the token budget |
| `threshold` | The relevance cut (`None` at `c = 1.0`, when capped or under a budget) |
| `tokens`, `token_share` | The tokens selected, and their share of the document's content and heading tokens |
| `chunks` | The selected elements as [chunks](../guides/chunking-for-rag.md), contiguous runs merged |

`readout(selection)` turns these into the one line the CLI and the [playground](../guides/playground.md)
print. The line has this shape (an illustration, not a result):
`≈ 90% of relevant content kept · 3,240 tokens (12%) · guaranteed on calibration-like documents`.

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
