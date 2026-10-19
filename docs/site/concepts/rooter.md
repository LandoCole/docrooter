# The rooter

The rooter is one configurable pipeline, `process()`. For each page it decides whether to read
the text layer or to OCR the page. For each task it decides which engine or backend runs. It
ships three **profiles** and records every decision it makes in a route log.

```python
analysis = process(path, profile="balanced", ocr="none", doc_type="keyword", sections="keyword")
```

`process(source, *, config=None, profile=None, **overrides)` returns an `Analysis`:

| Field | Holds |
| --- | --- |
| `document` | The `Document`, with its [elements](elements.md) (none under `fast`) and the route log in `meta["routes"]` |
| `doc_type`, `sections` | The document-type and section results, or `None` when the task is `"none"` |
| `summaries` | Key sentences and key phrases, by section id and by element id |
| `embeddings` | One vector per element, only when the `embeddings` task is set |
| `config` | The resolved configuration that ran |
| `timings` | Seconds per stage |

## Profiles

| Task | `fast` | `balanced` (default) | `accurate` |
| --- | --- | --- | --- |
| `text` | `auto` | `auto` | `auto` |
| `ocr` | `none` | `auto` | `auto` |
| `layout` | `none` (no elements) | `heuristic` | `heron` |
| `tables` | `lines` | `lines` | `lines` |
| `doc_type`, `sections` | the default backend | the default backend | the default backend |
| `summaries` | `none` | `centroid` | `centroid` |
| `escalate` | `none` | `none` | `jev:jev-1.13.0` if `TYPESAFE_API_KEY` is set, else `none` |

- **`fast`** is the v0.3 pipeline: no elements, no OCR, no summaries. Its output is
  byte-identical to calling `load()`, `classify_document()` and `normalize_sections()` yourself,
  and a test asserts it. It runs at v0.3's speed (within 5%: +0.8% to +2.8% in the measured
  runs, `docs/performance.md`).
- **`balanced`** adds elements from the heuristic rules, OCR when an engine is available, and
  centroid summaries. It is fast, and its text coverage is good.
- **`accurate`** swaps the rules for the heron layout model (extra `layout`) and escalates
  uncertain labels to Jev when a key is set. It is for structure quality: heuristic layout
  scores a macro F1 of 0.257 on DocLayNet against heron's 0.736 (see
  [Elements](elements.md#heuristic-or-layout-model)), and it costs more time. Per 100 pages it
  takes 0.99 s in `fast`, 4.64 s in `balanced` with PyMuPDF and 64.6 s in `accurate`
  (`benchmarks/results/2026-10-18/profile-elements.json`).

Element mode moves page furniture (running headers, footers and page numbers) to the start or
end of its page, by design; `fast` leaves those lines where the extractor found them.

`tables="none"` is the speed switch inside element mode: it skips the search for ruled tables,
which is most of the extra cost over `fast`.

## Configuration

A configuration is a `RouterConfig`: the task-to-engine mapping, the profile, `max_pages`, the
OCR language (`ocr_lang`, `"en"`) and render resolution (`ocr_dpi`, 200), and the escalation
threshold. Unknown keys are errors. It comes from three places, in this order of precedence:

1. arguments to `process()` (any `RouterConfig` field: `ocr="tesseract"`, `tables="none"`, ...);
2. a TOML or JSON file (`config="rooter.toml"`, or `--config` on the command line);
3. the profile's defaults.

Keys a file leaves out take its `profile`'s values. `docrooter config show` prints the resolved
configuration as TOML you can save and edit:

```bash
docrooter config show --profile accurate > rooter.toml
docrooter elements report.pdf --config rooter.toml
docrooter process report.pdf --profile balanced --out analysis.json --chunks
```

A small file that only changes what differs from `balanced`:

```text
profile = "balanced"
ocr = "tesseract"
tables = "none"
doc_type = "keyword"
sections = "keyword"
```

Engine and backend names are checked when `process()` resolves them, so engines registered later
with `register_ocr_engine` and `register_layout_engine` work in a configuration. You can also
pass an engine or backend object instead of a name; the caller then owns it and it is never
closed by the rooter.

## Page routing

In element mode each PDF page is measured from its text engine's objects, and routed one of
three ways:

| Route | When | What happens |
| --- | --- | --- |
| `ocr` | The page is mostly one image and has no text layer; or more than 10% of its characters are garbled (replacement characters, `(cid:N)`, private-use or control characters, or text that is not word-like); or its invisible text layer is junk | The page's text-layer lines are replaced by OCR lines |
| `text+ocr` | A scan that carries some text | The text layer is kept and OCR adds only the lines it lacks |
| `text` | Anything else | The text layer is used |

Separately, an image of at least 5% of a page that has no text inside it is OCR'd as a region, and
the result becomes that figure's text (`source="ocr_region"`), never paragraph text. Text layers
in Chinese, Japanese, Korean, Cyrillic, Arabic and other non-Latin scripts are not mistaken for
garbage: the word-like test applies only when at least half of the letters are Latin.

OCR feeds everything in element mode. OCR lines are inserted in page order before headings are
found, so sections, classification views and elements all see them, and a PDF that mixes scanned
and text pages gets one continuous section tree. With `ocr="none"`, or when `auto` finds no
engine, a page that needed OCR keeps its text layer, the route log says `needs_ocr`, and
`meta["warnings"]` gets `"ocr_skipped"` (or `"ocr_unavailable"` for `auto`).

How accurate is the routing? On the 51 development PDFs (5,069 born-digital pages; an
in-sample result, since the routing thresholds were set on them), 5,055 pages were routed
`text`, 13 were routed `text+ocr` (full-page images that carry 21 to 254 characters of text
layer, such as covers and photo pages) and 1 was routed `ocr`: a page with no text layer at
all. No page with a usable text layer was replaced. All 60 image-only test pages were routed
`ocr`. Source: `benchmarks/results/2026-10-18/ocr.json`, section `routing`.

### The route log

Every decision is a `RouteDecision(scope, target, task, engine, reason, metrics)` in
`doc.meta["routes"]`: first the document-scope choices of the configuration (which OCR engine,
which layout, which backends), then one per page, and one per OCR region. Each is also emitted as a
`route` [event](events.md), which the playground shows on its timeline.

`doc.meta["routes"]` is a list of dictionaries with those fields. Under `fast` there is no
routing, no object pass and no `routes` key.

## Escalation

When `escalate` names a backend and a result's confidence is below `escalate_below` (0.5), the
same view is classified again by the escalation backend, and the escalated result replaces the
primary one. Confidences from different backends are not calibrated against each other, so
they are not compared: configure the escalation backend as the stronger one. If it changes the
document type, the sections are normalised once with the new type's taxonomy. Sections that are
still below the threshold are re-labelled in one batch. Both results and the reason are in the
route log. An escalation backend that raises (a network error, a rejected key) never aborts the
analysis: the primary result is kept and the route log gets a warning with the error's type.

## Many documents

`process_many(paths, ...)` yields one `Analysis` at a time and shares one OCR engine, one
layout engine and one set of backends across the documents. They are created on first use and
closed when the iterator is exhausted or closed, so a model loads once per run, not once per
file. At most one model-loading process runs at a time, within the 8 GiB memory budget (see
[Memory and performance](../guides/memory-performance.md)).

## Next

- [OCR and layout models](../guides/ocr-and-layout.md): the extras, the engines and how to add
  your own.
- [Chunking for RAG](../guides/chunking-for-rag.md).
