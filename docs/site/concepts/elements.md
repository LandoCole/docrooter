# Elements

A `Document` is a tree of sections. Version 0.4 adds a second view of the same document: an
ordered list of **elements**, the typed blocks a reader sees on the page. Each element has a
kind (paragraph, table, figure, ...), a box on its page, a place in reading order and the
section it belongs to.

Elements are optional. `load()`, `load_pdf()` with its default arguments, `classify_document()`
and `normalize_sections()` behave exactly as they did in v0.3 and build none. Element mode starts
only when you ask for it: `load_pdf(path, elements=True)`, `load(path, elements=True)`, or
[`process()`](rooter.md) with the `balanced` or `accurate` profile.

## The kinds

| Kind | What it is |
| --- | --- |
| `title` | The document title, found on the first page |
| `heading` | A section heading; each one opens a `Section` |
| `paragraph` | Running text. Code blocks are paragraphs with the flag `code` |
| `list_item` | A bulleted or enumerated item, with `list_depth` from its indent |
| `table` | A table, with `TableData` (rows, header rows) and `to_markdown()` |
| `figure` | A picture or a vector drawing, with the text found inside it |
| `caption` | A figure or table caption, linked both ways (`caption_for`, `caption_id`) |
| `footnote` | A note at the foot of a page, with its marker |
| `formula` | A displayed equation |
| `contents` | A printed table-of-contents page |
| `page_header`, `page_footer`, `page_number` | Running lines and page numbers (page furniture) |

An element carries `id` (`e0`, `e1`, ... in reading order, stable for the same input and
configuration), `kind`, `text`, `boxes` (in points, origin top left; one box per page fragment,
none for Markdown), `order`, `section_id`, and when they apply `level`, `list_depth`, `marker`,
`table`, `figure`, `source` (`text` or `ocr`), `confidence` and `flags` (`continued` for a
paragraph that crosses a column or page, `code`, `form`, `low_confidence`).

## Reading elements

This example writes a small two-page report and reads it with the rooter. It lives in
[`examples/elements_and_chunks.py`](../../../examples/elements_and_chunks.py) and needs
`docrooter[pdf]`; the keyword backends need no model and `ocr="none"` needs no OCR engine.

```python
from docrooter import process

analysis = process(path, profile="balanced", ocr="none", doc_type="keyword", sections="keyword")
doc = analysis.document
for element in doc.elements:
    label = element.text[:36].replace("\n", " ")
    print(f"{element.order:>2} p{element.boxes[0].page} {element.kind:<12}{label}")
```

```text
 0 p1 title       Annual Report 2025
 1 p1 heading     1. Business
 2 p1 paragraph   Our sensors measure pressure, temper
 3 p1 paragraph   Competition remains strong, and our 
 4 p1 heading     2. Financial results
 5 p1 table       Segment | 2024 | 2025 Sensors | 412 
 6 p1 caption     Table 1. Revenue by segment (million
 7 p1 page_number Page 1
 8 p2 heading     3. Outlook
 9 p2 paragraph   We expect revenue to keep growing as
10 p2 page_number Page 2
```

- The first page has two text columns. The left column is read before the right one (elements 2
  and 3), and the table is not split into columns.
- The table's rows are in `element.table`; `element.table.to_markdown()` gives a pipe table.
- The caption is linked to the table it describes.
- The page numbers sit at the end of their pages. See [page furniture](#page-furniture).

The same from the shell, one row per element:

```bash
docrooter elements report.pdf                          # the balanced profile
docrooter elements report.pdf --kinds heading,table    # only these kinds
docrooter elements report.pdf --json > elements.json   # every field
```

`docrooter elements` needs a profile that builds elements, so `--profile fast` is an error.

## How elements are built

The builder works on engine-neutral page objects (text lines, image placements, vector drawings
and ruled tables), so PyMuPDF and pdfplumber give the same kinds. In element mode the pipeline
is:

1. Route each page: read its text layer, or OCR it (see [the rooter](rooter.md)).
2. Put the lines of each page in reading order.
3. Find headings, and build the section tree from them.
4. Type every line: furniture, contents, tables, figures, captions, footnotes, lists, formulas,
   then paragraphs.

Every line with text belongs to exactly one element, and element mode builds the table of
contents from its heading elements, so sections, summaries and chunks always agree with the
elements. A section's `element_ids` lists its own elements, and its text is their texts in
order, without furniture or `contents`.

### Reading order

Pages are read with a recursive XY-cut: full-width boxes split a page into bands, vertical
gutters inside a band make columns, and the pieces are visited top to bottom and left to right.
A page with no column gutter keeps the extractor's order exactly. A gutter that runs between the
row-aligned labels and numbers of an unruled table is not a column boundary, so financial
tables keep their rows together.

The heuristic order has limits. A table with wide text cells can still be cut as if it were two
columns, and the axis labels of a chart have no meaningful order. The layout model orders by
region instead (see below).

### Page furniture

Running headers, footers and page numbers are moved out of the body flow in element mode, by
design: lines in the upper half of the page go to the start of that page's elements and lines
in the lower half go to its end. Their text is not in `Section.text`. In `fast` mode they stay
where the extractor put them. On the 4,519 dev-corpus pages where XY-cut finds no gutter, the
body order equals `fast`'s, and 272 of those pages differ from `fast` only in where furniture
sits (`benchmarks/results/2026-10-19/elements-reading-order.json`).

### Markdown

Markdown builds elements too: ATX and setext headings, paragraphs, list items with depth, pipe
tables, fenced code (a paragraph with the flag `code`), `$$` blocks (formulas) and images
(figures, with the alt text). Their `boxes` are empty.

## Heuristic or layout model

The default **heuristic** engine types lines with the rules above. The `accurate` profile
(`layout="heron"`, extra `layout`) adds Docling's heron layout model, run through onnxruntime
(no torch): it finds labelled regions on the rendered page, and each region sets the kind of
the lines inside it. Lines outside every region keep the heuristic rules.

The difference was measured on 400 DocLayNet test pages (born-digital PDFs from six document
categories). Predicted boxes are matched one to one to gold boxes at IoU 0.5:

| | Heuristic (pymupdf) | Heron |
| --- | ---: | ---: |
| Macro F1 over 11 classes | 0.257 | 0.736 |
| Table F1 | 0.290 | 0.801 |
| Text F1 | 0.407 | 0.836 |
| Text coverage of Text | 0.927 | 0.949 |
| Seconds per page | 0.029 | 0.609 |

Text coverage is the share of the characters in the gold Text regions that land in a predicted
element of the right kind, whatever its box. The heuristic reads almost all of the text and
puts it in paragraphs; what it gets wrong is the boundaries between regions, tables without
ruling lines, pictures, formulas and headings. So:

- **`balanced`** (heuristic) is fast and its text coverage is good. Use it when you need the
  text of a document in order and sections that follow its headings.
- **`accurate`** (heron) is for structure quality: tables, figures, captions, headers and
  footers, and headings.

The heuristic rules miss the target set for them (F1 of 0.8 for Text, Table and Picture on
financial reports and scientific articles) by a wide margin. Two signals also need several
pages: running headers are found by recurrence across pages, and the document title is the
first large heading, so single-page PDFs such as DocLayNet's understate the heuristic on both.

Source: `benchmarks/results/2026-10-18/elements-doclaynet.json`, with the method and the
per-class tables in [Element benchmarks](../benchmarks/elements.md). Those scores were produced
before the XY-cut table fix (commit `c09b9f7`) and were not re-scored; they may move.

## Speed

Seconds per 100 pages on ten development PDFs (1,439 pages), with no OCR and one worker process
per system:

| System | s per 100 pages | Peak memory |
| --- | ---: | ---: |
| `fast` (no elements) | 0.99 | 200 MiB |
| `balanced`, PyMuPDF | 4.64 | 263 MiB |
| `balanced`, pdfplumber | 7.65 | 308 MiB |
| `accurate` (heron) | 64.6 | 904 MiB |

Source: `benchmarks/results/2026-10-18/profile-elements.json`; method in
[Performance and memory](../benchmarks/performance.md). They were measured before the XY-cut
table fix, which adds one cheap test per candidate gutter.

Most of element mode's extra cost is the search for ruled tables. `tables="none"` skips it, and
is the speed switch for documents full of ruled tables, such as datasheets. Set it in a
[configuration file](rooter.md#configuration):

```text
tables = "none"
```

```bash
docrooter elements datasheet.pdf --config rooter.toml
```

With `tables="none"` the lines of a ruled table are typed by the other rules instead of becoming
a `table` element, so use it when speed matters more than table structure.

## Next

- [The rooter](rooter.md) chooses engines per task and per page, and records why.
- [OCR and layout models](../guides/ocr-and-layout.md) covers the extras and engines.
- [Chunking for RAG](../guides/chunking-for-rag.md) turns elements into retrieval chunks.
