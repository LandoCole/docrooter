# Chunking for RAG

Retrieval works best on pieces of a document that stand on their own. Cutting every N tokens
splits tables in half and loses the heading that says what a paragraph is about.
`chunk()` cuts along the document's own structure instead: it packs the
[elements](../concepts/elements.md) of each section, in reading order, into chunks of at most
`max_tokens`, and starts every chunk with its heading path.

## Cutting a document

```python
chunks = chunk(analysis, max_tokens=100)
for piece in chunks:
    print(f"{piece.id}: {piece.tokens} tokens, {' > '.join(piece.heading_path)} {piece.flags}")
print(chunks[2].text)  # a table is never split: a description line, then its rows
```

On the sample report of [`examples/elements_and_chunks.py`](../../../examples/elements_and_chunks.py)
(two pages: two text columns, a ruled table with a caption, a short second page), this prints:

```text
c0: 4 tokens, Annual Report 2025 ['heading_only']
c1: 94 tokens, 1. Business []
c2: 65 tokens, 2. Financial results []
c3: 12 tokens, 2. Financial results []
c4: 25 tokens, 3. Outlook []
2. Financial results

Table: Table 1. Revenue by segment (millions).; 4×3
|  |  |  |
| --- | --- | --- |
| Segment | 2024 | 2025 |
| Sensors | 412 | 468 |
| Controls | 198 | 215 |
| Services | 96 | 121 |
```

`chunk()` takes an `Analysis` (from [`process()`](../concepts/rooter.md)) or a `Document` loaded
with `elements=True`, and returns `Chunk` objects:

| Field | Holds |
| --- | --- |
| `id` | `c0`, `c1`, ... in reading order |
| `section_id` | The section the chunk is from |
| `element_ids` | The elements whose text is in the chunk |
| `text` | The heading path joined with ` > `, a blank line, then the element texts joined with blank lines |
| `heading_path` | The section titles from the top of the tree down to this section |
| `tokens` | The estimated size: 1.3 tokens per whitespace-separated word, not a model's tokenizer |
| `flags` | `oversize` or `heading_only` (below) |

## The rules

- **A chunk never crosses a section boundary and never splits an element.** Elements are
  packed in reading order until the next one would push the chunk, heading path included, past
  `max_tokens`.
- **A table is always one chunk.** Its text is a description line (its caption, or its first
  row, and its size in rows by columns) followed by the markdown of its rows, so an embedding model
  sees what the table is before it sees the numbers.
- **A single element longer than `max_tokens` becomes its own chunk**, flagged `oversize`. If
  the heading path alone is longer than `max_tokens`, every chunk of that section is flagged.
- **Every heading is in exactly one chunk.** A heading that opens a section is the last item of
  the heading path, so it is not repeated in the body; it is listed in `element_ids` of the
  section's first chunk. A section with no body, such as a title followed at once by a
  subsection, gets a chunk of its own that holds just the path, flagged `heading_only` (`c0`
  above).
- **Page furniture and `contents` are skipped.** Running headers, footers, page numbers and
  printed contents pages are not worth retrieving; pass `include_furniture=True` to keep them.
  In element mode these lines are moved to the start or end of their page, by design, so they
  never interrupt a paragraph; this is also why the page numbers in the example come last on
  their pages.
- **`overlap_elements=N`** repeats up to `N` elements from the end of a chunk at the start of the
  next one in the same section (never a table, and only while the result still fits).

## From the command line

```bash
docrooter process report.pdf --profile balanced --out analysis.json --chunks
```

The analysis JSON gains a `chunks` list. `--chunks` needs a profile that builds elements, so it
does not work with `--profile fast`.

## Embedding chunks

`embed(analysis, "embedding:BAAI/bge-small-en-v1.5", unit="chunk")` embeds the same texts, with
the heading path in front, and returns an `EmbeddingSet`: the chunk ids, L2-normalised vectors
(float16 by default) and the model and revision that made them, so that queries can be embedded
with the same model later. It needs the `models` extra. `unit="element"` embeds each element
instead, except page furniture and `contents`, and `kinds=` picks exactly the element kinds
you want. `EmbeddingSet.to_npz(path)` writes a compact file and `EmbeddingSet.from_npz(path)`
reads it back.

To have the rooter do it in the same run, set the `embeddings` task to an embedding backend
spec in the configuration; `Analysis.embeddings` then holds one vector per element. Documents
are embedded bare; a question is embedded with the backend's query prompt (`Recipe.query_prompt`,
for example the `Instruct: ... Query:` form of UltraBERT), and ranking by dot product works
because the vectors are normalised.

## Choosing a profile for retrieval

- **`balanced`** (heuristic layout) is fast, and nearly all of a document's text is kept as
  paragraphs: text coverage of 0.927 on DocLayNet test pages. 4.64 s per 100 pages with PyMuPDF.
  Its weak points are tables without ruling lines, which it reads as paragraphs, and
  headers and footers on single-page files, where nothing repeats.
- **`accurate`** (heron) is for structure quality. DocLayNet macro F1 is 0.736 against
  0.257, and Table F1 is 0.801 against 0.290, so tables become table chunks more often. It takes
  64.6 s per 100 pages.
- **`tables="none"`** is the speed switch if your documents are full of ruled tables and you
  do not need them as tables; those lines are then typed as paragraphs or other kinds.
- Scanned pages are OCR'd in `balanced` and `accurate` when an engine is installed (see
  [OCR and layout models](ocr-and-layout.md)); OCR text becomes elements and chunks like any other.

Sources: `benchmarks/results/2026-10-18/elements-doclaynet.json` and
`benchmarks/results/2026-10-18/profile-elements.json`; the DocLayNet scores predate the XY-cut
table fix (commit `c09b9f7`) and were not re-scored.

## Summaries

`process()` also writes a short summary of each section and of each long paragraph or list item
into `Analysis.summaries`, keyed by section id and element id: up to three key sentences
(chosen without redundancy) and eight key phrases, with no model. A table gets a one-line
description. They make useful index fields or snippets next to the chunk text.
