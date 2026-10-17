# The TOC engine

`load()` turns a file into a `Document`: a tree of `Section`s in reading order. For Markdown the
headings are the `#` lines. For PDFs there are no heading tags, so docrooter reads every text
line with its font, size, weight and position, and decides which lines are headings. This page
explains how, so you can tell why a heading was found or missed.

## The document model

```python
for entry in doc.toc():  # reading order; depth is the position in the tree
    section = doc.sections[entry.id]
print([leaf.title for leaf in doc.leaves()])  # sections without children
print(doc.section("2. Risk factors").text)  # look up by id or by title
```

- `doc.sections` maps a section id (`s1`, `s1.2`, `s1.2.3`, ...) to a `Section` with `title`,
  `level`, `text` (the body up to the next heading), `parent_id`, `children`, `page_start`
  (PDFs), `heading_score` and `flags`.
- `doc.toc()` lists the headings in reading order with their `level` and their `depth` in the
  tree. `doc.walk()` yields every section, `doc.leaves()` the ones without children.
- `doc.meta` (PDFs) holds `engine`, `n_pages`, `timings` (seconds per stage), `bookmarks` (how
  many the file has), `printed_toc_entries`, `body_size` and `warnings`
  (`"no_text_layer"`, `"truncated_pages"`).
- Text before the first heading becomes a level-0 preamble section, which is not in `toc()`.
- `Document.from_sections([...])` builds a document from headings you already have.

The full script is [`examples/toc.py`](../../../examples/toc.py).

## How PDF headings are found

**1. Lines.** The engine extracts each text line with its size, boldness, page and position.
No OCR is involved: a page without a text layer yields nothing.

**2. Document signals.** Before looking at headings, the engine measures the document:

- the **body size** (the most common text size) and the median gap between lines;
- **running headers and footers**: lines in the top or bottom 8% of the page that recur on at
  least 30% of pages (minimum 3), or whose text, minus a page number, recurs on 3 pages. Page
  numbers ("5", "Page 5 of 80", "v") are excluded too;
- **printed contents pages**: lines ending in a page number after dot leaders
  (`Introduction ........ 5`), at least 5 per page, or a page headed "Contents";
- the PDF's **bookmarks** (its outline), if it has any.

**3. Candidates.** A line can be a heading if it is large (at least 1.15 × body size), or
isolated (a gap above it of at least 1.2 × the median gap) and bold or numbered. Lines smaller
than body text, longer than 20 words, running headers, page numbers and lines on contents pages
never are. Consecutive candidate lines of the same style are merged into one wrapped title.

**4. Scoring.** Each candidate gets a score between 0 and 1:

| Signal | Weight |
| --- | ---: |
| Looks like a title (title case, no sentence punctuation, not a URL, not repeated, ...) | 0.40 × title score |
| Size above body text (full weight at 1.6 × body) | up to 0.25 |
| Bold | 0.10 |
| Isolated from the text around it | 0.10 |
| Numbered ("2.1", "IV.", "Item 7", "Chapter 3") | 0.10 |
| Matches an entry of the printed contents page | 0.25 |
| Matches a bookmark (same title, within one page) | 0.25 |
| Followed by a line of the same style (a list or table, not a heading) | −0.15 |

Candidates scoring at least `min_heading_score` (0.55 for PDFs, 0.5 for Markdown) are kept.
Kept headings below 0.7 get the flag `low_heading_confidence`.

**5. Levels.** Heading styles (size, then bold) are ranked: the largest recurring style is
level 1, the next level 2, and so on. A style used only once or twice (a cover title) shares the
level of the next recurring style. When at least half the headings carry dotted numbering
("2.1.3"), the numbering sets the level. A matching bookmark's level wins over both (flag
`level_from_bookmark`). Finally, gaps are closed so a level never jumps by more than one
(flag `level_repaired`).

**6. Tree.** The headings and the text between them become sections, with page numbers.

Markdown documents skip steps 1-4 but get the same title scoring (false headings below 0.5 are
merged into the text before them) and the same level repair: in a Markdown file whose headings
are mostly numbered, `## 1. Business` becomes level 1.

## Engines

`load(path, engine=...)` and `load_pdf(path, engine=...)` take:

| Engine | Extra | What it does |
| --- | --- | --- |
| `auto` (default) | `pdf` or `pdf-mit` | PyMuPDF if installed, otherwise pdfplumber |
| `pymupdf` | `pdf` (AGPL-3.0) | The fastest: about 1 s per 100 pages on the benchmark machine |
| `pdfplumber` | `pdf-mit` (MIT) | Permissive licence, about 6 times slower |
| `both` | both | PyMuPDF headings kept when pdfplumber agrees or either engine is at least 0.8 confident |
| `pymupdf4llm` | `pdf` | The legacy v0.1 path through Markdown, kept for comparison |

On the 24 bookmarked development PDFs the three main engines score within 1.3 points of each
other (F1 at depth ≤ 2: 0.703, 0.711, 0.716). See the [PDF benchmarks](../benchmarks/pdf.md).

## Options

```python
plain = load_pdf(path, engine="pdfplumber", use_bookmarks=False, max_pages=50)
strict = load(path, min_heading_score=0.7)
```

- `use_bookmarks=False` ignores the PDF's outline. With its 102 bookmarks, Apple's 2025 10-K
  gave 94 headings, 87 of them with the bookmark's level; without them, 32 from typography
  alone. Bookmarks are the strongest signal when a file has them. (The benchmarks run with
  `use_bookmarks=False`, because the bookmarks are their answer key.)
- `use_printed_toc=False` ignores the printed contents page.
- `max_pages=N` reads only the first N pages, which bounds time and memory on very long files;
  `meta["warnings"]` then contains `"truncated_pages"`.
- `min_heading_score` trades recall for precision.
- `repair=False` keeps the raw levels (Markdown and `pymupdf4llm`).
- `on_event` reports the stages `extract` (with progress every 10 pages), `signals`,
  `headings` and `assemble`; see [Events](events.md).

## Known limitations

These come from the benchmarks; the [held-out analysis](../benchmarks/heldout.md) has the
details.

- **No OCR.** Scanned PDFs give an empty document (`meta["warnings"]` has `"no_text_layer"`).
- **Headings set at body size that are neither bold nor numbered** are not candidates, and
  headings set *below* body size (some small-caps styles) never are. Earnings releases whose section titles are bold body
  text are a known weak spot.
- **Slide decks and bilingual layouts.** Slide titles and parallel-column statutes produce poor
  tables of contents.
- **Repeated headings in the page margins.** A real heading printed at the same height in the
  top or bottom 8% of 3 or more pages is treated as a running header.
- **Levels after a split cover title** can be one too deep.
