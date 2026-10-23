# The TOC engine

`load()` turns a file into a `Document`: a tree of `Section`s in reading order. For Markdown the
headings are the `#` lines. For PDFs, docrooter asks PyMuPDF4LLM, whose layout model reads each
page and writes Markdown with its headings marked; docrooter turns that into the tree. This page
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
- `doc.meta` (PDFs) holds `engine` (`"pymupdf4llm"`), `n_pages`, `timings` (seconds per stage),
  `bookmarks` (how many were used) and `warnings` (`"no_text_layer"`, `"truncated_pages"`).
- Text before the first heading becomes a level-0 preamble section, which is not in `toc()`.
- `Document.from_sections([...])` builds a document from headings you already have.

The full script is [`examples/toc.py`](../../../examples/toc.py).

## How PDF headings are found

**1. PyMuPDF4LLM reads the pages.** `pymupdf4llm.to_markdown(path, page_chunks=True,
header=False, footer=False)` runs its layout model (PyMuPDF-Layout, ONNX on the CPU) over each
page, classifies the blocks (section header, text, table, picture, page header, footer, ...)
and writes Markdown: section headers become `#` to `######` lines, with the level PyMuPDF4LLM
gives them. Page headers and footers are left out. No OCR runs unless you pass `ocr="auto"`.

**2. docrooter cleans the headings.** Titles lose Markdown emphasis (`**`, `_`), HTML tags
(`<u>`, `<mark>`) and a leading bullet. A heading with no letter or digit left is dropped, and
its text joins the section before. There is no scoring: every heading PyMuPDF4LLM marks is kept.

**3. Levels.** PyMuPDF4LLM's levels are kept, with two adjustments. Gaps are closed so a level
never jumps by more than one below its parent (flag `level_repaired`; `repair=False` keeps the
raw levels). And a heading whose title matches one of the PDF's bookmarks takes the bookmark's
level (flag `level_from_bookmark`; `use_bookmarks=False` turns this off). A title that differs
from the bookmark only slightly (a typo, a plural, joined words: 90% similar) still matches
when the bookmark points at the heading's page or the next or previous one. A bookmarked title
PyMuPDF4LLM left as body text, alone on its line on the bookmark's page, becomes a heading
there (flag `heading_from_bookmark`). The gaps are then closed again, so the headings after a
bookmarked one sit at most one level below it.

**4. Tree.** The headings and the text between them become sections, each with the page it
starts on. Text before the first heading is the level-0 preamble.

Markdown documents skip step 1 and are parsed directly. Their headings still go through title
scoring (false headings below 0.5 are merged into the text before them) and the same level
repair: in a Markdown file whose headings are mostly numbered, `## 1. Business` becomes level 1.

**Element mode** (`elements=True`, the `balanced` and `accurate` profiles) is a separate reader:
it extracts native text lines with PyMuPDF or pdfplumber, types every block (heuristic rules or
the heron layout model) and builds the sections from the typed headings. See
[Elements](elements.md) and [the rooter](rooter.md).

## Engines

Without element mode, `engine` is `auto` (the default), `pymupdf` or `pymupdf4llm`, all read
the section tree the same way, through PyMuPDF4LLM (extra `pdf`, AGPL-3.0). `pdfplumber`
(extra `pdf-mit`, MIT) reads native lines for element mode only. See the
[PDF benchmarks](../benchmarks/pdf.md) for the measured quality.

## Options

```python
plain = load_pdf(path, use_bookmarks=False, max_pages=50)
```

- `use_bookmarks=False` ignores the PDF's outline. Bookmarks are the strongest signal for levels
  when a file has them. (The benchmarks run with `use_bookmarks=False`, because the bookmarks
  are their answer key.)
- `max_pages=N` reads only the first N pages, which bounds time and memory on very long files;
  `meta["warnings"]` then contains `"truncated_pages"`.
- `ocr="auto"` lets PyMuPDF4LLM OCR pages without a text layer, with an OCR engine it can find
  (RapidOCR, Tesseract).
- `repair=False` keeps PyMuPDF4LLM's raw levels.
- `on_event` reports the stages `extract`, `headings` and `assemble`; see [Events](events.md).

## Known limitations

- **Speed.** The layout model reads about 0.3 to 0.4 s per page on a laptop CPU, far slower
  than plain text extraction; use `max_pages` on very long files.
- **Over-marking.** Short bold lines (questions in an interview, labels beside charts, running
  section tabs) can come back as headings at the deepest level. Selection (`max_sections`, levels
  1 and 2 first) usually keeps them out of what a model sees.
- **Weakly styled headings.** Documents whose headings look like body text, such as many court
  opinions, still yield thin trees.
- **No OCR by default.** Scanned PDFs give an empty document (`meta["warnings"]` has
  `"no_text_layer"`) unless `ocr="auto"` finds an OCR engine.
