# OCR and layout models

Two optional pieces make element mode work on scans and on hard layouts. OCR reads pages that
have no usable text layer. The heron layout model finds tables, figures, captions and headers
that the heuristic rules miss. Both are extras, imported only when used, and neither pulls in
torch. This guide covers installing and choosing them; [the rooter](../concepts/rooter.md)
explains when a page is sent to OCR.

## Install

| Extra | Adds | Used for |
| --- | --- | --- |
| `ocr` | `rapidocr`, `onnxruntime`, `pypdfium2` | The `rapidocr` engine; `pypdfium2` renders pages when PyMuPDF is not installed |
| `ocr-mac` | `ocrmac` (macOS only) | The `ocrmac` engine, which uses Apple Vision |
| `layout` | `onnxruntime`, `pypdfium2`, `huggingface_hub` | The `heron` layout model |
| *(none)* | Tesseract, the system binary | The `tesseract` engine: `brew install tesseract` or `apt install tesseract-ocr` |

```bash
pip install 'docrooter[pdf,ocr]'               # PyMuPDF + RapidOCR
pip install 'docrooter[pdf,ocr-mac]'           # PyMuPDF + Apple Vision (macOS)
pip install 'docrooter[pdf,layout]'            # PyMuPDF + the heron layout model
pip install 'docrooter[pdf-mit,ocr,layout]'    # the MIT-only PDF stack
```

Pages are rendered for OCR and for the layout model with PyMuPDF when it is installed, and with
`pypdfium2` otherwise, so the permissive stack (`pdf-mit` with `ocr` or `layout`) never imports
PyMuPDF. If an engine's package or binary is missing, the error names what to install. The
engines are never swapped silently: a profile that asks for `heron` fails when `onnxruntime` or
the weights are missing, rather than falling back to the heuristic rules.

## Choosing an OCR engine

`ocr="auto"` is the default of `balanced` and `accurate`. It picks the first engine that is
available, in the order of the measured error rate:

1. `tesseract`, if the binary is on PATH;
2. `ocrmac`, on macOS with `docrooter[ocr-mac]`;
3. `rapidocr`, with `docrooter[ocr]`.

With none available, `auto` becomes `none`: pages that need OCR keep their text layer and the
route log carries a warning. Name an engine to override the choice:

```bash
docrooter elements scan.pdf --profile balanced --config rooter.toml
```

with `ocr = "rapidocr"` in the file (see [Configuration](../concepts/rooter.md#configuration)).
`ocr_lang` sets the language (`"en"` by default; Tesseract takes codes such as `deu+eng`;
RapidOCR's default models read English and Chinese) and `ocr_dpi` the resolution pages are
rendered at (200).

### How the engines compare

Measured on 60 pages rendered at 200 dpi into image-only PDFs from 20 development PDFs. The
ground truth is the page's own text layer. The error rates are pooled over all characters and
words after normalisation:

| Engine | Character error rate | Word error rate | OCR s per page | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| `tesseract` 5.5.1 (`--psm 3`, `eng`) | 0.058 | 0.168 | 1.72 | 803 MiB + 263 MiB for the binary |
| `rapidocr` 3.10.0 (CPU) | 0.089 | 0.196 | 3.04 | 3,311 MiB |
| `ocrmac` 1.0.1 (Apple Vision) | 0.102 | 0.208 | 0.73 | 1,184 MiB |

Source: `benchmarks/results/2026-10-18/ocr.json`; method in
[Element benchmarks](../benchmarks/elements.md#ocr-and-routing-spec-104).

- Tesseract is the most accurate here, and the order of `auto` follows it.
- `ocrmac` is the fastest and fine on prose, but it fails on legal pages (three CHIPS Act pages
  scored a character error rate of 0.51 to 0.65, against 0.04 to 0.06 for the other engines)
  and can miss number columns in tables. Choose it for speed on clean prose, not for filings.
- `rapidocr` makes small errors on most pages, and is the slowest and the largest in memory.
- The pages are clean renders of born-digital pages, not photographs of paper. Expect worse
  numbers on real scans.
- The reading order of an OCR'd page depends on how the engine cuts lines. If it returns one line
  across a gutter, XY-cut cannot separate the columns. In a smoke test on a short two-column page
  rendered to an image, Tesseract returned lines that ran across both columns, so the two
  columns came out interleaved in a single paragraph.
- OCR'd pages have no ruled tables, no bold and no font information. Tables become paragraphs,
  headings come from size and isolation only, and figures are missed. In the same 60 pages the
  element kind of an OCR'd element agreed with its text-layer twin for 83% to 89% of elements.

## Your own OCR engine

An engine is an object with a `name`, a `recognize(image, *, width_px, height_px, page)` method
and a `close()` method. `recognize` receives a PNG of the page rectangle `page` (a `Box` in
points) and returns a list of `OcrLine(text, box, confidence)` objects, with boxes already in
page points and confidences from 0 to 1; the order of the lines is yours, because XY-cut orders
the page afterwards. `register_ocr_engine(name, factory)` adds the engine to the registry, where
`factory(lang=...)` returns it, so `ocr="name"` works in `process()` and in `--config` files. An
engine object you pass to `process()` yourself is yours to close. `register_layout_engine(name,
factory)` does the same for a layout model: its `detect(path, pages, *, dpi=None)` returns
`LayoutRegion` objects.

## The heron layout model

The `accurate` profile (or `layout="heron"`) runs Docling's heron layout model, an RT-DETR
detector in ONNX form (`docling-project/docling-layout-heron-onnx`, Apache-2.0), on onnxruntime.
The 171 MB weights are downloaded once through `huggingface_hub`, at a pinned revision, into the
Hugging Face cache. They are data, not code, so no remote code runs. The session is created on
the first page, reused for the whole run (and across documents in `process_many`) and released
on `close()`.

```bash
docrooter elements report.pdf --profile accurate
```

Each detected region with confidence of at least 0.5 sets the kind of the lines inside it; lines
outside every region keep the heuristic rules, and regions are read in XY-cut order. A Table
region that the ruled-table search did not find is read with the engine's text-table strategy.

### Heuristic or heron

This is the trade-off, on 400 DocLayNet test pages (see
[Elements](../concepts/elements.md#heuristic-or-layout-model)):

| | Heuristic (`balanced`) | Heron (`accurate`) |
| --- | ---: | ---: |
| Macro F1 over 11 classes | 0.257 | 0.736 |
| Text coverage of Text | 0.927 | 0.949 |
| Seconds per 100 pages (ten development PDFs) | 4.64 | 64.6 |

`balanced` is fast with good text coverage: nearly every character of body text lands in a
paragraph. `accurate` is for structure quality: tables, figures, captions, headers and footers
and headings. It costs about 14 times the time of `balanced` and 65 times that of `fast`.
Sources: `benchmarks/results/2026-10-18/elements-doclaynet.json` and
`benchmarks/results/2026-10-18/profile-elements.json`. The DocLayNet scores were produced before
the XY-cut table fix (commit `c09b9f7`) and were not re-scored.

## Making it faster

| Per 100 pages | Seconds |
| --- | ---: |
| `fast` | 0.99 |
| `balanced`, PyMuPDF | 4.64 |
| `balanced`, pdfplumber | 7.65 |
| `accurate` (heron) | 64.6 |

`tables="none"` is the speed switch in element mode: it skips the search for ruled tables, which
is most of the extra cost over `fast`, at the price of table elements. Set it in a configuration
file, or pass `tables="none"` to `process()` or `load_pdf(elements=True)`. Use `fast` when you
do not need elements.

Page furniture is the one visible difference in order: in element mode, running headers, footers
and page numbers move to the start or end of their page, by design, where `fast` leaves them where
the extractor found them.

## Memory

At most one model-loading process runs at a time, inside the 8 GiB budget (see
[Memory and performance](memory-performance.md)). The largest model process measured was
RapidOCR at 3.3 GiB; heron peaked at 904 MiB. `process_many` loads each engine once for the whole
run.
