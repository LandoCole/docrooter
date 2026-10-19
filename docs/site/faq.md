# FAQ

## Which backend should I use?

| Situation | Backend |
| --- | --- |
| Trying the API, writing tests | `keyword` (no models; not accurate) |
| Data must stay on the machine, or no budget for an API | the default, `embedding` (UltraBERT) |
| Accuracy matters most | `jev` |
| You already run an OpenAI-compatible model server | `openai:<model>` (evaluate it first) |
| You have tens of labelled examples per category | `trained:<bundle>` from `docrooter train` |

## Why is Jev more accurate than the local models? {#why-is-jev-more-accurate}

On document types, Jev was right on 49 of 51 development and 37 of 40 held-out documents; the
local default on 39 and 33. The local models compare *similarity* between the document and each
type's description, which gets the broad group right but cannot apply distinctions such as "not
peer-reviewed" (white paper vs research paper) or "issued by an agency". Jev answers a typed
choice question over the document view and the definitions together. Details and costs are in
[Jev and OpenAI-compatible LLMs](guides/llm-backends.md#why-jev-is-more-accurate).

## Local or hosted? {#local-or-hosted}

Use the local default when documents must not leave the machine, when you need to run offline,
or when about 80% document-type accuracy is enough. Use Jev when accuracy matters and sending
the views (a few hundred tokens per document, not the whole file) to a hosted API is
acceptable; it cost $0.0044 to type 51 documents in the benchmark. You can mix them: type
everything locally and send only low-confidence documents, or only some taxonomy levels, to Jev
(see [Backends](concepts/backends.md#routing-different-backends-for-different-decisions)).

## AGPL or MIT: which PDF engine? {#agpl-or-mit-which-pdf-engine}

- **`pymupdf`** (extra `pdf`) is the default and the fastest (about 1 s per 100 pages). PyMuPDF
  is licensed under the **AGPL-3.0** (or a commercial licence from Artifex). If you distribute
  software that includes it, or run it as part of a network service, the AGPL's obligations may
  apply to your code. docrooter itself is Apache-2.0 and imports PyMuPDF only when it loads a
  PDF.
- **`pdfplumber`** (extra `pdf-mit`) is MIT-licensed. It is about 6 times slower and scored
  about the same on the development corpus (F1 0.711 vs 0.703), but on the held-out set it
  returned only 3 headings for one 816-page specification without raising an error. A candidate
  cause was found and fixed afterwards on other documents; whether it was that file's cause is
  not confirmed.

If you only install `pdf-mit`, `load(path)` uses pdfplumber automatically. This is a summary,
not legal advice.

## Does it work on scanned PDFs?

Yes, in element mode; no, on the default path. `load()` and the `fast` profile read only the
PDF's text layer: a scan without one loads as one empty section, `doc.meta["warnings"]` contains
`"no_text_layer"`, and document typing reports `stopped_reason="empty_text"`. The `balanced` and
`accurate` profiles of [`process()`](concepts/rooter.md) route each page to its text layer or to
OCR, and install-wise OCR is an extra (`docrooter[ocr]`, `docrooter[ocr-mac]` or the Tesseract
binary). See [OCR and layout models](guides/ocr-and-layout.md).

## Why does the keyword backend give such low confidences?

It scores the word overlap (Jaccard) between the text and each node's label, description and
examples. Those overlaps are small and close together, so the probabilities stay nearly flat:
with 19 document types the winner gets well under 20%. It exists
for tests and for trying the API without downloads. Use a real backend for real documents.

## How do I make the TOC better on my documents?

- Check `doc.meta["bookmarks"]`: if the PDF has an outline, docrooter uses it (and it is the
  strongest signal).
- Look at `heading_score` and `flags` on the sections you expected or did not expect.
- Raise `min_heading_score` (default 0.55) for fewer false headings, lower it for more recall.
- Try `engine="both"`, which keeps PyMuPDF headings that pdfplumber agrees with.
- If you already have headings from another tool, build the document with
  `Document.from_sections([...])` and use the rest of docrooter on it.

## Can I use it without a PDF at all?

Yes. `load()` takes Markdown text or a `.md` / `.txt` file, and `Classifier.classify_texts`
classifies plain strings against a taxonomy.

## Is the model download safe? What is `trust_remote_code`?

The default model ships its own modelling code. docrooter runs it only at a pinned commit, so a
later change to the model repository cannot change what runs on your machine. Other models that
need remote code are refused until you opt in. See
[Backends](concepts/backends.md#trust_remote_code).

## Limitations

- **OCR is optional and only in element mode** (`balanced`, `accurate`); `load()` and `fast` give
  scanned PDFs as empty documents.
- **TOC extraction is weaker on unusual layouts**: earnings releases with bold body-size
  headings, slide decks, bilingual parallel-column documents, and documents whose cover title
  spans two heading lines. On the held-out set the default engine's F1 at depth ≤ 2 was 0.576.
- **Headings set below body size** (some small-caps styles) are never detected, and a real
  heading repeated at the same height in the page margins of 3 or more pages is treated as a
  running header.
- **English-centric.** The built-in taxonomies, the title heuristics and the benchmarks are
  English. Other languages may work with a multilingual embedding model but are untested.
- **Small benchmarks.** The PDF corpora have 2-4 documents per type, and the section labels are
  silver (assigned by an agent, not checked by a human). Treat differences of a few points as
  noise.
- **Few-shot training of document types did not transfer** to new publishers at this corpus
  size; it does help section-level tasks with more examples per class.
- **The keyword backend is not a real classifier.**
- **The playground is a local, single-user tool** without authentication.
- **No stable API yet.** The version is below 1.0, so a minor release may change the API;
  changes are listed in the [changelog](about/changelog.md).
