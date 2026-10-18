# docrooter program: from TOC extraction to a document intelligence layer

- **Date:** 2026-10-18
- **Status:** Approved direction. Agreed with the user in conversation on 2026-10-18; the user asked me to plan and build it on my own judgement and report results.
- **Builds on:** v0.1–v0.3, branch `feat/v0.3` (package version 0.2.0).
- **Research:** `docs/superpowers/research/2026-10-18-element-classification-research.md`.

## 1. Why

Today docrooter turns a document into a TOC and classifies the document type and its sections from the headings. That is cheap, and it works well for gating a large corpus: "only embed shareholder letters".

It is not enough for the user's real target: LLM data extraction and RAG over large corpora of long documents (financial filings, fund documents, contracts, reports). Those workflows need to know what each paragraph, table and figure is, not only what each heading says. Then a downstream LLM reads only the parts that matter, at a completeness level the user chooses.

The user's words, paraphrased: docrooter should grow from a simple TOC service into a fuller document classification and summarisation tool for more accurate LLM processing and chunking, with a cheap tier for large-scale classification and an accurate tier when it is worth it. The "rooter" is the plumbing that decides when to OCR, when to extract text, and when to use an LLM, Jev or a local model.

## 2. What docrooter is responsible for (and what it is not)

Agreed split (user, 2026-10-18).

**docrooter owns:**
- Parsing and routing: per page, text extraction or OCR; per task, which engine and backend run, with user-defined engines and shipped defaults.
- Structure: the TOC (as today), plus elements: headings, paragraphs, lists, tables, figures/charts, captions, footnotes, page furniture. Each element carries a page, a bounding box and its place in reading order and in the section tree.
- Labels: the document type, section labels and element tags, each with a calibrated probability.
- Key sentences and key phrases per section and element.
- Embeddings, opt-in, recording model and revision.
- Learning from the user's own corpus: a teacher labels a sample, and a small local student model is trained and calibrated (TELEClass-style). The teacher is a pluggable engine: by default a local open-weight LLM with a permissive licence, helped by free weak labels from document structure. Users may configure their own LLM provider, under that provider's terms.
- Selection: given what the caller needs (tags, a field or a query), a completeness setting and an optional token budget, return which elements to read.

**The user's application owns:** LLM extraction prompts, schemas and validation, the vector database, question answering, orchestration and storage.

A demo extraction and retrieval system is built to validate selection, but it lives outside the package (`benchmarks/extraction/`, never shipped in the wheel).

## 3. Tiers

Each tier is a shipped profile of the router. Users can override any engine within it.

| Tier | Profile | What runs | Typical use |
| - | - | - | - |
| 1 | `fast` | Text layer → TOC → document type and section labels from headings (today's pipeline) | Gating millions of documents |
| 2 | `balanced` | + elements, reading order, OCR only where a page has no usable text, local element tags, key sentences | Chunking and indexing, local and private |
| 3 | `accurate` | + teacher backends (Jev or LLM) on low-confidence elements, table structure, figure tagging | High-value extraction, small volumes |

## 4. Phases

Each phase gets its own spec and implementation plan, written just before it starts, built by subagent-driven development, reviewed per task and as a whole branch. Branches stack: `feat/v0.4` from `feat/v0.3`, and so on. Nothing is pushed or merged without the user.

### v0.4: Elements and the rooter (foundation, torch-free by default)
- The element model and the element builder for born-digital PDFs: paragraphs, lists, tables, figures (including vector charts), captions, footnotes, headers, footers and page numbers.
- Reading order with column detection. This fixes two-column papers, where the TOC currently comes out of order.
- The rooter: an engine registry, per-page text-versus-OCR decisions, per-task engine selection, profiles (`fast`, `balanced`, `accurate`), a config file, and route events that show why each decision was made.
- OCR engines as an optional extra. Scanned pages go through OCR, and their lines feed the same element builder.
- Key sentences and key phrases per section and element, torch-free.
- Opt-in element embeddings.
- `Document.elements` in the output schema; the CLI `docrooter elements`; an Elements tab in the playground.
- **Measured:**
  - element-kind accuracy against a public layout dataset;
  - reading order on multi-column documents;
  - OCR character error rate on rendered pages;
  - no TOC regression on dev.

### v0.5: Corpus, teacher, and the learned element tagger
- A training corpus of thousands of public documents (SEC EDGAR filings and fund documents, government and central-bank reports, court opinions, standards, papers with permissive licences). It has no overlap with the frozen held-out corpus.
- An element tag taxonomy with a definition for every tag, built on the existing section taxonomies, plus a `financial.fact_sheet` document type.
- The TELEClass-style loop:
  1. A cheap model shortlists candidate tags. The teacher, a local open-weight LLM by default, picks among them. Structural weak labels (heading inheritance, EDGAR item membership) are added alongside.
  2. Corpus term mining enriches the taxonomy.
  3. Embedding similarity-gap confidence keeps the confident half.
  4. A BERT-class student is fine-tuned within the 8 GB budget.
  5. Uncertainty sampling sends the hardest elements back to the teacher.
  6. Each tag is calibrated.
- Document type aggregated from element evidence.
- **Measured:**
  - element tags on an independently annotated gold set, never teacher labels;
  - student versus teacher versus zero-shot;
  - cost per 1,000 elements and throughput;
  - public datasets where licences allow;
  - document type on the frozen held-out corpus.

### v0.6: Selection and the completeness dial
- Needs: per-document-type fields mapped to tags (query-independent), and free-text queries scored by embedding similarity with a tag prior.
- Calibrated element relevance.
- The dial: completeness 0–1, mapped to a selection threshold with conformal risk control so that average recall meets the target on calibration data. Context expansion (neighbours, the section opening) and an optional token budget.
- `select()` in the API, `docrooter select` in the CLI, a dial with a recall-and-token readout in the playground.
- The demo extraction and retrieval harness, outside the wheel.
- **Measured:**
  - recall of answer-bearing elements against tokens sent;
  - tokens needed for 95% recall;
  - realised recall against the dial's target, on held-out documents;
  - end-to-end extraction accuracy where an LLM endpoint is available.

### v0.7: Product polish and release candidate
- Docs-site guides for each tier, the router and the training loop.
- Playground updates; performance and memory checks for the element pipeline.
- A Rust decision revisited with element-pipeline profiles.
- Whole-program review; release notes for package version 0.3.0.

## 5. Program-wide rules (carry into every spec)

- The core stays torch-free; models, OCR and ML layout are optional extras. Unit tests stay torch-free.
- Memory: one model-loading process at a time, within an 8 GB budget, on MPS or CPU.
- Licences: defaults are Apache/MIT/BSD-compatible. AGPL or GPL engines are opt-in extras, labelled as such (PyMuPDF is already one).
- Data: the frozen held-out corpus (`benchmarks/heldout/`) is never used for tuning. New gold sets are written before any model sees them, and are kept apart from teacher labels. Dev-tuned numbers are always marked as such.
- **Jev is never a teacher.** TypeSafe's Master Customer Agreement §2.3(b), last updated 23 Sep 2026, forbids using its Output for model distillation or to train models that imitate it. Jev is used only as a runtime engine and as a system we compare against, and Jev outputs never enter training or calibration data. Every training label records its provenance: teacher engine, model and revision.
- **Claude-written labels are only for evaluation gold sets, never for training data.**
- Runtime Jev calls during this program (benchmarks, comparisons) are capped at **US$25 in total**, tracked in a ledger file, and reported.
- Secrets stay in `.env`. Never print or commit them.
- Results claim only what committed results files support.
- Performance depends on corpus size. The docs say plainly that the learned tier needs a large unlabelled corpus from the user for best results, and show the learning curve.

## 6. Success criteria for the program

1. **Tier 1 unchanged:** TOC and document-type numbers on dev and held-out are no worse than v0.3.
2. **Elements:** on the public layout benchmark's born-digital categories, element-kind macro F1 is reported. Paragraphs, tables and figures are each recognised at F1 ≥ 0.8 on the mapped classes. Reading order on two-column dev documents is fixed.
3. **Learned tagger:** on the independent gold set, the student's macro F1 is within 5 points of the teacher's and at least 10 points above zero-shot UltraBERT. It runs locally at ≥ 50 elements per second on the M1 within the memory budget.
4. **Selection:** at completeness 0.9, realised recall on held-out documents is ≥ 0.9 on average, while sending well under the whole document. Report the token share.
5. **Everything is documented:** the docs site, the CLI and the playground cover the new tiers, and every number traces to a committed results file.

These are targets, not promises. Results that miss them are reported as they are.
