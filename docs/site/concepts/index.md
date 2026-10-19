# Concepts

docrooter is a short pipeline. Each stage is a plain function or class you can use on its own.

```text
 file (.pdf / .md / .txt) or Markdown text
        │
        ▼
 load()  ──────────────►  Document            the TOC engine: headings, levels, pages
        │                 sections, toc()
        ▼
 view    ──────────────►  text per section    SectionView / DocumentView, within a token budget
        │                 or per document
        ▼
 backend ──────────────►  scores              keyword, embedding, nli, trained, jev, openai
        │                 per candidate
        ▼
 strategy ─────────────►  a path down         Greedy, Beam(n), Flat
        │                 the taxonomy
        ▼
 results                  DocumentResult      per section: best path, steps, alternatives
                          DocTypeResult       per document: type, confidence, leaf_probs
                          NormalizationResult per section: canonical label
```

- **[The TOC engine](toc-engine.md)** turns a file into a `Document`: a tree of `Section`s with
  titles, levels, page numbers, heading scores and flags.
- **[Elements](elements.md)** are the typed blocks of a document in reading order: paragraphs,
  list items, tables, figures, captions, footnotes and page furniture. They are optional and
  built only in element mode.
- **[The rooter](rooter.md)** is the configurable pipeline, `process()`: per page it chooses the
  text layer or OCR, per task an engine, in one of three profiles (`fast`, `balanced`,
  `accurate`), and it logs why.
- **[Taxonomies](taxonomies.md)** are the categories: a tree of labelled nodes with descriptions
  and examples, written in JSON or CSV. Two families are built in: document types and section
  types.
- **[Views](views.md)** decide what text a backend sees: a section's path, title, lead sentences
  and key phrases, or a whole document's title, statistics, contents and opening text, cut to a
  token budget.
- **[Backends](backends.md)** score a text against candidate nodes: local models, hosted LLMs,
  or your own object.
- **[Strategies](strategies.md)** decide how a text walks down the tree: one greedy path, a beam
  of paths, or every leaf at once.
- **[Results](results.md)** keep every decision, so you can see why a label was chosen and how
  close the alternatives were.
- **[Events](events.md)** report each stage as it runs, with timings; the playground is built on
  them.

Three entry points cover most uses:

| You want | Call | Taxonomy |
| --- | --- | --- |
| The type of a document | `classify_document(doc, taxonomy)` or `Classifier.classify_documents(docs)` | `builtin_taxonomy("doc_types")` or yours |
| A canonical label per heading | `normalize_sections(doc, doc_type=...)` | chosen from the document type |
| Your own categories per section | `Classifier(taxonomy).classify(doc)` | yours |
