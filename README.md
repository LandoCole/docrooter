# docrooter

docrooter extracts a table of contents from documents and classifies each section against a
user-supplied taxonomy. The core install is lightweight; model-backed features, PDF parsing and
LLM backends are optional extras (`docrooter[models]`, `docrooter[pdf]`, `docrooter[llm]`, `docrooter[jev]`).
Note that PDF support relies on PyMuPDF / pymupdf4llm, which is licensed under AGPL-3.0 and is
installed only through the `[pdf]` extra.
