# Views

Backends never see a raw section or a whole PDF. They see a **view**: a short text rendered from
the document, as `Label: content` lines within a token budget. Choosing what goes into the view
is often worth more accuracy than choosing a bigger model, and printing the view is the first
thing to check when a label looks wrong.

## SectionView: one text per section

Used by `Classifier.classify` and `normalize_sections`.

```python
view = SectionView(fields=("path", "title", "lead"), max_tokens=64)
for section_id, rendered in view.render_all(doc, [s.id for s in doc.leaves()]).items():
    print(f"[{section_id}] {rendered.n_tokens} tokens, truncated={rendered.truncated}")
    print(rendered.text)
```

```text
[s1.1] 28 tokens, truncated=False
Path: Annual report > Income taxes
Title: Income taxes
Lead: The effective tax rate was 21.4 percent. Deferred tax assets rose.
```

| Field | Renders |
| --- | --- |
| `path` | The titles from the top of the tree down to this section |
| `title` | The section title |
| `keyphrases` | Up to `keyphrases` (8) distinctive phrases of the section, scored against the rest of the document |
| `lead` | The first `lead_sentences` (3) sentences |
| `text` | The whole body; for an empty section, its children's titles (`empty_section`) |

The default is all five fields within `max_tokens=512`. Fields are rendered in the order given;
the first field that does not fit is cut to the longest prefix that does, and the rest are
dropped (`rendered.truncated` is then true). A view is never empty: it falls back to the title.

Pass a view to the classifier to change what every section contributes:

```python
classifier = Classifier(taxonomy, "keyword", view=view, threshold=0.0)
```

`normalize_sections` uses `path` + `title` (96 tokens) when the section taxonomy has
`examples` (the built-in ones do), and `title` + `lead` otherwise; this was chosen on the
benchmark silver set.

## DocumentView: one text per document

Used by `classify_document` and `Classifier.classify_documents` to decide the document type.

```python
doc_view = DocumentView(fields=("title", "stats", "toc", "lead"), max_tokens=128)
print(doc_view.render(doc).text)
```

```text
Title: Annual report
Stats: 3 sections
Contents: 1 Annual report; 1.1 Income taxes; 1.2 Risk factors
Lead: The effective tax rate was 21.4 percent. Deferred tax assets rose. Our business faces risks from competition and regulation.
```

| Field | Renders |
| --- | --- |
| `title` | The document title (else the file name) |
| `stats` | Page and section counts |
| `toc` | Numbered headings down to `toc_depth` (2), at most `toc_entries` (40) |
| `lead` | The first `lead_words` (120) words of body text, skipping the cover, notices, contents, disclaimers and forward-looking statements |
| `keyphrases` | Up to `keyphrases` (12) phrases pooled over the sections |

The default is all five fields within 512 tokens. On the development corpus, Jev classified 49
of 51 documents correctly from this view, using a tenth of the tokens of sending raw page text
in chunks; see the [benchmarks](../benchmarks/index.md).

## Token counting

Budgets are counted with the backend's own tokenizer when it has one, and estimated (1.3
tokens per word) otherwise. The view's `n_tokens` reports the count used.

The script with both views is [`examples/views.py`](../../../examples/views.py).
