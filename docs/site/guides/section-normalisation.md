# Section normalisation

Different documents of the same type name the same section differently: "Item 1A. Risk
Factors", "Principal risks and uncertainties", "Risk management". Normalisation maps each
heading to a **canonical** section of its document type (`risk_factors`), so downstream code
can ask for "the risk section" of any annual report.

## The short version

```python
norm = normalize_sections(doc, doc_type="legal.contract", backend="keyword")
print(norm.taxonomy, norm.doc_type)
for section_id, section in norm.sections.items():
    print(f"{section_id} {section.title!r} -> {section.canonical} ({section.score:.2f})")
```

```text
section_types/contract legal.contract
s1 'Master Services Agreement' -> term (0.10)
s2 '1. Definitions' -> definitions (0.18)
s3 '2. Term' -> term (0.12)
s4 '3. Fees and payment' -> payment (0.22)
s5 '4. Termination' -> termination (0.10)
```

From the shell:

```bash
docrooter normalize contract.pdf --doc-type legal.contract
docrooter normalize report.pdf --levels 1,2,3 --json
```

The script is [`examples/normalize.py`](../../../examples/normalize.py). It uses the keyword
backend, which is why the title line gets a label and the scores are low; leave `backend` out
to use the default model.

## How it chooses the section taxonomy

1. `taxonomy=` if you pass one: a `Taxonomy`, or a built-in name (`"contract"` or
   `"section_types/contract"`).
2. Otherwise the taxonomy mapped from `doc_type=`: annual and quarterly reports and proxy
   statements use `annual_report`; contracts `contract`; research papers `research_paper`;
   manuals and datasheets `technical_document`; every other type (and an unknown one)
   `generic`. A group id such as `"financial"` uses its first leaf.
3. With neither, it first classifies the document type with `classify_document`, using the same
   backend, and records `doc_type_confidence`.

```python
norm = normalize_sections(doc, taxonomy="contract", backend="keyword", levels=[1])
auto = normalize_sections(doc, backend="keyword")
print(auto.doc_type, round(auto.doc_type_confidence, 2), auto.taxonomy)
```

**Pass `doc_type` when you know it.** The automatic path makes one decision that can be wrong
with low confidence, and a wrong type selects the wrong section taxonomy for every section. On
the held-out corpus it cost 17 points of accuracy against the known type (0.406 vs 0.579 with
UltraBERT): one contract typed as "terms and policies" at confidence 0.21 put all 40 of its
sections in the generic taxonomy. Typing the document first with
[document-type gating](doctype-gating.md), and checking the confidence, avoids that.

## Which sections are labelled

- `levels` (default `(1, 2)`) chooses the heading levels. The preamble (level 0) never is.
- Every selected section always gets a label (the classification is `Flat`, threshold 0), even
  when nothing fits well; use `score` to judge it. Only the annual-report taxonomy has an
  `other` node, so a cover line or an unusual clause is forced into the nearest section.
- Each section is classified from its heading path and title (with the built-in taxonomies,
  whose nodes list heading synonyms as `examples`), or from its title and lead sentences (for a
  custom taxonomy without examples). An explicit `view=` always wins.

## Your own canonical sections

Any taxonomy works. Give each node the headings you see in practice as `examples`:

```python
custom = normalize_sections(doc, taxonomy=mine, backend="keyword")
print({section.title: section.canonical for section in custom.sections.values()})
```

To reuse a loaded model across many documents, pass a backend instance or a prebuilt
`classifier=` (built on the chosen taxonomy); `normalize_sections` closes only the models it
loaded itself.

## How accurate is it?

Accuracy against silver labels (assigned by an agent, not checked by a human) for the
level-1/2 sections of 6 development documents (142 scored sections) and 4 held-out documents
(133). The defaults (the `path + title` view and the heading synonyms) were chosen on the dev
sections, so the *Dev* column is optimistic:

| Backend | Dev | Held-out |
| --- | ---: | ---: |
| `jev` | 0.725 | 0.692 |
| UltraBERT (the default) | 0.514 | 0.579 |
| bge-small | 0.408 | 0.496 |
| UltraBERT with the type classified automatically | 0.479 | 0.406 |

Many errors are between neighbouring sections ("risk management" vs "corporate governance"),
and old or unusual structures (Shannon's 1948 paper in five Parts) fit no modern taxonomy. See
the [section normalisation benchmarks](../benchmarks/pdf.md#section-normalisation).
