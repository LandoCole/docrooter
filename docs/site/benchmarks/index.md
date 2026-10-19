# Benchmarks

docrooter is measured on three kinds of data. The measured numbers on these pages come from
results files committed under `benchmarks/results/`, produced by the scripts in `benchmarks/` at
a recorded commit. A few figures from development runs (some memory peaks and an A/B timing on
the performance page) are not in a results file; the pages say so where they appear.

| Page | Data | What it measures |
| --- | --- | --- |
| [PDF benchmarks (dev corpus)](pdf.md) | 51 real PDFs, 18 document types | TOC extraction, document type, section normalisation |
| [Held-out evaluation](heldout.md) | 40 different PDFs, frozen before any run | The same three tasks, once, with no tuning |
| [Element kinds (DocLayNet)](elements.md) | 400 DocLayNet test pages, six categories | Element kinds against human layout labels |
| [Public datasets](public-datasets.md) | EDGAR-CORPUS 10-K items, DBpedia classes, LEDGAR clauses | Zero-shot section classification by backend |
| [Performance and memory](performance.md) | ten PDFs, 1-642 pages; a training run | Peak memory, where the time goes, load speed |

## Headline results

**Development vs held-out.** The development corpus was used to choose the defaults (heading
rules, views, thresholds), so its numbers are optimistic. The held-out corpus shares no
documents or publishers with it and was evaluated once, at a frozen commit; it is the honest
estimate.

| Task | System | Dev | Held-out |
| --- | --- | ---: | ---: |
| TOC, F1 at depth ≤ 2 (micro) | `pymupdf` (default engine) | 0.703 | 0.576 |
| TOC, F1 at depth ≤ 2 (micro) | `pdfplumber` | 0.711 | 0.458 |
| TOC, F1 at depth ≤ 2 (micro) | `both` | 0.716 | 0.593 |
| Document type, accuracy | `jev-view` (Jev on a `DocumentView`) | 0.961 | 0.925 |
| Document type, accuracy | `jev-chunks` (Jev on raw page text) | 0.922 | 0.975 |
| Document type, accuracy | `ultrabert-view` (local default, zero-shot) | 0.765 | 0.825 |
| Document type, accuracy | `bge-view` (bge-small, zero-shot) | 0.608 | 0.700 |
| Document type, accuracy | trained on dev, tested on held-out | 0.706 (LOO) | 0.675 |
| Section normalisation, accuracy | `jev` | 0.725 | 0.692 |
| Section normalisation, accuracy | `ultrabert` (local default) | 0.514 | 0.579 |
| Section normalisation, accuracy | type classified automatically | 0.479 | 0.406 |

What these say:

- **Document typing generalises.** Jev stays at 0.92-0.98, and the local zero-shot views do
  better on held-out than on dev. A head *trained* on the dev documents does worse on new
  publishers than zero-shot, so the zero-shot default is the right shipping choice.
- **Section normalisation holds** for the local models and drops slightly for Jev. Passing the
  document type explicitly matters: the automatic path loses 17 points on held-out.
- **TOC extraction is the weak spot.** The default engine drops about 12 points at depth ≤ 2.
  About half of that comes from bookmark sets that are not tables of contents (design-tool page
  labels, "Slide N:" titles, a scan); the rest is real (bold body-size headings, a bilingual
  statute, a split cover title). `pdfplumber` returned only 3 headings for an 816-page
  specification, which drives its collapse.
- **Small samples.** With 40 documents, one document is 2.5 points, and the 95% intervals of
  document-type accuracy are 25-30 points wide for the local models (7.5-17.5 for Jev). Only
  large differences mean anything.

**Public datasets (zero-shot, hierarchical F1).** UltraBERT is the best local backend on 10-K
items (0.752) and LEDGAR clauses (0.381); every zero-shot backend is weak on DBpedia's
fine-grained classes (0.11-0.27), where a few labelled examples per class help most (see
[Training and evaluation](../guides/training-evaluation.md#when-does-training-help)).

**Speed.** Loading a PDF with the default engine takes about 1.05 s per 100 pages. In the full
local pipeline, PyMuPDF's text extraction and model inference take most of the time;
docrooter's own Python is about 13%.

## Reproducing

The benchmark corpora are fetched from their public sources by scripts in `benchmarks/`; the
PDFs themselves are not committed. See
[`benchmarks/README.md`](../../../benchmarks/README.md) for the setup, and run one model
process at a time.
