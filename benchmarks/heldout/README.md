# Held-out PDF corpus

A frozen, labelled set of 40 PDFs for ONE final generalisation check of `docrooter`
(table-of-contents extraction and document-type classification).

**Rule: held-out, never used for tuning.** Nothing in `src/`, the taxonomies, the prompts, the
thresholds or the benchmark scripts may be adjusted by looking at results or documents from this
set. Run it once, report the numbers, and stop. If you need to iterate, do it on the development
corpus in `benchmarks/corpus/`. If this set is ever used for tuning, it is no longer held-out and
a new set must be collected.

**Status: spent.** It was run once, at the frozen commit `72f6135`
([results](../../docs/benchmarks-heldout.md)). Its `pdfplumber` result led to the word-splitting
fix `f3baa43`, which was developed on dev and synthetic PDFs only. Because a result from this set
shaped a change, collect a new held-out set before the next generalisation claim.

## How it differs from the dev corpus

- Same taxonomy: the 18 leaf ids of `benchmarks/corpus/doc_types.json`, unchanged.
- Different documents (no URL in common with `benchmarks/corpus/manifest.jsonl`) and no publisher
  in common (`_meta.publisher_overlap_with_dev` in the manifest is empty). No Berkshire Hathaway,
  Amazon, Tesla, NVIDIA, Google, Apple, Texas Instruments or Raspberry Pi documents.
- Wider spread: UK, EU (Ireland, ETSI, Siemens, Roche), Canadian and Australian sources besides US
  ones; mining, retail-REIT, health, payments, chemicals and public-sector issuers; documents from
  1896 (Plessy v. Ferguson) to 2026; scanned-looking or image-only files (Donoghue v Stevenson has no
  text layer; the U.S. Reports cases, the 1948 Shannon paper and the 1985 Fairfax letter are scans or
  old typesetting); a one-page letter up to an 816-page standard.

## Files

| File | What |
|---|---|
| `manifest.jsonl` | line 1 is a `_meta` record (created_at, the held-out rule, dev-overlap list); then one record per PDF: `id`, `file`, `url`, `title`, `publisher`, `label`, `pages`, `bytes`, `sha256`, `label_rationale`, `bookmarks` |
| `fetch.py` | stdlib downloader that re-fetches every PDF from its original publisher and checks sha256 (`python benchmarks/heldout/fetch.py`) |
| `pdfs/` | the downloaded PDFs; not committed (`benchmarks/heldout/.gitignore` ignores it) |

Gold labels are correct by construction: each document was chosen as a clear instance of its type
and the type was confirmed from first-page text (`label_rationale` quotes what the first page shows).
The one exception is `ho_018`, an image-only scan with no text layer; its title page was confirmed
by rendering page 1. `bookmarks` is the number of PDF outline entries (`pymupdf` `get_toc()`),
which is the reference count for TOC evaluation; it is not a hand-checked TOC gold.

## Counts per label

| Label | Docs |
|---|---|
| `financial.annual_report` | 2 |
| `financial.quarterly_report` | 2 |
| `financial.shareholder_letter` | 2 |
| `financial.earnings_release` | 2 |
| `financial.prospectus` | 2 |
| `financial.proxy_statement` | 2 |
| `legal.contract` | 2 |
| `legal.terms_policy` | 2 |
| `legal.court_opinion` | 3 |
| `legal.legislation` | 3 |
| `research.research_paper` | 2 |
| `research.technical_standard` | 3 |
| `research.white_paper` | 2 |
| `technical.user_manual` | 2 |
| `technical.datasheet` | 3 |
| `government.government_report` | 2 |
| `corporate.sustainability_report` | 2 |
| `corporate.investor_presentation` | 2 |

Total: 40 documents, 18 labels, at least 2 per label.

## Size distribution

- Pages: min 1 (`ho_005`), median 61, max 816 (`ho_025`).
  Buckets: 1-3 pages: 3; 4-20: 9; 21-100: 20; 101-150: 2; over 150: 6.
- Bytes: total 87.2 MB; largest `ho_038` (20.0 MB).
- Bookmarks: 23 documents have 5 or more outline entries, 1 have 1 to 4, 16 have none.

## Sources and licensing

Every file comes from a public URL (no login, no paywall, none from sec.gov) fetched with
`curl -L --fail --max-time 120 -A "docrooter-benchmark/0.1 (research)"` with a pause of
1.5 s between downloads, and each starts with `%PDF`. Sources include company investor sites,
UK, Canadian, Irish and Australian government sites, the Library of Congress, BAILII, arXiv,
standards bodies (Ecma, ETSI) and vendor documentation sites.

The PDFs remain the property of their publishers and are not redistributed in this repository;
the manifest records only URLs, hashes and our own labels and notes. Use is for non-commercial
research evaluation. Several are published under open or government licences (for example UK Open
Government Licence, arXiv, CRAN/GPL documentation) but licences were not audited per file.

Notes:
- Some publishers overwrite files in place. `ho_021` (PIPEDA consolidation, 'current to' date)
  and the `ho_009` Vanguard prospectus are likely to change; a sha256 mismatch from `fetch.py` then
  means the document has been updated, not that the download is broken. Use `--lenient` and note
  the drift when reporting.
- `ho_006` (ADP letters) is an excerpt whose cover still carries a proxy-statement banner; it is
  labelled `shareholder_letter` by content and is a deliberately noisy case.
- `ho_029` and `ho_028` are vendor/foundation explainers and protocol papers labelled `white_paper`;
  `ho_024` (Shannon 1948) is an original journal paper and is labelled `research_paper`.
