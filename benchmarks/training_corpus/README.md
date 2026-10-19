# docrooter v0.5 training corpus

About 1,100 public, native-PDF documents, finance-heavy, used to train and calibrate the learned element tagger (spec `docs/superpowers/specs/2026-10-18-docrooter-v0.5-learned-tagger-design.md` §5). No PDF is committed or redistributed. What is committed is `manifest.jsonl`, which holds URLs, checksums, licences and splits, plus the scripts that fetch the PDFs again.

## Files

| File | What it is |
| - | - |
| `fetch.py` | The fetcher: `python -m benchmarks.training_corpus.fetch`. |
| `sources/*.py` | One module per source. Each lists candidate PDFs and states its licence and doc-type hint. |
| `_net.py` | Polite HTTP: the User-Agent, per-host rate limits, robots.txt, cached listings, and capped, checked downloads. |
| `pdf_meta.py` | Reads the page count and text-layer size of one PDF. It runs in a `python -I` child process. |
| `manifest.jsonl` | One record per document. It is written by the fetcher. |

Everything else lives in `benchmarks/.cache/training_corpus/`, which is git-ignored:
- `pdfs/<id>.pdf`;
- `fetched.jsonl`, the per-document checkpoint;
- `drops.jsonl`;
- `listings/`, the cached API and index responses;
- `summary.json`;
- `dry_run.jsonl`.

## Reproducing

```
PYTHONPATH=src .venv/bin/python -m benchmarks.training_corpus.fetch --target 1100
PYTHONPATH=src .venv/bin/python -m benchmarks.training_corpus.fetch --dry-run --target 30
PYTHONPATH=src .venv/bin/python -m benchmarks.training_corpus.fetch --sources fed arxiv --target 60
```

- **Resuming.** The fetcher resumes: a re-run skips every document already fetched or permanently dropped, and reads its listings from the cache. Delete `listings/` to list afresh.
- **Duration.** A full run takes a few hours. Most of the time goes to arXiv's 15 s crawl delay, 1 request/s elsewhere, and three `DEMO_KEY` listing requests spaced 6 minutes apart.
- **Hashes.** Publishers sometimes replace a file at the same URL, so a later fetch can differ from the manifest's sha256.

## Manifest record

`id`, `url`, `source`, `licence`, `sha256`, `bytes`, `pages`, `fetched_at`, `doc_type_hint`, `family_hint`, `split`, `title` and `publisher`, plus source extras:
- arXiv: `arxiv_id`, `authors` (for CC BY attribution) and `categories`;
- EDGAR: `accession` and `cik`;
- funds: `sponsor`, `ticker` and `kind`;
- others: the source's own id or date.

`doc_type_hint` is a docrooter doc type id for the source's own type. `family_hint` is the element topic family. `docrooter corpus build DIR --from-manifest benchmarks/training_corpus/manifest.jsonl --pdf-dir benchmarks/.cache/training_corpus/pdfs` reads it directly.

**Splits.** Splits are assigned by document and stratified by source, with seed 0:
- 10% `gold` (rounded);
- 5% `calib`;
- the rest `train`, about 85%.

They are a pure function of the set of documents, so the full fetch fixes them, and the L5 gold set is drawn after it.

## Sources

Spec targets are given at 1,100 documents. `--target` scales targets and caps. Shortfalls are made up in this order, finance first, each source up to its cap: `edgar_ars`, `fund_docs`, `fed`, `edgar_ex10`, `crs`, `govinfo_gao`, `arxiv`, `uscourts`, `rfc`.

| Source | Spec target | Cap | What, and from where | Licence / status | Availability (2026-10-19) |
| - | - | - | - | - | - |
| `edgar_ars` | 300 | 450 | Form ARS glossy annual reports. PDF main documents only (no exhibits, no HTML), one per accession, March-April 2024-2026. Listed through EDGAR full-text search. | SEC public filing, not redistributed. | **Unavailable.** www.sec.gov answers 403 to the project User-Agent, which has no contact email by ruling. Skipped with no retries. |
| `fund_docs` | 150 | 284 | 284 explicit sponsor URLs: 154 SSGA SPDR fact sheets, plus 65 iShares annual and 65 semi-annual shareholder reports. | Sponsor documents, local research use only, never redistributed. | **Works.** 284 of 285 URLs answered with a PDF on 2026-10-19. |
| `govinfo_gao` | 150 | 250 | GAO reports issued 2004-2008, `docClass` REPORT, through the govinfo `published` API. PDFs come from www.govinfo.gov/content. | US government work, not copyrighted (third-party images excepted). | **Works with `DEMO_KEY`.** One request lists 1,000 packages. About 800 are reports. |
| `crs` | 100 | 200 | CRS Reports, active or archived, through api.congress.gov `crsreport`. PDFs come from www.congress.gov/crs_external_products, with a 2 s crawl delay. | US government work. Copyright status unverified, and reports may contain third-party material. | **Works with `DEMO_KEY`.** About 160 Reports per 250-item request, with up to 3 requests. |
| `fed` | 60 | 220 | FEDS and IFDP working papers (2022-2025 index pages) and FOMC minutes. | Public domain unless otherwise indicated. | **Works.** No robots.txt. About 550 papers and 46 minutes are listed. |
| `arxiv` | 150 | 250 | OAI-PMH `arXiv` format, sets q-fin, econ, stat and cs (2025). Only records whose licence is CC BY 4.0 or CC0 are kept. | CC-BY-4.0 / CC0-1.0. Authors are kept for attribution. | **Works.** About 35% of q-fin records qualify. OAI requests are spaced 3 s apart, and PDF requests 15 s apart (robots crawl delay). |
| `uscourts` | 100 | 150 | Federal appellate opinions from one govinfo `search` request (1,000 results), interleaved across circuits. PDFs come from www.govinfo.gov/content. | US government work, verify per document. | **Works with `DEMO_KEY`.** About 13 circuits. |
| `rfc` | 40 | 60 | RFCs 8650-9850 (the v3 format has a PDF rendering), in an order shuffled with seed 0. | IETF Trust Legal Provisions: unmodified copies only. | **Works.** |
| `edgar_ex10` | 50 | 75 | Exhibit-10 material contracts filed as PDF, through EDGAR full-text search. | SEC public filing, not redistributed. | **Unavailable**, for the same reason as `edgar_ars`. Even with access, PDF exhibits are rare because EDGAR's official exhibit format is HTML. |

Without EDGAR, the 350-document shortfall is made up as follows:
- `fund_docs` grows to its cap of 284;
- `fed` grows to its cap of 220;
- `crs` covers the rest.

Finance therefore comes from fund documents, Federal Reserve papers and minutes, and q-fin/econ arXiv papers, not from corporate annual reports. To bring EDGAR back, rerun with a User-Agent that the SEC accepts. That needs a real contact, which is the user's decision.

## Rules the fetcher follows

- **User-Agent.** `docrooter-research/0.5 (research; contact via repository)`, with no personal data.
- **Rate limits.** Per host or bucket:
  - SEC: at most 5 requests/s, shared by all SEC hosts;
  - arXiv: 1 request per 3 s across hosts, and PDFs at most 1 per 15 s, the arxiv.org crawl delay;
  - api.data.gov `DEMO_KEY`: 1 request per 6 minutes, shared by govinfo and congress.gov. That is 10/hour, as its response headers report;
  - everything else: 1 request/s, or the robots.txt crawl delay if longer.
- **robots.txt.** robots.txt is obeyed, with `*`/`$` patterns and the longest match winning. A 401/403 or a server error means "do not crawl". Documented API hosts are exempt.
- **No aggressive retries.** A 401/403/429 stops that host for the run. Network errors are retried only by the next run.
- **API keys.** Keys come from the process environment only: `GOVINFO_API_KEY`, `CONGRESS_API_KEY` or `DATA_GOV_API_KEY`. `.env` is never read. Without a key, `DEMO_KEY` is used for the three listing requests, because one request lists hundreds of documents.
- **Untrusted downloads.**
  - Each PDF is written to `<id>.pdf.part`, then renamed into place.
  - A PDF must start with `%PDF-` and be at most 50 MB.
  - It is parsed only by `pdf_meta.py` under `python -I` and is never executed.
  - Scans are dropped: a PDF needs at least 200 text characters in its first five pages.
  - Byte-identical duplicates are kept once.
- **Exclusion (no dev or held-out document).** Only `benchmarks/corpus/manifest.jsonl` and `benchmarks/heldout/manifest.jsonl` are read, and only their `sha256`, `url` and `title` fields. Nothing else under `benchmarks/heldout/` is opened. A document is dropped when any of these match:
  - its normalised URL;
  - its normalised title;
  - its sha256, so the same bytes re-hosted elsewhere are dropped too.

  Spec §5 matches on (publisher, title). The publisher is not read, by ruling, so titles match without it, which can only drop more. Held-out drops are logged without their URL or title.

## Counts

The fetcher rewrites this block at the end of every non-dry run.

<!-- counts:begin -->

Last fetch: target 10, 10 documents in the manifest (train 10, calib 0, gold 0).

| Source | Scaled target | Documents | Made up | Shortfall | Status |
| - | - | - | - | - | - |
| edgar_ars | 3 | 0 | 0 | 3 | unavailable: https://www.sec.gov/Archives/edgar/data/: robots.txt refused: www.sec.gov answered HTTP 403; not asking it again this run |
| fund_docs | 1 | 3 | 2 | 0 | ok |
| govinfo_gao | 1 | 1 | 0 | 0 | ok |
| crs | 1 | 2 | 1 | 0 | ok |
| fed | 1 | 2 | 1 | 0 | ok |
| arxiv | 1 | 1 | 0 | 0 | ok |
| uscourts | 1 | 1 | 0 | 0 | ok |
| rfc | 0 | 0 | 0 | 0 | ok |
| edgar_ex10 | 1 | 0 | 0 | 1 | unavailable: https://www.sec.gov/Archives/edgar/data/: robots.txt refused: www.sec.gov answered HTTP 403; not asking it again this run |
| **total** | 10 | 10 | | | |

Drops this run: none.

<!-- counts:end -->
