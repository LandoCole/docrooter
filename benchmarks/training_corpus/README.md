# docrooter v0.5 training corpus

About 1,100 public, native-PDF documents, finance-heavy, used to train and calibrate the learned element tagger (spec `docs/superpowers/specs/2026-10-18-docrooter-v0.5-learned-tagger-design.md` §5). No PDF is committed or redistributed. What is committed is `manifest.jsonl`, which holds URLs, checksums, licences and splits, plus the scripts that fetch the PDFs again.

## Files

| File | What it is |
| - | - |
| `fetch.py` | The fetcher: `python -m benchmarks.training_corpus.fetch`. |
| `sources/*.py` | One module per source. Each lists candidate PDFs and states its licence and doc-type hint. |
| `_net.py` | Polite HTTP: the User-Agent, per-host rate limits, robots.txt, vetted redirects, cached listings, and capped, checked downloads. |
| `pdf_meta.py` | Reads the page count and text-layer size of one PDF. It runs in a `python -I` child process with a memory cap. |
| `manifest.jsonl` | One record per document. It is written by the fetcher. |

Everything else lives in `benchmarks/.cache/training_corpus/`, which is git-ignored:
- `pdfs/<id>.pdf`;
- `fetched.jsonl`, the per-document checkpoint;
- `drops.jsonl`;
- `listings/`, the cached API and index responses;
- `archives/cuad/CUAD_v1.zip`, the CUAD archive, downloaded once and checked against a pinned sha256;
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
- **EDGAR.** `edgar_ars` and `edgar_ex10` are kept but run only when named with `--sources`, because www.sec.gov refuses the project User-Agent.

## Manifest record

`id`, `url`, `source`, `licence`, `sha256`, `bytes`, `pages`, `fetched_at`, `doc_type_hint`, `family_hint`, `split`, `title` and `publisher`, plus source extras:
- arXiv: `arxiv_id`, `authors` (for CC BY attribution) and `categories`;
- CUAD: `cuad_file` (the PDF's path in the archive) and `attribution`;
- IR annual reports: `year`;
- funds: `sponsor`, `ticker` and `kind`;
- EDGAR: `accession` and `cik`;
- others: the source's own id or date.

Ids match `^[a-z0-9_-]{1,80}$`; the fetcher refuses any other id. `doc_type_hint` is a docrooter doc type id for the source's own type. `family_hint` is the element topic family. `docrooter corpus build DIR --from-manifest benchmarks/training_corpus/manifest.jsonl --pdf-dir benchmarks/.cache/training_corpus/pdfs` reads it directly.

**Splits.** Each document's split comes from its id alone: `sha256("split:" + id)` read as a number in [0, 1) gives:
- `gold` below 0.10;
- `calib` below 0.15;
- `train` otherwise, about 85%.

A document that already has a split in the manifest keeps it. Adding or removing documents therefore never moves another document, and a gold document can never drift into training. Stratification by source holds in expectation; the counts block below reports the realised shares per source.

## Sources

Spec targets are given at 1,100 documents. `--target` scales targets and caps. Shortfalls are made up in this order, finance first, each source up to its cap: `ir_annual_reports`, `fund_docs`, `fed`, `cuad`, `crs`, `govinfo_gao`, `arxiv`, `uscourts`, `rfc`.

| Source | Spec target | Cap | What, and from where | Licence / status | Availability (2026-10-19) |
| - | - | - | - | - | - |
| `ir_annual_reports` | 250 | 255 | 255 explicit URLs of company annual reports for 2019-2025, 55 companies across banking, insurance, pharma, industrials, telecoms, retail and media. Each URL is on the company's own IR site. Replaces EDGAR Form ARS. | The publisher's document: local research use only, never redistributed. | **Works.** Every URL answered a 1 KB range request with a PDF of at most 50 MB on 2026-10-19. The 2023 Berkshire report is a dev document and is excluded. |
| `fund_docs` | 150 | 349 | 349 explicit sponsor URLs: 154 SSGA SPDR fact sheets, 65 iShares summary prospectuses, and 65 annual and 65 semi-annual iShares shareholder reports. | Sponsor documents: local research use only, never redistributed. | **Works.** All 349 URLs answered with a PDF. |
| `govinfo_gao` | 150 | 250 | GAO reports issued 2004-2008, `docClass` REPORT, through the govinfo `published` API. PDFs come from www.govinfo.gov/content. | US government work, not copyrighted (third-party images excepted). | **Works with `DEMO_KEY`.** One request lists 1,000 packages. About 800 are reports. |
| `crs` | 100 | 200 | CRS Reports, active or archived, through api.congress.gov `crsreport`. PDFs come from www.congress.gov/crs_external_products, with a 2 s crawl delay. | US government work. Copyright status unverified, and reports may contain third-party material. | **Works with `DEMO_KEY`.** About 160 Reports per 250-item request, with up to 3 requests. |
| `fed` | 60 | 220 | FEDS and IFDP working papers (2022-2025 index pages) and FOMC minutes. | Public domain unless otherwise indicated. | **Works.** No robots.txt. About 550 papers and 46 minutes are listed. |
| `arxiv` | 150 | 250 | OAI-PMH `arXiv` format, sets q-fin, econ, stat and cs (2025). Only records whose licence is CC BY 4.0 or CC0 are kept. | CC-BY-4.0 / CC0-1.0. Authors are kept for attribution. | **Works.** About 35% of q-fin records qualify. OAI requests are spaced 3 s apart, and PDF requests 15 s apart (robots crawl delay). |
| `uscourts` | 100 | 150 | Federal appellate opinions from one govinfo `search` request (1,000 results), interleaved across circuits. PDFs come from www.govinfo.gov/content. | US government work, verify per document. | **Works with `DEMO_KEY`.** About 13 circuits. |
| `rfc` | 40 | 60 | RFCs 8650-9850 (the v3 format has a PDF rendering), in an order shuffled with seed 0. | IETF Trust Legal Provisions: unmodified copies only. | **Works.** |
| `cuad` | 100 | 150 | Contract PDFs from CUAD v1: 510 commercial contracts originally filed on EDGAR as exhibits. A sample is taken in an order shuffled with seed 0. Replaces EDGAR exhibit-10. | CC BY 4.0; attribution below and in every record. | **Works.** The archive (106 MB, Zenodo record 4595826) is downloaded once. |
| `edgar_ars` (optional) | 300 | 450 | Form ARS glossy annual reports, PDF main documents only, through EDGAR full-text search. | SEC public filing, not redistributed. | **Unavailable.** www.sec.gov answers 403 to the project User-Agent, which has no contact email by ruling. |
| `edgar_ex10` (optional) | 50 | 75 | Exhibit-10 material contracts filed as PDF, through EDGAR full-text search. | SEC public filing, not redistributed. | **Unavailable**, for the same reason. |

**CUAD attribution.** CUAD v1, The Atticus Project. Hendrycks, Burns, Chen and Ball, "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review", NeurIPS 2021. Licensed CC BY 4.0, https://www.atticusprojectai.org/cuad. Only the contract PDFs are used. CUAD's clause annotations (`master_clauses.csv`, the JSON and the label spreadsheets) are never read, so no CUAD label enters training, calibration or the gold set.

## Rules the fetcher follows

- **User-Agent.** `docrooter-research/0.5 (research; contact via repository)`, with no personal data.
- **Rate limits.** Per host or bucket:
  - SEC: at most 5 requests/s, shared by all SEC hosts;
  - arXiv: 1 request per 3 s across hosts, and PDFs at most 1 per 15 s, the arxiv.org crawl delay;
  - api.data.gov `DEMO_KEY`: 1 request per 6 minutes, shared by govinfo and congress.gov. That is 10/hour, as its response headers report;
  - everything else: 1 request/s, or the robots.txt crawl delay if longer (zenodo.org: 10 s).
- **robots.txt.** robots.txt is obeyed, with `*`/`$` patterns and the longest match winning. Documented API hosts are exempt.
  - An explicit disallow drops a document for good.
  - A robots.txt that cannot be read (401/403/429, a server or network error) stops that host for this run only. It is transient: its documents are retried on the next run.
- **Redirects.** Every hop is vetted like a first request: http(s) only, no https-to-http downgrade, the target host's block list, robots.txt and rate limit.
- **No aggressive retries.** A 401/403/429 stops that host for the run. Network errors are retried only by the next run.
- **API keys.** Keys come from the process environment only: `GOVINFO_API_KEY`, `CONGRESS_API_KEY` or `DATA_GOV_API_KEY`. `.env` is never read. Without a key, `DEMO_KEY` is used for the three listing requests. It is documented by api.data.gov for trying its APIs, and one request lists hundreds of documents.
- **Untrusted downloads.**
  - Each PDF is written to `<id>.pdf.part`, then renamed into place.
  - A PDF must start with `%PDF-` and be at most 50 MB.
  - It is parsed only by `pdf_meta.py` under `python -I`, with a 180 s timeout and a 2 GB memory cap: `setrlimit` where the OS enforces it (Linux), plus a resident-size watchdog in the parent (macOS).
  - Nothing downloaded is executed. The CUAD archive is never extracted as a tree; single members are read with the same checks.
  - Scans are dropped: a PDF needs at least 200 text characters in its first five pages.
  - Byte-identical duplicates are kept once.
- **Exclusion (no dev or held-out document).** Only `benchmarks/corpus/manifest.jsonl` and `benchmarks/heldout/manifest.jsonl` are read, and only their `sha256`, `url` and `title` fields. Nothing else under `benchmarks/heldout/` is opened. A document is dropped when any of these match:
  - its normalised URL;
  - its normalised title;
  - its sha256, so the same bytes re-hosted elsewhere are dropped too.

  Every record is checked again when the manifest is written. Spec §5 matches on (publisher, title). The publisher is not read, by ruling, so titles match without it, which can only drop more. Held-out drops are only counted: no id, URL, title or matching field is logged or stored (`drops.jsonl` keeps a hash of the id, so the document is not retried).

## Counts

The fetcher rewrites this block at the end of every non-dry run.

<!-- counts:begin -->

Last fetch: target 2, 4 documents in the manifest (train 4, calib 0, gold 0).

| Source | Scaled target | Documents | Made up | Shortfall | Status |
| - | - | - | - | - | - |
| cuad | 2 | 2 | 0 | 0 | ok |
| **total** | 2 | 2 | | | |

Realised splits per source (hash-assigned, so stratified in expectation):

| Source | train | calib | gold | gold % |
| - | - | - | - | - |
| ir_annual_reports | 2 | 0 | 0 | 0.0 |
| cuad | 2 | 0 | 0 | 0.0 |

Drops this run: none.

<!-- counts:end -->
