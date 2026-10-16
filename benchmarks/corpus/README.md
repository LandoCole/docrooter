# docrooter benchmark corpus

A small labelled corpus of real PDFs, large and small, used to benchmark docrooter's table-of-contents extraction and document-type classification. Every document is a genuine public file from its original publisher, and its gold label was fixed when the document was chosen (an arXiv paper is a `research_paper`, a Form 10-Q is a `quarterly_report`), then confirmed by reading the first page of the downloaded PDF.

## Files

| File | What it is |
|---|---|
| `doc_types.json` | The label taxonomy (docrooter nested format: `{"nodes": [...]}`), with a definition for every node. |
| `manifest.jsonl` | One JSON object per document: `id`, `file`, `url`, `title`, `publisher`, `label`, `pages`, `bytes`, `sha256`, `label_rationale`. |
| `fetch.py` | Stdlib-only downloader that fetches every manifest entry into `pdfs/` and checks its sha256. |
| `pdfs/` | The downloaded PDFs (git-ignored; created by `fetch.py`). |
| `section_gold.jsonl` | Silver section labels for section normalisation (see below): a `{"_meta": ...}` header line, then one `{doc_id, section_title, page, level, canonical, note?}` object per level-1/2 section of six documents. |

## Reproducing

```
python benchmarks/corpus/fetch.py
```

It downloads each file with a polite 1.5 s delay, skips files already present with the right sha256, and exits non-zero on a failed download or sha256 mismatch (`--lenient` downgrades mismatches to warnings; `--dest DIR` changes the output directory; `--only doc_001 ...` fetches a subset). A mismatch means the publisher replaced the file at that URL since the manifest was written. Several URLs point at "living" documents (for example the GNU manuals, the AWS and Raspberry Pi documents), so expect occasional drift over time; the label stays correct, but page counts and hashes may change.

## Label definitions

Leaf ids are `<parent>.<leaf>`. Parents: `financial`, `legal`, `research`, `technical`, `government`, `corporate`.

| id | label | definition |
|---|---|---|
| `financial` | **Financial** | Documents issued by or about companies and funds that report results or offer securities. |
| `financial.annual_report` | Annual report | A yearly report to shareholders covering a whole fiscal year, either a Form 10-K filing or a glossy annual report, with business overview, management discussion and audited financial statements. Not a 10-Q (partial year) or a standalone shareholder letter (a few pages of narrative only). |
| `financial.quarterly_report` | Quarterly report | An interim report covering a single quarter or half-year, such as a Form 10-Q or a quarterly or half-year report, with unaudited or reviewed financial statements and short management discussion. Not a press-release style earnings_release and not a full-year annual_report. |
| `financial.shareholder_letter` | Shareholder letter | A letter from a company's management or an investment manager addressed to shareholders or investors that reviews performance and outlook in narrative prose, typically signed by the CEO, chairman or fund manager. Not a full annual_report with financial statements. |
| `financial.earnings_release` | Earnings release | A short press release announcing quarterly or annual financial results, with headline figures, a business summary and condensed financial tables, usually 3 to 15 pages. Not a full quarterly_report or an investor_presentation slide deck. |
| `financial.prospectus` | Prospectus | A formal securities-offering or fund document describing an investment's objectives, risks, fees, and terms so investors can decide whether to buy, such as a mutual fund, ETF or IPO prospectus. Not an annual_report and not a marketing white_paper. |
| `financial.proxy_statement` | Proxy statement | A document sent to shareholders ahead of a shareholder meeting that describes the items to be voted on, the board nominees, executive compensation and governance, such as a Schedule 14A or DEF 14A. Not an annual_report even when it is bundled with one. |
| `legal` | **Legal** | Documents that create, state or interpret legal rights and obligations. |
| `legal.contract` | Contract | A legally binding agreement between named parties, with recitals, defined terms, numbered clauses and signature blocks, such as a services, license, lease, employment or supply agreement. Not a public terms_policy that is posted unilaterally for all users. |
| `legal.terms_policy` | Terms and policies | A unilaterally published set of terms of service, terms of use, privacy policy, cookie policy or acceptable-use policy that governs the public's use of a product or service. Not a negotiated contract between named parties. |
| `legal.court_opinion` | Court opinion | A judicial decision issued by a court, including the syllabus, the majority opinion and any concurring or dissenting opinions, with case caption, docket number, holdings and reasoning. Not legislation and not a legal brief filed by a party. |
| `legal.legislation` | Legislation | The text of enacted or proposed law, such as a statute, bill, act, treaty or administrative regulation, organized in titles, sections and subsections with enacting clauses. Not a court_opinion interpreting the law and not a government_report about it. |
| `research` | **Research** | Documents that report original research or formally specify technology. |
| `research.research_paper` | Research paper | A scholarly paper reporting original research, with an abstract, introduction, methods, results and references, typically from arXiv, journals or conferences. Not an industry white_paper or a technical_standard. |
| `research.technical_standard` | Technical standard | A normative technical standard or specification, such as an RFC, IETF, W3C, ISO or IEEE document, that uses requirement keywords (MUST, SHOULD) to define a protocol, format or interface precisely. Not a vendor datasheet or a user_manual. |
| `research.white_paper` | White paper | An industry, vendor, consultancy or think-tank report that argues for a position, explains a topic or presents analysis aimed at a professional audience, without being peer-reviewed or normative. Not a research_paper (academic) or a government_report (issued by an agency). |
| `technical` | **Technical** | Product documentation describing how to use or select a hardware or software product. |
| `technical.user_manual` | User manual | Task-oriented documentation that guides an end user or administrator through installing, configuring, operating and troubleshooting a product, with chapters, procedures and a table of contents. Not a datasheet, which lists specifications rather than procedures. |
| `technical.datasheet` | Datasheet | A concise specification document for a hardware component or product, with features, pin or port descriptions, electrical or performance characteristics, ordering information and package details, normally from the manufacturer. Not a procedural user_manual. |
| `government` | **Government** | Documents published by public institutions in their official capacity. |
| `government.government_report` | Government report | A report published by a government agency, central bank, legislative office or audit office, presenting analysis, findings, monetary policy, oversight results or recommendations to policymakers and the public. Not legislation itself and not a think-tank white_paper. |
| `corporate` | **Corporate** | Documents a company publishes to communicate non-statutory information about itself. |
| `corporate.sustainability_report` | Sustainability report | A company report on environmental, social and governance (ESG), sustainability, climate or impact performance, with metrics, targets and frameworks such as GRI, SASB or TCFD. Not a financial annual_report, although the two are sometimes combined. |
| `corporate.investor_presentation` | Investor presentation | A slide deck prepared for investors and analysts, such as an earnings call presentation or investor day deck, laid out as landscape slides with short bullets, charts and forward-looking-statement disclaimers. Not a prose earnings_release. |

## Counts per label

| label | documents |
|---|---|
| **financial** | 14 |
| `financial.annual_report` | 3 |
| `financial.quarterly_report` | 2 |
| `financial.shareholder_letter` | 3 |
| `financial.earnings_release` | 2 |
| `financial.prospectus` | 2 |
| `financial.proxy_statement` | 2 |
| **legal** | 12 |
| `legal.contract` | 2 |
| `legal.terms_policy` | 3 |
| `legal.court_opinion` | 4 |
| `legal.legislation` | 3 |
| **research** | 11 |
| `research.research_paper` | 4 |
| `research.technical_standard` | 4 |
| `research.white_paper` | 3 |
| **technical** | 7 |
| `technical.user_manual` | 3 |
| `technical.datasheet` | 4 |
| **government** | 3 |
| `government.government_report` | 3 |
| **corporate** | 4 |
| `corporate.sustainability_report` | 2 |
| `corporate.investor_presentation` | 2 |

Total: 51 documents, every leaf type has at least 2.

## Size distribution

| pages | documents |
|---|---|
| <10 | 8 |
| 10-49 | 20 |
| 50-100 | 8 |
| >100 | 15 |

Page counts range from 1 to 642 (median 39). Total download size is about 125 MB; the largest file is 14.8 MB.

## Sources and licensing

Documents come from public, no-login publisher sites: arXiv, the RFC Editor, Berkshire Hathaway, company investor-relations sites, supremecourt.gov, govinfo.gov, federalreserve.gov, 9-11commission.gov, the GNU Project, Raspberry Pi, Texas Instruments, Vishay, Diodes Inc., NIST and others (see `publisher` and `url` in the manifest). SEC EDGAR is deliberately not used. The documents remain the property of their publishers and carry their own terms: this repository does not redistribute them, it only records where to find them, a checksum, and our own labels and rationale. `pdfs/` is git-ignored. Use the corpus for internal evaluation only and respect each publisher's terms when downloading.

## Labelling notes

- Labels describe the document's genre, not its subject or publisher. Several publishers appear under more than one label (Amazon, Berkshire Hathaway, NVIDIA, Tesla).
- Boundary cases kept deliberately: `doc_037` (the Bitcoin paper) is labelled `research.white_paper` because it is the canonical self-published "white paper", not a peer-reviewed paper; the Amazon annual report is a glossy report bundling the CEO letter with the Form 10-K, labelled `financial.annual_report`; the Apple privacy policy is a single very long PDF page.
- Court opinions include both slip opinions and Official Reports preliminary prints; both are `legal.court_opinion`.
- Debt issuance programme base prospectuses are used for `financial.prospectus` because they are dated, stable securities-offering documents (fund prospectuses are overwritten in place by their sponsors).

## Section labels (silver)

`section_gold.jsonl` holds **agent-assigned silver labels, not human gold**. An agent (Claude Opus 5.5) loaded `doc_027`, `doc_028`, `doc_029` (research papers), `doc_001`, `doc_002` (annual reports) and `doc_016` (contract) with the default engine, listed every level-1 and level-2 section, read each heading with the first ~300 characters of its body, and chose the best node of the doc type's section taxonomy (`DOC_TYPE_TO_SECTION_TAXONOMY`) from the node definitions. The labels were written and committed before any model was run on them. The header line records the labelling time and the src commit the documents were loaded at.

- `canonical: null` marks parser junk or text that is not a section (title and author blocks, printed-TOC lines, headings repeated on continuation pages, run-in paragraph leads, form fields). These rows are counted but excluded from accuracy.
- `note` justifies ambiguous choices and names the runner-up node. Annual-report cover pages and tables of contents are `other`, as that node's definition says; research papers and contracts have no such node, so their title blocks are junk.
- Sections are matched to a freshly loaded document by (casefolded title, page). If the loader's output changes, unmatched rows are reported by `benchmarks/normalize_bench.py`, not silently dropped.
