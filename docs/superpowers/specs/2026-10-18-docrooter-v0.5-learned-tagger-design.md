# docrooter v0.5: training corpus, teacher, and the learned element tagger

- **Date:** 2026-10-18
- **Status:** Approved under the user's standing instruction of 2026-10-18: "keep looping until the more capable docrooter is complete … compile a training dataset large enough to finetune the BERT model".
- **Program:** `docs/superpowers/program/2026-10-18-docrooter-element-intelligence-program.md`, phase v0.5.
- **Research:**
  - `docs/superpowers/research/2026-10-18-element-classification-research.md` §5, §7, §8;
  - `docs/superpowers/research/2026-10-18-v0.5-teacher-corpus-research.md`.
- **Builds on:** v0.1–v0.4, especially the v0.4 elements, rooter, and `train()`.

## 1. Intent

v0.4 tells us what each element **is**: a paragraph, table or figure. v0.5 adds what it is **about** and what it **does**.

It learns those tags from a large unlabelled corpus, following the TELEClass recipe:
- a local, permissively licensed LLM teacher labels a sample, choosing from a shortlist;
- the corpus enriches the taxonomy;
- an embedding-gap confidence keeps the reliable labels;
- a BERT-class student is fine-tuned on them, calibrated, and runs locally at scale.

The user's application can then read only the relevant elements of a filing, and gate or route by tag.

## 2. Goals

1. **Element tag taxonomies** with a definition for every tag. There are two facets:
   - **topic**: what the element is about. It is hierarchical, single-label, per document type, and reuses the v0.2 section taxonomies.
   - **function**: what the element does. It is flat, multi-label, and works across domains.
2. **A `financial.fact_sheet` document type** with a definition. Fund fact sheets are a gap the user hit.
3. **A corpus builder** in the package (`docrooter corpus build`) and a training corpus for us: at least 1,000 public native-PDF documents, finance-heavy, with no overlap with the dev or held-out corpora.
4. **A teacher engine protocol**, with a default local LLM (Qwen3-4B-Instruct-2507, 4-bit, MLX), multiple-choice log-probability scoring, a cache and provenance. Jev is never a teacher (program §5).
5. **The TELEClass loop:**
   - candidate shortlist;
   - teacher labels plus structural weak labels;
   - corpus term mining;
   - similarity-gap confidence;
   - confident subset;
   - student training (frozen-body heads first, then a fine-tuned body);
   - one active-learning round;
   - calibration.
6. **The student in the rooter:** an `element_tags` task, `Element.tags`, `docrooter tag`, and the playground.
7. **Measurement** on an independently annotated gold set, never teacher labels:
   - student against teacher, zero-shot and keyword;
   - learning curve over corpus size;
   - throughput and memory;
   - document type with element evidence on dev.

## 3. Non-goals and anti-goals

- **Non-goals:**
  - selection and the completeness dial (v0.6);
  - generating synthetic training documents (the user has large corpora);
  - publishing model weights (that needs the user's go-ahead);
  - non-English corpora.
- **Anti-goals:**
  - No Jev output may ever enter training, calibration or teacher data.
  - No held-out document (`benchmarks/heldout/`) may enter the corpus, and none is opened in v0.5.
  - No dev-corpus document may enter the training corpus: the dev doc-type benchmark must stay a fair test.
  - No PDF is committed. The corpus is a manifest of URLs and checksums plus fetch scripts, as for dev and held-out.
  - Nothing exceeds the 8 GB memory budget. Only one model process runs at a time.

## 4. Taxonomies

### 4.0 Rulings after the builder check (binding)

1. **One merged topic tree.** `element_topics.json` is: root → doc family → section leaf. The families are `annual_report`, `fund_document`, `contract`, `research_paper`, `technical_document` and `generic`; each family's leaves are its section taxonomy plus `other`.
   - **One bundle:** one shared body, one `TrainedModel`, one training run.
   - **Calibration:** per facet.
   - **"Predicted only via its parent"** means the family level.
   - **Hierarchical F1** is computed over family → leaf.
2. **The student scores every element over the whole tree (Beam(3)).**
   - `Element.tags["topic"]` reports the best leaf within the document's family, as given by its doc type through the existing map.
   - `TagResult.family_probs` keeps the unconstrained family distribution.
   - The doc-type evidence in §11 uses `family_probs`, so it is not circular.
3. **Fund documents.**
   - `financial.prospectus` and `financial.fact_sheet` map to `fund_document`.
   - Fund shareholder reports (N-CSR and similar) are left out of the v0.5 corpus.
   - Government, court and RFC documents keep the structural `generic` family. Their topic is expected to be mostly structural, and their function facet carries most of the value; the docs say so.
4. **Gold-set scoring has two modes.**
   - **Primary:** the family comes from the gold document's known doc type. This isolates the tagger, and the targets apply to it.
   - **Secondary:** end to end, using the rooter's predicted doc type. Both are reported.
5. **Teacher wall-clock.**
   - **Ceiling:** 12 h in total for the first pass plus active learning, measured.
   - **Prompt:** the cached prefix per family holds the instruction and ALL of that family's lettered leaves with their definitions. Each element's suffix lists its 5 shortlisted letters, and the answer is restricted to those letters.
   - **Topic sample:** 12,000 elements if it fits the ceiling, else as many as fit, but at least 6,000.
   - **Functions:** asked only on a 50% subsample, for the top 3 functions by prescreen.
   - The "2 items/s" rule is replaced by the ceiling.
6. **Unasked functions.** For each function tag, elements where it was not asked are excluded from that tag's training data. Elements whose prescreen score for it is < 0.2 count as implicit negatives.
7. **Filter, TELEClass-faithful.**
   - **Order:** teacher labels, then term mining (on teacher-labelled train elements), then centroids (teacher-labelled elements containing a class keyword), then the gap, then the filter.
   - **Core classes:** an element's core classes are those ranked above its largest cosine gap.
   - **Keep rule:** keep the element if the teacher's choice is one of its core classes, or the teacher's probability is ≥ 0.8.
   - **Scope:** this filter applies to the topic facet only.

### 4.1 Topic facet: `element_topics/<doc_family>`

The topic facet reuses the section taxonomies (`section_types/annual_report`, `contract`, `research_paper`, `technical_document`, `generic`) unchanged, and adds an `other` leaf to each.

The doc-type → section-taxonomy map stays as it is. `financial.fact_sheet` maps to a new `section_types/fund_document`, which covers prospectuses, fact sheets and shareholder reports:
- investment objective;
- fees and expenses;
- principal investment strategies;
- principal risks;
- performance;
- holdings and allocation;
- management;
- purchase and sale;
- tax information;
- other.

Each has a definition and heading synonyms, as the v0.2 taxonomies do.

An element's topic is chosen within the taxonomy of its document's type, as already used by `normalize_sections`.

### 4.2 Function facet: `element_functions.json`

Flat and multi-label. An element may carry zero or more functions. Each tag needs a definition, 3–6 positive examples written for the taxonomy (not taken from any eval set), and a "not this" clause.

| Tag | Meaning |
| - | - |
| `forward_looking` | Statements about expected future results, plans or outlook |
| `risk_disclosure` | Describes a risk, uncertainty or adverse scenario |
| `financial_data` | Reports quantitative financial figures: statement rows, KPIs, fees |
| `accounting_policy` | States how amounts are measured or recognised |
| `legal_boilerplate` | Standard legal text: disclaimers, safe-harbour, notices, copyright |
| `definition` | Defines a term |
| `obligation` | Imposes a duty or prohibition on a party: shall, must, may not |
| `procedure` | Step-by-step instructions or a process |
| `contact_or_admin` | Addresses, phone numbers, filing metadata, form headers |
| `signature` | Signature blocks, attestations |
| `citation` | Bibliography or reference entries, footnote citations |
| `marketing` | Promotional claims about products or performance |

### 4.3 Document types

Add `financial.fact_sheet` to `doc_types.json`, with a definition in the existing style:

> A short (typically 1–4 page) periodic summary of a single fund or share class: objective, key facts, fees, performance table and chart, top holdings and allocation, as of a date. Not a full prospectus and not an annual report.

The v0.3 held-out doc-type numbers predate this change. The docs say so. The held-out corpus is not re-run until the program's final evaluation.

## 5. The training corpus (`benchmarks/training_corpus/`)

- **Sources and target counts.** Native PDFs only. Fetched politely, with a User-Agent naming the project and a contact, and within each source's published rate limits (EDGAR at most 5 requests/s).

  | Source | Target count |
  | - | - |
  | EDGAR Form ARS annual reports (PDF main documents only) | 300 |
  | Fund documents with PDF availability (summary prospectuses, annual and semi-annual shareholder reports, fact sheets from public fund-sponsor URLs), never redistributed | 150 |
  | govinfo GAO reports | 150 |
  | CRS reports | 100 |
  | Federal Reserve publications | 60 |
  | arXiv papers with CC-BY or CC0 licences only (licence read from metadata) | 150 |
  | govinfo USCOURTS opinions | 100 |
  | RFCs | 40 |
  | Contracts: EDGAR exhibit-10 material contracts filed as PDF | 50 |

  Total: at least 1,100. Shortfalls in one source are made up from the others and recorded.
- **Manifest** (`manifest.jsonl`): `id`, `url`, `source`, `licence/status`, `sha256`, `bytes`, `pages`, `fetched_at`, `doc_type_hint` (the source's own type, e.g. Form ARS → annual report), plus a `split` field:
  - `train`: about 85%;
  - `calib`: about 5%, used for calibration;
  - `gold`: about 10%, held aside for the gold set and never used for training or tuning.

  Splits are assigned by document, stratified by source, with seed 0.
- **Exclusion:** drop any document whose sha256 or normalised URL matches the dev or held-out manifests, or whose (publisher, normalised title) matches one. Record the drops.
- **Element extraction:** `process(profile="accurate")`, with heron (ruling E11a). Read at most the first 10 pages plus up to 20 more pages sampled with seed 0, and record which pages were read. Skip furniture and `contents` elements. Store the elements of every document in a corpus store (§6).
- **Budget:** about 1,100 docs × at most 30 pages × about 0.6 s/page with heron is about 5.5 h on the M1. It runs in the background under the one-model-process guard, is resumable, and is checkpointed per document.

## 6. Corpus store (`docrooter.corpus`)

- **Layout:**
  - `<dir>/manifest.jsonl`;
  - `<dir>/docs/<id>.json`: the document id, doc-type hint, pages read, and the elements with `id`, `kind`, `text`, `heading_path`, `section_title` and `page`;
  - `<dir>/labels/<labeler>.jsonl`: one row per element, keyed by `doc_id` + `element_id` + `text_sha1`, holding the label distribution and the provenance (engine, model, revision, prompt hash, taxonomy hash, timestamp);
  - `<dir>/stats.json`.
- **API:** `Corpus(dir)` with `add(path_or_url_record)`, `iter_elements(split=…, kinds=…)`, `labels(labeler)` and `write_labels(labeler, rows)`.
- **Properties:** JSON lines, with no new dependency. Idempotent: re-adding a document with the same sha256 is a no-op.
- **CLI:** `docrooter corpus build DIR --from-manifest manifest.jsonl [--profile accurate] [--max-pages 30]` and `docrooter corpus stats DIR`.

## 7. Teacher (`docrooter.teach`)

### 7.1 Protocol

```python
class TeacherEngine(Protocol):
    name: str          # e.g. "mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d4277"
    def choose(self, items: Sequence[TeachItem], *, instruction: str) -> list[dict[str, float]]: ...
    def close(self) -> None: ...

class TeachItem(BaseModel):
    text: str                      # element text, truncated to 384 tokens of the teacher tokenizer
    context: str                   # "Document type: …\nSection: A > B\nElement: paragraph"
    options: list[tuple[str, str]] # (tag id, "label: definition"), at most 8, lettered A..H in order
```

The return value is a probability per tag id: a softmax over the option letters' next-token log-probabilities.

### 7.2 Engines

- **`mlx`:** the default on macOS arm64. Extra `teach-mlx = ["mlx-lm>=0.32; sys_platform=='darwin' and platform_machine=='arm64' and python_version>='3.11'"]`.
  - Model `mlx-community/Qwen3-4B-Instruct-2507-4bit` at revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
  - Fallback model `mlx-community/Qwen3-8B-4bit` at revision `545dc4251c05440727734bcd94334791f6ab0192`, with thinking disabled.
  - One forward pass per item. The shared prefix (instruction plus option list) is cached, and the cache is trimmed back after each item.
- **`transformers`:** any platform. Same prompt and scoring, using the model's chat template, on MPS, CUDA or CPU.
- **`openai`:** any OpenAI-compatible endpoint that returns `top_logprobs`. Letters missing from the top-k get the minimum seen probability minus 1 nat. The user configures it and is responsible for the provider's terms.
- **`jev`:** refused with a clear error that cites the TypeSafe MCA §2.3(b).

### 7.3 Labelling procedure

For each topic facet, per element:
1. **Shortlist.** Zero-shot UltraBERT-Embed scores the element (as `heading path + text`) against the leaves of the document's topic taxonomy. Take the top 5, plus `other` if it is absent.
2. **Teacher choice.** The teacher chooses among the shortlist (TELEClass step 1).

For the function facet, per element, the teacher answers one yes/no question per function on the 4 most likely functions. Their order comes from a zero-shot embedding prescreen against the function definitions; the top 4 always includes a tag the prescreen puts above 0.5. P(yes) is the softmax over the "Yes" and "No" tokens.

Weak labels are recorded alongside, never mixed in silently:
- the section's normalised label from tier 1, when its confidence ≥ 0.5;
- for EDGAR 10-K-style headings ("Item 1A"), the item mapped to the topic.

### 7.4 Budget and sampling

The teacher labels the stratified `train` + `calib` sample:
- at most 60 elements per document (furniture excluded), stratified by kind;
- at least 12,000 elements, and at most 24,000 for the first pass.

Measured throughput is recorded. If it is below 2 items/s, cap the pass at 12,000 and record that.

Labels are cached: a re-run never re-scores an element with the same text sha1, taxonomy hash and teacher revision.

## 8. Enrichment, confidence and the training set (TELEClass steps 2–3)

- **Term mining (§3.2 of TELEClass, adapted):**
  - **Candidates:** n-grams of 1–3 tokens with document frequency ≥ 5 across the train split, without stop-word edges.
  - **Score:** for each tag and its siblings, `aff = pop^0.5 · dist^0.5 · max(0, cos(term, tag label))`. Here `pop = log(1 + df within the tag's elements)`, `dist = exp(BM25 within tag) / (1 + Σ exp(BM25 within each sibling))`, and the cosine uses UltraBERT-Embed.
  - **Output:** the top 10 terms per tag go to `<taxonomy>.enriched.json` as `keywords` (a new optional node field; old files stay valid).
  - The shortlist (§7.3) uses the enriched label text in a second labelling pass only if the gold-free calib check (§8, last bullet) improves.
- **Similarity-gap confidence:**
  - **Class centroids:** each class's centroid is the mean UltraBERT-Embed embedding of its teacher-labelled train elements whose text contains at least one of the class keywords.
  - **Gap:** each element's gap is the largest drop in its sorted cosine scores against the centroids.
  - **Combined confidence:** `teacher_prob × gap_rank_percentile`.
  - **Agreement:** when the teacher's choice is above the gap cut, they agree. The training set keeps elements whose teacher choice equals the top-gap class, or whose teacher probability is ≥ 0.8.
  - **Trim:** keep the top 75% by combined confidence per class, with at least 20 per class where available. A class with fewer than 20 confident elements is reported and left untrained; it is predicted only via its parent.
- **Calibration split:** the `calib` documents, with teacher labels. This check is free of gold labels.

## 9. Student

- **Stage A, heads only (cheap baseline).** `train(taxonomy, dataset, base_model="dmis-lab/UltraBERT-Embed-150M", finetune_body=False)` on the confident set. The input text is `"{heading path}\n\n[{kind}] {text}"`, truncated to 256 tokens.
- **Stage B, fine-tuned body (the deliverable).**
  - **Run:** the same call with `finetune_body=True`, `max_seq_len=256`, `gradient_checkpointing=True`, batch size from the memory budget, and `max_triplets` 40,000. Device: MPS if available, under the guard.
  - **Hardware fit:** if `train()`'s sentence-transformers path proves unstable on MPS (research §4.2), the task may switch to a plain-`transformers` loop with cross-entropy over the leaves, using the same bundle format. Any switch is recorded.
  - **Time:** budget at most 6 h wall-clock. If one epoch exceeds it, sub-sample to fit and record that.
- **Function heads:** one-vs-rest logistic regression per function on the student's embeddings, trained on the teacher's P(yes) ≥ 0.7 positives and ≤ 0.2 negatives. Saved in the same bundle under `functions`.
- **Active learning, one round:**
  - Score the unlabelled train elements with the stage-B student.
  - Send the 3,000 with the smallest top-2 margin, stratified by class, to the teacher.
  - Add those labels, retrain stage B once, and report before and after.
- **Calibration:** temperature per facet on `calib`, via `calibrate()`, stored in the bundle.
- **Bundle:** saved to `benchmarks/.cache/bundles/element-tagger-v1/`, never committed and never published. Docs show users how to train their own.

## 10. In the package

- **`Element.tags`:** `Element.tags: dict[str, TagResult] | None = None`, keyed by facet (`"topic"`, `"function"`). It is additive and omitted when None, so v0.4 JSON is unchanged.
  - `TagResult(labels: list[str], probs: dict[str, float], confidence: float, source: str)`.
- **Rooter task:** `element_tags`, set to `none` (the default in every profile), a trained bundle path, `embedding:<spec>` (zero-shot) or `teacher:<engine spec>`. The rooter applies it to paragraph, list_item, table, caption, footnote and formula elements, in batches, with one model load per `process_many`.
- **Documents:** `docrooter tag FILE --bundle PATH [--profile balanced] [--json]`.
- **Corpora:** `docrooter tag-corpus DIR --bundle PATH` writes a labels file into the corpus store.
- **Training:** `docrooter train-elements DIR --teacher mlx … --out BUNDLE` runs §7–§9 end to end on the user's corpus. Each stage is resumable.
- **Playground:** an element list with tag chips and probabilities, when the server is started with `--element-tags BUNDLE`.

## 11. Gold set and measurement

- **Gold set (independent of the teacher):**
  - 900 elements from the `gold` split documents, stratified by doc family, element kind and the teacher-free zero-shot topic: 300 annual reports, 150 fund documents, 150 research papers, 150 government/legal, 150 technical/RFC.
  - **Annotation:** by Claude subagents working from the definitions only, without seeing any model output. 20% (180) are double-annotated by a second, independent subagent, then adjudicated by a third.
  - Report Cohen's κ for the topic facet and per-function κ.
  - Labels are written to `benchmarks/training_corpus/gold.jsonl` and committed **before** any model scores the gold documents.
  - **Disclosure:** "Claude-annotated gold", like the v0.2 silver section labels.
- **Systems** on gold:
  - keyword;
  - zero-shot UltraBERT-Embed;
  - zero-shot with the enriched taxonomy;
  - the teacher (Qwen3-4B), and the 8B fallback on a 300-element subset;
  - stage-A student;
  - stage-B student, before and after the active-learning round.
- **Metrics:**
  - **topic:** accuracy and macro P/R/F1 per doc family, and hierarchical F1;
  - **function:** micro and macro F1, and per-tag F1;
  - **calibration:** ECE before and after temperature.
- **Learning curve:** stage B (or stage A, if B is too slow to repeat) trained on 1/8, 1/4, 1/2 and all of the confident set. Report gold macro F1 at each size. This is the evidence for "performance depends on corpus size".
- **Throughput and memory:**
  - student elements/s on M1 (MPS and CPU);
  - teacher items/s;
  - corpus build seconds/page;
  - peak memory for each stage.
- **Document type with element evidence (dev corpus):**
  - Score each dev document with `doc_type` from the view as today, and separately with an element-evidence mix: `0.7 · view distribution + 0.3 ·` the doc-type distribution implied by its elements' topic tags. The tag → doc-family map comes from the taxonomy structure.
  - The 0.7/0.3 weight is fixed here and never tuned on dev.
  - Report against v0.3/v0.4 dev numbers. The held-out corpus is not run in v0.5.
- **Program success targets** (program §6.3):
  - the student's gold macro F1 within 5 points of the teacher's;
  - at least 10 points above zero-shot UltraBERT;
  - at least 50 elements/s on M1;
  - within 8 GB.

  Report as is.

## 12. Packaging

- **Extras:**
  - `teach-mlx` as in §7.2;
  - `teach = ["transformers>=5", "torch>=2.4", "accelerate>=1.0"]`, or reuse the `models` extra if it already has these; record the choice.
- **Core dependencies:** none.
- **mypy:** ignore_missing_imports for `mlx`, `mlx_lm`.
- **Tests:** unit tests use fake teachers and fake embedders. Model runs are integration tests behind markers `teach` and `models`.

## 13. Left to the builder

- The exact prompt wording. Record it, hash it into the provenance, and keep it in `src/docrooter/teach/prompts.py`.
- Fetch-script details per source, within each source's rules.
- How to stratify the teacher sample, within §7.4.
- Batch sizes. Use the largest that fits the budget, measured.
