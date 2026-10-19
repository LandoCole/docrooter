# docrooter v0.6: selection and the completeness dial

- **Date:** 2026-10-19
- **Status:** Approved under the user's standing instruction of 2026-10-18 ("keep looping until the more capable docrooter is complete"). The dial was the user's own idea: "a setting so user can set 1-0 to adjust the range or margin of error … balance precision with recall/completeness, e.g. efficiency with context".
- **Program:** `docs/superpowers/program/2026-10-18-docrooter-element-intelligence-program.md`, phase v0.6.
- **Research:** `docs/superpowers/research/2026-10-18-element-classification-research.md` §6 (conformal risk control) and §9 (chunking).
- **Builds on:** v0.4 elements and chunks, and the v0.5 element tagger with calibrated tags.

## 0. Rulings after the builder check (binding)

1. **Data separation.** The v0.5 calib split (62 documents) is partitioned by a document hash into **calib-fit** (about half) and **calib-crc** (the rest).
   - calib-fit trains the v0.5 tagger temperatures and the v0.6 relevance combiner.
   - calib-crc is used only for CRC.
   - This amends v0.5 §9: calibration uses calib-fit only. v0.5 had not run calibration when this was decided.
2. **Full labels where recall is measured.**
   - The local teacher labels the topic of EVERY content element of the calib documents and of the gold-split documents. Gold labels are already committed, and teacher labels on gold documents are used only for evaluation, never for training.
   - This adds roughly 6 h of teacher time. It runs after the v0.5 pipeline, under the 12 h-per-pass ceiling, and the measured time is reported.
   - Relevance for CRC and for the main evaluation is "the teacher topic is one of the field's topics" (teacher relevance).
   - The 900 gold elements give a second, independent check: recall over gold-annotated elements, with relevance taken from the gold topic. It is reported beside the main evaluation, together with the measured teacher–gold topic disagreement, so the guarantee is never claimed against labels it was not calibrated on.
3. **CRC unit.**
   - The unit is a (document, field) pair with at least one relevant element, pooled within each family. n is reported per family.
   - The finite-sample term 1/(n+1) is shown. Where it makes a target c unreachable, `select()` returns the whole document with `guarantee=True` (recall is then 1 trivially) and `capped=True`, and the readout says "dial capped by calibration size".
   - Presets are kept.
   - Pairs from the same document are not exchangeable in a strict sense; the docs disclose this.
4. **Free-text queries.** `guarantee=False` always. The readout shows the model's expected recall, labelled as an estimate. FinanceBench and TAT-DQA results are descriptive only, and no query-side CRC is built.
5. **Evaluation bar.** The program target is mean teacher-relevance recall ≥ c on gold-split documents, at c = 0.9, reporting the 10th percentile and the token share. The gold-annotated subset recall is secondary and is always shown.
6. **Doc-type mismatch.**
   - The field's family is authoritative for `p_topic`. If the analysis's doc type maps to another family, `select()` re-tags the content elements restricted to the field's family.
   - In that case it sets `guarantee=False` and flags `doc_type_mismatch=True`, because CRC was calibrated on correctly typed documents.

## 1. Intent

A downstream LLM should read only the parts of a document it needs. docrooter answers one question: **which elements should be read for need X, at completeness c?**

- c = 1.0 means the whole document; nothing is ever missed.
- Lower values send fewer tokens.
- docrooter states the expected share of relevant content kept. On documents like its calibration set, it guarantees that share on average.

The user's application performs the extraction itself (program §2). docrooter only selects.

## 2. Goals

1. **Needs.**
   - A **field** is a named, query-independent target tied to topic and function tags. The built-in field sets cover annual reports, fund documents and contracts; users can define their own in JSON.
   - A **query** is free text, scored by embedding similarity combined with a tag prior.
2. **Relevance.** A calibrated per-element probability P(relevant | element, need).
3. **The dial.** `completeness ∈ [0, 1]` maps to a threshold through conformal risk control (CRC), with the per-document miss rate as the loss. Selection adds context expansion, and an optional token budget can override the dial.
4. **API and surfaces.** `select()`, `docrooter select`, and a playground dial that shows a live readout of expected recall and tokens.
5. **Evaluation.**
   - Realised recall against the target.
   - Tokens sent against recall (the curve), and the tokens needed for 95% recall.
   - Runs on the v0.5 gold split (fields as topics), and on a public evidence-labelled QA set used for evaluation only.
   - A demo extraction harness, outside the wheel.

## 3. Non-goals and anti-goals

- **Non-goals:**
  - docrooter does not prompt LLMs for extraction;
  - no vector database;
  - no re-ranking with cross-encoders, which is left to the application.
- **Anti-goals:**
  - c = 1.0 must return every content element. It is never approximate.
  - The stated guarantee must hold on held-out documents, or the docs must say plainly that it did not.
  - No evaluation set marked non-commercial is redistributed or used for training or calibration.

## 4. Needs

```python
class Field(BaseModel):
    id: str                       # "annual_report.risk_factors"
    label: str
    description: str
    topics: list[str]             # element_topics leaf ids that carry it
    functions: list[str] = []     # function tags that strengthen it (e.g. "risk_disclosure")
    query: str | None = None      # optional natural-language form, for the embedding term
```

- **Built-in field sets** (`fields/annual_report.json`, `fields/fund_document.json`, `fields/contract.json`): about 8–15 fields each, derived from the topic leaves plus a few cross-cutting ones, e.g. "expense ratio and fees" maps to `fund_document.fees_and_expenses` with `financial_data`.
- **A need** is a `Field`, a list of fields, or a `str` query.

## 5. Relevance model

Features per (element, need):
- `p_topic`: the tagger's probability mass on the need's topics (0 when the element has no tags);
- `p_func`: the maximum probability over the need's functions;
- `sim`: cosine between the element's `heading path + text` embedding and the need text (description, or query);
- `section_hit`: whether the element's section's normalised label is one of the need's topics;
- `kind`: one-hot over the six content kinds;
- `log_tokens`.

The combiner is a logistic regression per need type (field, or query). It is trained on the v0.5 **calib** split:
- **Field relevance:** an element is relevant to a field when its teacher topic label (calib has teacher labels) is one of the field's topics.
- **Query relevance:** queries are synthesised from field descriptions, and relevance follows the field. This is weak but gold-free and needs no LLM.

The weights are saved in the tagger bundle, or in a separate `selector.json` when no tagger is used. If no tagger is present, the combiner falls back to `sim` and `section_hit` only. Docs say so.

## 6. The dial (CRC)

For each calibration document *d* and threshold λ, define the loss

L_d(λ) = 1 − |selected_λ ∩ relevant_d| / |relevant_d|

over the documents with at least one relevant element. L_d is non-increasing in λ, because a lower λ includes more elements.

For target completeness c (α = 1 − c), choose

λ̂(c) = max { λ : (n/(n+1)) · mean_d L_d(λ) + 1/(n+1) ≤ α }

(Angelopoulos et al., conformal risk control). This guarantees E[recall] ≥ c on exchangeable new documents.

- **Storage:** λ̂ is precomputed on a grid of c ∈ {0.5, 0.6, …, 0.95, 0.98, 0.99}, interpolated monotonically, and stored per need type.
- **Calibration set:** the calib split, 62 documents. That is small, so the guarantee's slack (1/(n+1) ≈ 0.016) is stated.
- **c = 1.0:** bypasses CRC and returns everything.
- **Context expansion,** applied after thresholding and counted in the token readout:
  - for each selected element, also include the immediately preceding element when it is in the same section;
  - include the section's heading and first paragraph;
  - keep tables whole, with their captions.

  Expansion only adds elements, so recall can only rise.
- **Token budget.** When given, rank by relevance and add elements and their expansions until the budget is reached. Report the expected recall at that cut, computed from the calibrated probabilities. That figure is not a guarantee.

## 7. API and surfaces

```python
class Selection(BaseModel):
    need: str
    completeness: float
    threshold: float | None
    element_ids: list[str]                  # in reading order
    tokens: int
    token_share: float                      # tokens / document content tokens
    expected_recall: float                  # CRC target, or model estimate under a budget
    guarantee: bool                         # True when the CRC guarantee applies
    chunks: list[Chunk]                     # v0.4 chunk() over the selected elements, contiguous spans merged

def select(analysis: Analysis, need: Field | list[Field] | str, *, completeness: float = 0.9,
           budget: int | None = None, selector: str | Path | None = None) -> Selection
```

- **CLI:** `docrooter select FILE --need <field id | "query"> [--completeness 0.9] [--budget N] [--tagger BUNDLE] [--json]`.
- **Playground:** a field and query box and a completeness slider. The readout shows, for example, "≈ 90% of relevant content kept · 3,240 tokens (12%) · guaranteed on calibration-like documents". Selected elements are highlighted in the Elements tab. Presets: Lean (0.8), Balanced (0.9), Thorough (0.98), Whole document (1.0).

## 8. Evaluation (`benchmarks/selection_bench.py`, results in `docs/benchmarks-selection.md`)

1. **Gold-split field recall** (the v0.5 gold set). For each gold document and each field whose topics appear among its gold elements, relevant elements are the gold elements labelled with the field's topics.
   - **Metrics** per c: the realised recall distribution (mean, 10th percentile), and the token share.
   - **Curve:** recall against token share, plus the tokens needed for 95% recall.
   - **Caveat:** only the 900 annotated elements carry relevance labels. Recall is computed over annotated elements, and tokens over the whole document. The docs say so.
2. **Public evidence QA (evaluation only).** FinanceBench open-source questions (PDFs with evidence pages, CC BY-NC 4.0), never redistributed or trained on. If it is unavailable, use TAT-DQA (CC BY 4.0) instead.
   - **Relevant:** elements on the evidence pages that overlap the evidence text, measured by character overlap of at least 0.5 of the element.
   - **Metrics:** evidence recall against tokens, for query needs and for the closest field.
3. **Baselines:**
   - whole document (c = 1);
   - lead-k tokens;
   - BM25 over elements (top-k to the same token budget);
   - embedding similarity only;
   - structure only (section label match).
4. **Demo extraction harness** (`benchmarks/extraction/`, not in the wheel). It is optional and runs only if an OpenAI-compatible endpoint, or a local MLX model, is configured. It asks the question with the selected context, scores exact match or numeric tolerance against the FinanceBench answers, and compares with the whole document. Jev is not used for this; it is a classifier.
5. **Program targets** (program §6.4): at c = 0.9, realised mean recall ≥ 0.9 on gold documents, with the token share reported.

## 9. Testing

- **Unit tests are torch-free:**
  - CRC threshold computation on synthetic losses, including monotonicity and the guarantee on a simulation;
  - context expansion invariants;
  - c = 1.0 returns everything;
  - budget mode;
  - field loading and validation against taxonomies;
  - `select()` with a fake tagger.
- **Integration (marker `models`):** select on two corpus documents with the trained bundle.

## 10. Left to the builder

- Field set contents. Write a definition per field; review them like taxonomies.
- The query synthesis wording.
- How BM25 tokenises.
- The playground slider UI, within the existing visual language.
