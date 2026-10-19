# Selection benchmarks

> **Results pending.** This page is a skeleton. The harnesses (`benchmarks/selection_bench.py`, `benchmarks/evidence_qa.py` and `benchmarks/extraction/run.py`) are in place. The numbers are filled in from `benchmarks/results/<date>/selection*.json` once the selector is fitted and the runs are made. No number on this page is final until this notice is removed.

`select()` answers one question: which elements should a downstream LLM read for a need, at completeness c? This page measures two things:

- whether the share of relevant content it keeps matches what it promises;
- how many tokens that costs, against simple baselines given the same tokens.

## Method

*Results pending.*

- **Documents.** The gold split of the training corpus: every gold document that has full teacher labels. Each one is read in two ways, and both are reported side by side (see [two readings](#two-readings-fit-identical-and-runtime)):
  - the **fit-identical reading**, exactly as the corpus build read it and the selector was fitted (same profile, page cap and pages; sections normalised with the doc type of the manifest's family), then tagged by the trained element tagger and scored by the fitted selector through `select()`'s own cut, dial and expansion. The doc type comes from the manifest, so no classifier error can drop the guarantee here: this row measures the dial, not the routing;
  - the **runtime path**, the way a user reads a document.
- **Units.** A unit is a (document, built-in field of the document's family) pair with at least one relevant content element.
- **Relevance (main number).** An element is relevant to a field when its teacher topic, from the local teacher's full labelling, is one of the field's topics. This is the target the dial was calibrated on (CRC).
- **Recall** is |selected ∩ relevant| / |relevant| over content elements, after context expansion, since that is what the reader gets. The **token share** is the selection's tokens over the document's content and heading tokens.
- **Grid.** c ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0}. Per c, per family and overall, we report:
  - the mean, 10th percentile and minimum recall;
  - the mean token share;
  - the shares of units with `guarantee=True` and `capped=True`.
- **Program target** (spec §0.5): mean teacher-relevance recall ≥ c at c = 0.9. It is judged twice, each verdict reported as PASS or FAIL:
  - the headline counts a capped unit (the whole document) at recall 1, since that is what the reader gets;
  - the uncapped-only verdict leaves capped units out and judges the dial itself.

  Both verdicts are shown with the share of capped units, the skipped documents and their reasons, and the gold elements whose text changed.
- **Coverage.** Every gold-split document is accounted for: read, with units, read without units, or skipped with its reason. A document without teacher labels is not skipped: its gold subset is evaluated (it is counted as "gold subset only"). The gold subset is evaluated whenever a document reads, whatever its teacher labels say.
- **Dials.** The per-family dial size n, the number of documents its units come from, and the slack 1/(n+1) are printed in a table.
- **Overrides.** A `--tagger` or `--embedder` that differs from the selector's recorded fit is logged as a WARNING and recorded in the results (`provenance.overrides`): the guarantee does not apply to such a run.

### Two readings: fit-identical and runtime

The guarantee holds on documents read, tagged and embedded as the selector was fitted. The fit-identical row measures exactly that. The runtime row measures what a user gets from the documented flow:

- each gold PDF goes through `process(pdf, profile=<the fit's profile>, pages=<the pages the build read>, element_tags=<the bundle>)`, with the profile's own doc-type classification and section labelling;
- then `select()` runs on that analysis with the selector and no tagger, as in the README. (The benchmark computes `select()`'s own scoring once per document, `prepare` then `select_scored`, which a unit test pins equal to `select()`.)

The runtime reading's elements are matched to the teacher labels by element id and text hash. Elements that do not match are excluded and counted, and the match rate is reported. The runtime row has its own program-target verdict and guarantee share, plus the counts of units with a `fit_mismatch` (a reading that differs from the fit's) and with a `doc_type_mismatch` (the classifier gave a type of another family, which drops the guarantee). A gap between the two rows is the cost of the runtime path: classification errors, a different section reading, and labels that no longer match.

## Main evaluation: teacher relevance

*Results pending.*

| family | c | n | recall mean | p10 | min | token share | guarantee | capped |
| - | - | - | - | - | - | - | - | - |
| overall | 0.90 | – | – | – | – | – | – | – |

## Secondary: gold-annotated subset and teacher–gold disagreement

*Results pending.*

- **Gold subset.** Relevant elements are the gold elements whose gold topic is one of the field's topics. Recall is computed over gold-annotated elements only, and tokens over the whole document, at the same grid. A gold element whose text changed in the re-read is counted and excluded.
- **Disagreement.** On the 900 gold elements, the teacher topic is compared with the gold topic. We report accuracy overall and per family, and the 10 most frequent confusions.

## Baselines

*Results pending.*

Every baseline gets the same tokens as the selector: for each unit and each c < 1, its budget is the selector's token count at that c.

The fills of BM25, similarity only and structure only are greedy: an element that does not fit is skipped, and smaller ones after it are still tried. This is slightly generous to the baselines, so it errs in their favour. Lead-k stops at the first element that does not fit.

| baseline | how it picks |
| - | - |
| whole document | Everything. Recall 1, share 1: the reference row. |
| lead-k | Content elements in reading order, while the next one fits. |
| BM25 | Elements ranked by BM25 (k1 = 1.5, b = 0.75; lower-cased words without stopwords) against the field's query, then each one that still fits. No expansion. |
| similarity only | The same fill, ranked by the embedding similarity feature. |
| structure only | Elements whose section matches the field, in reading order, then nothing. |

## The curve

*Results pending.*

The curve plots mean recall against mean token share. It is computed over a fine grid of thresholds on the selector's relevance probability, with expansion applied. **Tokens-to-95** is the smallest mean token share at which the mean recall reaches 0.95. It is reported from the curve and from the dial's own c grid.

## Evidence QA (evaluation only)

*Results pending.*

- **Dataset.** FinanceBench's open-source questions, with their evidence pages and texts (CC BY-NC 4.0). The data is used for evaluation only: it is never redistributed, trained on or calibrated on, and is kept in a git-ignored cache. If FinanceBench is unavailable, TAT-DQA (CC BY 4.0) is used instead, and the results name the dataset used.
- **Relevance.** An element is relevant when it lies on an evidence page and at least half of its characters overlap the evidence text.
- **Needs.**
  - The question itself as a free-text need: never guaranteed, so its numbers are descriptive.
  - The built-in field closest to the question, at c ∈ {0.8, 0.9, 0.98}. A closest-field row may carry `guarantee=True`, but that flag is about teacher relevance on calibration-like readings. It says nothing about evidence recall on these whole filings.

## Extraction demo

*Results pending.*

`benchmarks/extraction/run.py` is optional and is not part of the package. It runs only when an OpenAI-compatible endpoint or a local MLX model is configured, and it never uses Jev. It asks each question twice: once with the context selected at c = 0.9, and once with the whole document. Answers are scored by exact match or 1% numeric tolerance. The report gives the accuracy, the tokens, and the difference between the two.

## Caveats

*Results pending:* the per-family n, the capped shares and the measured teacher–gold disagreement go here once the runs are made.

- **Pairs from one document are correlated.** The fields of one document share its reading, tags and section structure, so units are not independent. CRC's exchangeability holds at best across documents, not across the pairs of one document.
- **The gold set is Claude-annotated.** The 900 gold labels were written by Claude subagents from the taxonomy definitions, not by domain experts. The teacher–gold disagreement shows how far the two label sources differ.
- **Teacher relevance is the CRC target.** The dial was calibrated on teacher labels, so the guarantee is a statement about teacher relevance. It is never claimed against gold labels, which are shown beside it.
- **Small calibration sets.** The dial's slack is 1/(n+1) for n calibration units per family. n per family is reported. Where the slack makes a target unreachable, the dial is capped and returns the whole document.
- **Queries are never guaranteed.** Free-text needs (including the evidence-QA questions) carry the model's estimate only, so their results are descriptive.
