# Selection benchmarks

> **Results pending.** This page is a skeleton. The harnesses (`benchmarks/selection_bench.py`, `benchmarks/evidence_qa.py` and `benchmarks/extraction/run.py`) are in place. The numbers are filled in from `benchmarks/results/<date>/selection*.json` once the selector is fitted and the runs are made. No number on this page is final until this notice is removed.

`select()` answers one question: which elements should a downstream LLM read for a need, at completeness c? This page measures two things:

- whether the share of relevant content it keeps matches what it promises;
- how many tokens that costs, against simple baselines given the same tokens.

## Method

*Results pending.*

- **Documents.** The gold split of the training corpus: every gold document that has full teacher labels. Each one is read again exactly as the corpus build read it (same profile, page cap and pages). It is then tagged by the trained element tagger and scored by the fitted selector through `select()`'s own cut, dial and expansion.
- **Units.** A unit is a (document, built-in field of the document's family) pair with at least one relevant content element.
- **Relevance (main number).** An element is relevant to a field when its teacher topic, from the local teacher's full labelling, is one of the field's topics. This is the target the dial was calibrated on (CRC).
- **Recall** is |selected ∩ relevant| / |relevant| over content elements, after context expansion, since that is what the reader gets. The **token share** is the selection's tokens over the document's content and heading tokens.
- **Grid.** c ∈ {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0}. Per c, per family and overall, we report:
  - the mean, 10th percentile and minimum recall;
  - the mean token share;
  - the shares of units with `guarantee=True` and `capped=True`.
- **Program target** (spec §0.5): mean teacher-relevance recall ≥ c at c = 0.9, reported as PASS or FAIL.

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
  - The built-in field closest to the question, at c ∈ {0.8, 0.9, 0.98}.

## Extraction demo

*Results pending.*

`benchmarks/extraction/run.py` is optional and is not part of the package. It runs only when an OpenAI-compatible endpoint or a local MLX model is configured, and it never uses Jev. It asks each question twice: once with the context selected at c = 0.9, and once with the whole document. Answers are scored by exact match or 1% numeric tolerance. The report gives the accuracy, the tokens, and the difference between the two.

## Caveats

- **Pairs from one document are correlated.** The fields of one document share its reading, tags and section structure, so units are not independent. CRC's exchangeability holds at best across documents, not across the pairs of one document.
- **The gold set is Claude-annotated.** The 900 gold labels were written by Claude subagents from the taxonomy definitions, not by domain experts. The teacher–gold disagreement shows how far the two label sources differ.
- **Teacher relevance is the CRC target.** The dial was calibrated on teacher labels, so the guarantee is a statement about teacher relevance. It is never claimed against gold labels, which are shown beside it.
- **Small calibration sets.** The dial's slack is 1/(n+1) for n calibration units per family. n per family is reported. Where the slack makes a target unreachable, the dial is capped and returns the whole document.
- **Queries are never guaranteed.** Free-text needs (including the evidence-QA questions) carry the model's estimate only, so their results are descriptive.
