# Element tag benchmarks

> **Results pending.** This page is a skeleton: the harness (`benchmarks/element_tags_bench.py`) is in place, and the numbers below are filled in from `benchmarks/results/<date>/element-tags*.json` once the student is trained and the runs are made. No number on this page is final until this notice is removed.

The element tagger (spec §11) is measured on an independently annotated gold set, never on teacher labels.

## Gold set

- 900 elements from the training corpus's `gold` split, stratified by document family, element kind and a teacher-free zero-shot topic. `benchmarks/training_corpus/gold.jsonl` holds keys and labels only; each element's text is read from the corpus store and checked against its `text_sha1`.
- **Disclosure: Claude-annotated gold.** The labels were written by Claude subagents from the taxonomy definitions only, without seeing any model output. 180 elements were annotated twice, independently, and the disagreements adjudicated by a third annotator. Agreement is in `benchmarks/training_corpus/gold_agreement.json`.
- The gold labels were committed before any model scored a gold document.

## Systems

| Name | What |
| - | - |
| `keyword` | Keyword overlap between the element and each tag's definition (no model). |
| `zero-shot` | UltraBERT-Embed-150M, zero-shot, against the tag definitions. |
| `zero-shot-enriched` | The same, with each topic leaf's corpus-mined key terms appended to its definition. |
| `teacher-4b` | The teacher, Qwen3-4B-Instruct-2507 (4-bit, MLX): topic choice among the family's leaves and a yes/no question per function tag. |
| `teacher-8b` | The fallback teacher, Qwen3-8B (4-bit, MLX), on a 300-element subset stratified by family. |
| `stage-a` | The student with frozen-body heads. |
| `stage-b` | The student with a fine-tuned body, before and after the active-learning round. |

## Scoring

- **Two modes.** *Primary*: the family comes from the gold document's known type; the program targets apply here. *End to end*: the family of the doc type the rooter predicts. Both are reported.
- **Topic:** accuracy and macro P/R/F1 per family and overall, and hierarchical F1 over family → leaf. Macro averages run over the classes present in gold (per family, that family's gold classes), so every system is scored on the same class set; a prediction outside it counts only as a miss.
- **Function:** micro and macro F1, and per-tag F1.
- **Calibration:** expected calibration error (15 bins) before and after temperature scaling.
- **Throughput and memory:** elements/s and peak memory per system, each in its own process under the one-model guard. The teacher's items/s counts the items it was actually asked.
- **No source hints:** every PDF a model reads is a neutral copy named `doc_<sha256[:12]>.pdf`, because the doc-type view falls back to the file name for a title and the corpus ids name their source.

## Results

_Results pending._

## Learning curve

_Results pending._ The student trained on 1/8, 1/4, 1/2 and all of the confident set, scored on gold: the evidence for "performance depends on corpus size".

## Document type with element evidence (dev corpus)

_Results pending._ Each dev document scored by the rooter's own doc type (the v0.3/v0.4 baseline), and by the fixed mix `0.7 · view + 0.3 · element evidence` (never tuned on dev). The view is the rooter's top-10 leaf distribution renormalised over the doc-type leaves, so the two parts each sum to 1.

## Reproducing

```
PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench gold-doctypes
PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench run --system keyword \
    --doc-types benchmarks/results/<date>/element-tags-gold-doctypes.json
PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench run --system stage-b \
    --tagger benchmarks/.cache/bundles/element-tagger-v1 --device mps \
    --doc-types benchmarks/results/<date>/element-tags-gold-doctypes.json
PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench learning-curve --stage A
PYTHONPATH=src .venv/bin/python -m benchmarks.element_tags_bench doctype-evidence \
    --bundle benchmarks/.cache/bundles/element-tagger-v1
```
