# Train your own element tagger

!!! warning "Experimental"
    This page describes the learned element tagger and its corpus tooling, which ship in 1.0 as **experimental**: outside the
    stability promise, with results pending. The v1 path is [sections first](sections-first.md).

[Element tags](../concepts/element-tags.md) come from a small student model, a BERT-class
encoder, that you train on your own documents. A local LLM, the **teacher**, labels a sample of
your elements; the corpus enriches the taxonomy and filters out the labels the teacher is least
sure of; the student learns from what is left and runs fast, offline, on every element.

```text
 your PDFs ──► corpus build ──► teach label ──► confident set ──► student ──► calibrate ──► bundle
               elements         teacher picks   term mining,      stage A:     temperature    tag, tag-corpus,
               per document     topics and      similarity gap,   heads        per facet      process(...)
                                functions       keep rule         stage B:
                                                                  fine-tune
```

`docrooter train-elements` runs everything after the corpus build, and every stage resumes where
it stopped. The Python example [`examples/train_element_tagger.py`](../../../examples/train_element_tagger.py)
runs the same stages on a tiny corpus with a keyword stand-in for the teacher, so you can see what
each stage produces without a model.

## Before you start

- **Extras.** `pip install 'docrooter[models,layout,teach-mlx]'`: `models` for the student (and
  for a `hf:` teacher), `layout` for the heron layout model that the corpus build uses by default,
  and `teach-mlx` for the default teacher on Apple silicon. Another platform uses a `hf:` teacher
  (extra `models`) or an `openai:` one (extra `llm`).
- **One model at a time, within 8 GiB.** The corpus build, the teacher and the student each load
  a model. Run one such command at a time. The [memory budget](memory-performance.md) applies to
  all of them.
- **Time.** The defaults stop labelling at 12 hours in all, and fine-tuning the body (stage B) at
  6 hours. Stage A, which trains only the heads on a frozen body, is the cheap first run.
- **Documents.** Native PDFs you are allowed to read, text layer included. A scan needs OCR in the
  build's profile (see [OCR and layout models](ocr-and-layout.md)).

## Licensing and terms

Read this before you pick a teacher.

- **Jev cannot be a teacher.** TypeSafe's Master Customer Agreement, section 2.3(b), forbids
  using Jev's output to distil another model. docrooter refuses it everywhere a teacher is named
  (`--teacher`, `element_tags="teacher:..."`, and label files that name it), and the corpus build
  switches off Jev escalation so no Jev output reaches the corpus. Jev stays available for
  [document typing and section normalisation](llm-backends.md); it is only the teacher role that
  is closed.
- **A teacher you bring is under its provider's terms.** The `openai:<model>` engine talks to any
  OpenAI-compatible endpoint that returns `top_logprobs`. You configure it, and you are bound by
  that provider's terms, many of which restrict using output to train other models. docrooter
  cannot check this for you.
- **The default teacher is open.** Qwen3-4B-Instruct-2507 is licensed Apache-2.0. docrooter runs
  its 4-bit MLX conversion, `mlx-community/Qwen3-4B-Instruct-2507-4bit`, pinned to commit
  `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. The fallback, `mlx-community/Qwen3-8B-4bit` at
  `545dc4251c05440727734bcd94334791f6ab0192`, runs with thinking switched off. Model weights carry
  their own licences on their Hugging Face pages; check them before you distribute a bundle.
- **CUAD needs attribution.** If you use the CUAD contracts (CC BY 4.0) in a corpus, credit them:
  CUAD v1, The Atticus Project; Hendrycks, Burns, Chen and Ball, "CUAD: An Expert-Annotated NLP
  Dataset for Legal Contract Review", NeurIPS 2021; https://www.atticusprojectai.org/cuad.
- **Your documents.** A bundle is derived from your documents and the teacher's labels of them.
  The corpus store holds their text; keep it private, as the project does.

## 1. Describe your documents

A manifest is a JSON-lines file with one record per document. These fields matter:

```json
{"id": "fund_2024_a", "sha256": "<64 hex digits>", "split": "train", "doc_type_hint": "financial.fact_sheet"}
```

- `id`: letters, digits, `_` and `-`; the PDF is `<id>.pdf`.
- `split`: `train` for learning, `calib` for calibration (about 5% of the documents), and `gold`
  for documents you keep aside to evaluate on. Labelling and training never read `gold`.
- `doc_type_hint` (or `family_hint`): what you know about the document. It picks the topic
  family, so `financial.fact_sheet` gets the fund-document topics. A document without a hint is
  `generic`.

Without a manifest every PDF in the folder is a `train` document, and you get no calibration.
Keep out of the corpus any document you will evaluate on, by checksum, by URL and by title.

## 2. Build the corpus store

```bash
docrooter corpus build corpus/ --from-manifest manifest.jsonl --pdf-dir pdfs/
docrooter corpus stats corpus/
```

This reads the first 10 pages of each document and a sample of up to 20 more, with the `accurate`
profile (`--profile balanced` is faster and needs no layout model; `--max-pages`, `--first-pages`
and `--seed` change the sample). It keeps every element's text and place, without page furniture
and contents pages, as JSON files under `corpus/`. The build is resumable: a document that failed
or crashed an earlier run is recorded in `corpus/failed.jsonl` and skipped unless you pass
`--retry-failed`.

## 3. Label and train

```bash
docrooter train-elements corpus/ --stage A --out bundles/tagger-a --teacher mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
```

This runs three stages, each skipped on the next run once it has finished:

1. **Label.** For each sampled element the teacher chooses a topic among five shortlisted leaves
   of the document's family (the shortlist comes from the embedding model), and answers yes or no
   for the functions the embedding prescreen thinks likeliest. `--per-doc` (60), `--target`
   (12,000) and `--ceiling-hours` (12) bound the run. To label separately, run
   `docrooter teach label corpus/`, which also takes `--floor`, `--functions-share` and
   `--batch-size`; omit `--teacher` there for the default.
2. **Build the confident set.** Terms are mined from the labelled elements and added to the
   taxonomy as keywords; each element is compared with its candidate classes; an element is kept
   when the teacher's choice is among its closest classes by similarity, or it is among the most
   confident tenth of its class; and each class keeps its top 75% by combined confidence. A topic
   leaf with fewer than 20 confident examples is **left untrained** and predicted only as its
   family; the report lists them.
3. **Train and calibrate.** Stage A trains the heads on the frozen body (a cheap baseline).
   Stage B (`--stage B`, the default) fine-tunes the body, within `--time-budget-hours`;
   `--device cpu` forces the CPU. The function heads are one-vs-rest classifiers on the student's
   embeddings. A temperature per facet is fitted on the calib-fit half of the `calib`
   documents. The `calib` split is halved by document (`docrooter.corpus.calib_half`), and the
   other half, calib-crc, is never fitted by anything: it is kept for conformal calibration.

Teachers are named `mlx:<repo>@<revision>`, `hf:<repo>@<revision>` or `openai:<model>`, and the
revision is required because every label records it. Labels are cached by element text, prompt,
taxonomy and teacher revision, so changing any of them labels again and nothing else does. The
bundle is a directory (`tagger.json` is written last); it is never uploaded anywhere.

Stage B on the same store, once stage A looks sane:

```bash
docrooter train-elements corpus/ --stage B --out bundles/tagger-b --teacher mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b
```

The labelling stage is shared between the two runs, so stage B does not label again.

## 4. Use and check the bundle

```bash
docrooter tag report.pdf --bundle bundles/tagger-b
docrooter tag report.pdf --bundle bundles/tagger-b --json
docrooter tag-corpus corpus/ --bundle bundles/tagger-b --split calib
docrooter serve --element-tags bundles/tagger-b
```

`docrooter tag` runs the rooter with the bundle as its `element_tags` and prints each tagged
element's topic and functions. In Python it is `process(path, element_tags="bundles/tagger-b")`,
which loads the model once and closes it at the end; use `process_many` for a folder.
`tag-corpus` writes `labels/tagger.jsonl` into the store (it never overwrites the training
labels), and it tags every split by default, gold included, so pass `--split` when you hold one
back. To see how a bundle compares with the teacher, tag a sample with `--bundle
teacher:mlx:...` and compare. **Evaluate on documents you labelled yourself, and score against
your labels, never the teacher's.**

## The stages in Python

The example builds a corpus of 64 fund-fact-sheet elements in a temporary folder, labels it with
a keyword teacher and a keyword embedder (they stand in for the LLM and UltraBERT-Embed and
implement the same protocols), and builds the confident set. No model and no network.

```python
topics = fund_topics()
corpus = tiny_corpus(directory)
embedder, teacher = KeywordEmbedder(), KeywordTeacher()
report = label_corpus(
    corpus,
    teacher,
    embedder,
    sample=SampleSpec(per_doc=8, target=64, floor=0),
    topics=topics,
)
enriched, training_set = build_training_set(
    corpus, corpus.labels("teacher"), topics, embedder, min_per_class=3
)
dataset = to_dataset(training_set, corpus)
```

```text
keyword:overlap@1: 64 topics, 32 functions
fund_document.investment_objective: 16 labelled, 12 kept, trained
fund_document.fees_and_expenses: 16 labelled, 12 kept, trained
fund_document.principal_risks: 12 labelled, 9 kept, trained
fund_document.performance: 20 labelled, 12 kept, trained
fund_document.other: 0 labelled, 0 kept, left untrained
fees keywords: fee, expenses, expense
45 training examples; the first:
['fund_document', 'fund_document.fees_and_expenses'] [paragraph] The management fee is 0.45% a year and
training itself needs the models extra: docrooter train-elements
```

`min_per_class=3` is only for this toy size; the default is 20. The toy teacher is crude (it
counts shared words), which is the point of the other stages: the similarity gap and the keep
rule exist because a teacher is sometimes wrong.

## What we trained on, and what to expect

The v0.5 corpus is a manifest plus fetch scripts under `benchmarks/training_corpus/`. No PDF is
committed or redistributed. It lists 1,100 public native-PDF documents with 81,822 pages in all:
925 in `train`, 62 in `calib` and 113 in `gold`. Sources are company annual reports, fund fact
sheets and prospectuses, GAO and CRS reports, Federal Reserve publications, CC-BY and CC0 arXiv
papers, federal court opinions, RFCs and CUAD contracts.

- **No EDGAR.** The plan was SEC Form ARS annual reports and exhibit-10 contracts, but the SEC
  refused the project's User-Agent, which names no contact email. Annual reports come from issuers'
  own investor-relations sites instead, mostly of non-US issuers, and the contracts from CUAD.
- **The gold set is Claude-annotated.** 900 elements were labelled by Claude subagents from the
  taxonomy definitions alone, 180 of them twice and independently, then adjudicated by a third.
  Cohen's kappa on the topic facet is 0.953. All annotators share one underlying model, so kappa
  measures consistency, not human-verified correctness: a human might disagree with all of them.
- **Performance depends on corpus size.** A tagger trained on your 200 documents will not match
  one trained on 1,100. The learning curve (the student trained on an eighth, a quarter, half and
  all of the confident set) is on the benchmark page.
- **Results** are on [Element tags (gold set)](../benchmarks/element-tags.md), with the method.
  No trained bundle is published; the point of this page is that you train your own.

## Next

- [Element tags](../concepts/element-tags.md): the taxonomies and what an element carries.
- [Training and evaluation](training-evaluation.md): training section classifiers, the same
  `train()` underneath.
- [Memory and performance](memory-performance.md).
