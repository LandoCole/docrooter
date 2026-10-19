# Python API

Everything below is importable from the top-level package (`from docrooter import ...`) and
listed in `docrooter.__all__`, except the result, event and backend types at the end, which
are returned by these functions or needed to extend them. This page is generated from the
docstrings. (`process`, `summarize`, `embed` and `select` are documented from the modules that
define them, `docrooter.rooter.process`, `docrooter.summarize`, `docrooter.embed` and
`docrooter.select.selection`, as their names are taken by modules; import them from
`docrooter`.)

## Loading documents

::: docrooter.load

::: docrooter.load_pdf

::: docrooter.Document

::: docrooter.Section

## Elements

::: docrooter.Element

::: docrooter.ElementKind

::: docrooter.Box

::: docrooter.TableData

::: docrooter.FigureData

::: docrooter.TagResult

## The rooter

::: docrooter.rooter.process.process

::: docrooter.process_many

::: docrooter.RouterConfig

::: docrooter.Analysis

::: docrooter.register_ocr_engine

::: docrooter.register_layout_engine

## Summaries

::: docrooter.summarize.summarize

::: docrooter.Summary

## Chunks and embeddings

::: docrooter.chunk

::: docrooter.Chunk

::: docrooter.embed.embed

::: docrooter.EmbeddingSet

## Selection

::: docrooter.select.selection.select

::: docrooter.Selection

::: docrooter.Selector

::: docrooter.Field

::: docrooter.builtin_fields

::: docrooter.load_fields

## Taxonomies

::: docrooter.Taxonomy

::: docrooter.TaxonomyNode

::: docrooter.builtin_taxonomy

::: docrooter.list_builtin_taxonomies

## Classification

::: docrooter.Classifier

::: docrooter.classify_document

::: docrooter.normalize_sections

### Strategies

::: docrooter.Greedy

::: docrooter.Beam

::: docrooter.Flat

## Views

::: docrooter.SectionView

::: docrooter.DocumentView

## Training and evaluation

::: docrooter.Dataset

::: docrooter.train

::: docrooter.TrainedModel

::: docrooter.calibrate

::: docrooter.evaluate

::: docrooter.compare

## Runtime

::: docrooter.configure

## Results

::: docrooter.DocTypeResult

::: docrooter.NormalizationResult

::: docrooter.classify.results.DocumentResult

::: docrooter.classify.results.SectionResult

::: docrooter.classify.results.PathResult

::: docrooter.classify.results.Step

::: docrooter.classify.normalize.SectionNorm

## Events

::: docrooter.events.Event

::: docrooter.events.pipeline_run

## Backends

::: docrooter.backends.Scores

::: docrooter.backends.register_backend_kind

## Errors

::: docrooter.DocrooterError

::: docrooter.MissingExtraError

::: docrooter.TaxonomyError

::: docrooter.UnknownTaxonomyError

::: docrooter.BackendError

::: docrooter.DatasetError
