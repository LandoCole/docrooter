"""The v0.5 training corpus: per-source candidate listings and the polite fetcher (spec §5).

    PYTHONPATH=src python -m benchmarks.training_corpus.fetch --target 1100 [--dry-run]

PDFs go to ``benchmarks/.cache/training_corpus/pdfs/<id>.pdf`` and are never committed; the
committed record is ``manifest.jsonl`` (URLs, checksums, licences, splits).
"""
