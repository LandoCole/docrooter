"""The corpus store and builder (spec §5, §6): a document corpus's elements and their labels,
as JSON files in one directory, written so that long runs resume without redoing or corrupting
finished work.

:class:`Corpus` is the store (manifest, per-document element files, label files, stats);
:func:`build_corpus` reads PDFs into it with the rooter.
"""

from docrooter.corpus.build import (
    SKIPPED_KINDS,
    BuildReport,
    build_corpus,
    corpus_elements,
    select_pages,
)
from docrooter.corpus.store import (
    CALIB_HALVES,
    SPLITS,
    Corpus,
    CorpusElement,
    LabelKey,
    LabelRow,
    ManifestRecord,
    calib_half,
    is_calib_crc,
    read_manifest,
    text_sha1,
)

__all__ = [
    "CALIB_HALVES",
    "SKIPPED_KINDS",
    "SPLITS",
    "BuildReport",
    "Corpus",
    "CorpusElement",
    "LabelKey",
    "LabelRow",
    "ManifestRecord",
    "build_corpus",
    "calib_half",
    "corpus_elements",
    "is_calib_crc",
    "read_manifest",
    "select_pages",
    "text_sha1",
]
