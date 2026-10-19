"""Fitting the selector (spec §0.1–§0.3, §5, §6): the four relevance combiners on calib-fit, the
completeness dials on calib-crc, relevance from the ``teacher-full`` labels.

**Documents.** Only manifest records with ``split == "calib"`` are read: calib-fit
(:func:`~docrooter.corpus.calib_half` ``== "fit"``) trains the combiners, calib-crc
(:func:`~docrooter.corpus.is_calib_crc`) calibrates the dials. A train or gold document is never
read, whatever half its id hashes to. A document's family is its manifest's
(:func:`~docrooter.teach.shortlist.doc_family`); one whose family has no field set is skipped
and counted.

**Features.** Each document is re-analysed (``analyse``) and scored with
:func:`~docrooter.select.selection.prepare` and
:func:`~docrooter.select.selection.need_features`, the functions :func:`select` uses, so the
combiners and dials are fitted on exactly the features they will see. The default ``analyse``
(:class:`CorpusPdfReader`) reads the PDF as the corpus build did (the profile's engines, the
same pages), then normalises its sections with the family's section taxonomy and a doc type
of that family (:func:`section_doc_type`) for ``section_hit``. Document-type classification
stays off: the family comes from the manifest.

**Relevance.** An element is relevant to a field when its ``teacher-full`` topic (the argmax
of its row) is one of the field's topics. Content elements are matched to labels by element id
and checked by text hash. An element without text (the labeller skips those) may have no label
and counts as not relevant; more than :data:`MAX_MISSING_LABELS` of a document's elements with
text unlabelled (an incomplete labelling run) raises, naming the document. One whose label was
given to another text is a hash mismatch, and more than :data:`MAX_HASH_MISMATCH` of a
document's content elements mismatching raises too (a reading that diverged from the corpus
would void the calibration).

**Rows** (spec §5). For each calib-fit document and each field of its family: the field
combiner and its fallback (on ``("sim", "section_hit")``) get the field's feature rows; the
query combiner and its fallback get the rows of the field's query (its ``query``, else its
description) as a free-text need, with the field's relevance. **Units** (spec §6). For each
calib-crc document and field: one :class:`~docrooter.select.crc.CalibUnit` with the fitted
field combiner's probabilities over every content element.

The returned :class:`~docrooter.select.selection.Selector` records the embedder spec and, in
``meta``, the profile, tagger, corpus manifest hash, document counts, label checks, class
balance, dial sizes and fit warnings.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from docrooter.backends.base import close_backend
from docrooter.corpus.store import Corpus, ManifestRecord, calib_half, is_calib_crc, text_sha1
from docrooter.errors import DocrooterError
from docrooter.select.crc import CalibUnit, calibrate_dial
from docrooter.select.fields import (
    Field,
    builtin_fields,
    calibration_of,
    list_builtin_field_sets,
)
from docrooter.select.relevance import Combiner
from docrooter.select.selection import (
    FALLBACK_FEATURES,
    DocContext,
    Selector,
    need_features,
    prepare,
)

if TYPE_CHECKING:
    from docrooter.rooter.analysis import Analysis

__all__ = [
    "LABELER",
    "MAX_HASH_MISMATCH",
    "MAX_MISSING_LABELS",
    "CorpusPdfReader",
    "fit_selector",
    "label_leaves",
    "labels_by_document",
    "section_doc_type",
]

MAX_HASH_MISMATCH = 0.02
"""The largest share of a document's content elements whose label is for another text."""
MAX_MISSING_LABELS = 0.02
"""The largest share of a document's content elements with text that may lack a label."""
LABELER = "teacher-full"
DEFAULT_PROFILE = "accurate"
DEFAULT_MAX_PAGES = 30  # the corpus build's

Analyse = Callable[[ManifestRecord], "Analysis"]
Progress = Callable[[int, int, str, str], None]
"""``on_progress(done, total, doc_id, half)`` after each calib document is scored."""


class CorpusPdfReader:
    """Reads a corpus document's PDF again as the corpus build read it.

    The PDF is ``pdf_dir/<id>.pdf`` (as ``docrooter corpus build`` finds it) and must hash to
    the record's sha256. It is read with :func:`~docrooter.corpus.build.reading_config` (the
    profile's text, OCR and layout engines; everything else off) on the pages the build read,
    which the store recorded; their number must be what ``max_pages`` gives the PDF's page
    count, else the build used another page cap. The sections are then normalised with the
    section taxonomy of the document's family and the doc type :func:`section_doc_type` names
    (so the section prompt is the one a correctly typed document gets at runtime), by the
    profile's ``sections`` backend, or by ``sections_backend`` when the caller shares one (the
    caller closes it). Engines and an own backend are created on first use and closed by
    :meth:`close`.
    """

    def __init__(
        self,
        corpus: Corpus,
        pdf_dir: str | os.PathLike[str],
        *,
        profile: str = DEFAULT_PROFILE,
        max_pages: int = DEFAULT_MAX_PAGES,
        sections_backend: Any = None,
    ) -> None:
        from docrooter.corpus.build import reading_config
        from docrooter.rooter.process import _Engines
        from docrooter.rooter.profiles import profile_defaults

        self.corpus = corpus
        self.folder = Path(pdf_dir)
        self.cfg = reading_config(profile, max_pages)
        self.sections_spec = str(profile_defaults(profile)["sections"])
        self._engines = _Engines(self.cfg, {})
        self.max_pages = max_pages
        self._sections: Any = sections_backend
        self._owns_sections = sections_backend is None

    def __call__(self, record: ManifestRecord) -> Analysis:
        from docrooter.classify.normalize import normalize_sections
        from docrooter.corpus.build import _page_count, _sha256
        from docrooter.rooter import process
        from docrooter.teach.shortlist import doc_family

        path = self.folder / f"{record.id}.pdf"
        if not path.is_file():
            raise DocrooterError(f"no PDF for {record.id} at {path}; pass --pdf-dir")
        if _sha256(path) != record.sha256:
            raise DocrooterError(f"{path} does not hash to {record.id}'s manifest sha256")
        pages = self.corpus.pages_read(record.id)
        n_pages = _page_count(path, self.cfg.text)
        if len(pages) != min(n_pages, self.max_pages) or any(p > n_pages for p in pages):
            raise DocrooterError(
                f"{record.id}: the store read {len(pages)} of its {n_pages} pages, which a page "
                f"cap of {self.max_pages} cannot give; pass the build's --max-pages"
            )
        layout, _ = self._engines.layout()
        ocr, _ = self._engines.ocr()
        analysis = process(
            path,
            config=self.cfg,
            pages=pages,
            layout=layout,
            ocr="none" if ocr is None else ocr,
        )
        if self._sections is None:
            from docrooter.backends.registry import resolve_backend

            self._sections = resolve_backend(self.sections_spec)
        family = doc_family(record)
        sections = normalize_sections(
            analysis.document,
            doc_type=section_doc_type(record, family),
            taxonomy=f"section_types/{family}",
            backend=self._sections,
        )
        return analysis.model_copy(update={"sections": sections})

    def close(self) -> None:
        try:
            self._engines.close()
        finally:
            if self._owns_sections and self._sections is not None:
                close_backend(self._sections)
                self._sections = None


def section_doc_type(record: ManifestRecord, family: str) -> str | None:
    """The doc type whose section prompt fit-time normalisation uses for a ``family`` document:
    the manifest's doc type when it maps to the family, else the family's first mapped doc type
    (``None`` when none maps to it)."""
    from docrooter.taxonomies import DOC_TYPE_TO_SECTION_TAXONOMY, family_for_doc_type

    hint = record.doc_type_hint
    if hint in DOC_TYPE_TO_SECTION_TAXONOMY and family_for_doc_type(hint) == family:
        return hint
    wanted = f"section_types/{family}"
    return next((t for t, name in DOC_TYPE_TO_SECTION_TAXONOMY.items() if name == wanted), None)


class _Rows:
    """Feature rows, labels and document ids of one combiner pair."""

    def __init__(self) -> None:
        self.X: list[np.ndarray] = []
        self.y: list[np.ndarray] = []
        self.docs: list[str] = []

    def add(self, X: np.ndarray, y: np.ndarray, doc_id: str) -> None:
        self.X.append(X)
        self.y.append(y)
        self.docs.extend([doc_id] * len(y))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.X:
            return np.zeros((0, 0)), np.zeros(0)
        return np.vstack(self.X), np.concatenate(self.y)

    def balance(self) -> dict[str, Any]:
        rows = len(self.docs)
        relevant = int(sum(int(y.sum()) for y in self.y))
        return {"rows": rows, "relevant": relevant, "share": relevant / rows if rows else 0.0}


def labels_by_document(corpus: Corpus) -> dict[str, dict[str, dict[str, str]]]:
    """``doc_id -> element_id -> text sha1 -> teacher-full topic leaf``. The selection benchmark
    reads the gold documents' relevance through this and :func:`label_leaves`, so recall is
    measured on exactly the labels and reading the selector was fitted on."""
    from docrooter.taxonomies import builtin_taxonomy
    from docrooter.teach.enrich import teacher_choices

    rows = corpus.labels(LABELER)
    if not rows:
        raise DocrooterError(
            f"the corpus at {corpus.dir} has no {LABELER} labels; label the calib split in "
            "full mode first (docrooter teach label --mode full)"
        )
    out: dict[str, dict[str, dict[str, str]]] = {}
    for key, choice in teacher_choices(rows, builtin_taxonomy("element_topics")).items():
        out.setdefault(key.doc_id, {}).setdefault(key.element_id, {})[key.text_sha1] = choice.label
    return out


def label_leaves(
    ctx: DocContext, labels: Mapping[str, Mapping[str, str]], doc_id: str
) -> tuple[list[str | None], dict[str, Any]]:
    """Each content element's teacher-full leaf (``None`` when unlabelled or mismatched), and
    the document's label check. An element the labeller skips (no text) may lack a label
    (``no_label``); one with text may not (``missing``). Raises when more than
    :data:`MAX_MISSING_LABELS` of the elements with text are missing or more than
    :data:`MAX_HASH_MISMATCH` of all have a label for another text."""
    index = ctx.index
    leaves: list[str | None] = []
    no_label = missing = mismatch = with_text = 0
    for pos in index.content:
        element = index.doc.elements[pos]
        labellable = _has_text(element)
        with_text += labellable
        by_hash = labels.get(element.id)
        if by_hash is None:
            if labellable:
                missing += 1
            else:
                no_label += 1
            leaves.append(None)
            continue
        leaf = by_hash.get(text_sha1(element.text))
        if leaf is None:
            mismatch += 1
        leaves.append(leaf)
    content = len(index.content)
    if with_text and missing / with_text > MAX_MISSING_LABELS:
        raise DocrooterError(
            f"{doc_id}: {missing} of its {with_text} content elements with text have no "
            f"{LABELER} label (more than {MAX_MISSING_LABELS:.0%}): its labelling is "
            "incomplete (a run stopped early?); finish it (docrooter teach label --mode full) "
            "before fitting"
        )
    if content and mismatch / content > MAX_HASH_MISMATCH:
        raise DocrooterError(
            f"{doc_id}: {mismatch} of {content} content elements have a {LABELER} label for "
            f"another text (more than {MAX_HASH_MISMATCH:.0%}): this reading diverged from the "
            "corpus build, so its labels do not apply; read it with the build's profile and pages"
        )
    check = {
        "content": content,
        "no_label": no_label,
        "missing": missing,
        "hash_mismatch": mismatch,
        "rate": (no_label + missing + mismatch) / content if content else 0.0,
    }
    return leaves, check


def _has_text(element: Any) -> bool:
    """Whether the full-mode labeller labels this element (``teach.label._has_text`` on the
    stored element): a table with cells, or any element with text."""
    table = getattr(element, "table", None)
    if element.kind == "table" and table is not None and table.to_markdown().strip():
        return True
    return bool(element.text.strip())


def _query_of(field: Field) -> str:
    """The free-text need synthesised from a field (spec §5): its query, else its description."""
    return field.query or field.description


def _fit(role: str, combiner: Combiner, rows: _Rows, fit_warnings: list[str]) -> Combiner:
    X, y = rows.arrays()
    if not len(y) or y.min() == y.max():
        raise DocrooterError(
            f"cannot fit the {role} combiner: calib-fit has {len(y)} rows and "
            f"{int(y.sum()) if len(y) else 0} relevant ones; it needs relevant and irrelevant "
            "elements"
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        combiner.fit(X, y, doc_ids=rows.docs)
    fit_warnings.extend(f"{role}: {w.message}" for w in caught)
    return combiner


def _spec_of(value: Any) -> str | None:
    if isinstance(value, (str, Path)):
        return str(value)
    spec = getattr(value, "spec", None) or getattr(value, "name", None)
    return spec if isinstance(spec, str) else None


def fit_selector(
    corpus: Corpus,
    tagger: Any,
    embedder: Any,
    *,
    fields_by_family: Mapping[str, Sequence[Field]] | None = None,
    analyse: Analyse | None = None,
    profile: str = DEFAULT_PROFILE,
    pdf_dir: str | os.PathLike[str] | None = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    on_progress: Progress | None = None,
) -> Selector:
    """Fit the selector on ``corpus``'s calib split (module docstring).

    ``tagger`` is an ``element_tags`` spec (a bundle directory) or an object with
    ``tag(elements, family)``; ``embedder`` an embedding backend spec or an object with
    ``encode``. A spec is loaded once here and closed at the end. ``fields_by_family`` defaults
    to every built-in field set.

    ``analyse(record) -> Analysis`` reads a calib document; by default a
    :class:`CorpusPdfReader` over ``pdf_dir`` (default ``<corpus>/pdfs``) with ``profile`` and
    ``max_pages``, which must be the ones the corpus was built with. ``profile`` is recorded.
    """
    if tagger is None or embedder is None:
        raise ValueError("fitting the selector needs a tagger and an embedder (the full features)")
    fields = (
        {family: list(fs) for family, fs in fields_by_family.items()}
        if fields_by_family is not None
        else {family: builtin_fields(family) for family in list_builtin_field_sets()}
    )
    from docrooter.teach.shortlist import doc_family

    labels = labels_by_document(corpus)
    calib = [r for r in corpus.manifest() if r.split == "calib"]
    fit_docs = [r for r in calib if calib_half(r.id) == "fit"]
    crc_docs = [r for r in calib if is_calib_crc(r)]
    skipped: Counter[str] = Counter()
    todo: list[tuple[ManifestRecord, str, str]] = []
    for half, records in (("fit", fit_docs), ("crc", crc_docs)):
        for record in records:
            family = doc_family(record)
            if fields.get(family):
                todo.append((record, family, half))
            else:
                skipped[family] += 1

    owned: list[Any] = []
    reader: CorpusPdfReader | None = None
    try:
        tagger_obj = tagger
        if isinstance(tagger, (str, Path)):
            from docrooter.tagger.element_tags import resolve_tagger

            tagger_obj = resolve_tagger(str(tagger))
            owned.append(tagger_obj)
        encoder = embedder
        if isinstance(embedder, str):
            from docrooter.backends.registry import resolve_backend

            encoder = resolve_backend(embedder)
            owned.append(encoder)
        if analyse is None:
            from docrooter.rooter.profiles import profile_defaults

            folder = Path(pdf_dir) if pdf_dir is not None else corpus.dir / "pdfs"
            same = isinstance(embedder, str) and embedder == profile_defaults(profile)["sections"]
            reader = CorpusPdfReader(
                corpus,
                folder,
                profile=profile,
                max_pages=max_pages,
                sections_backend=encoder if same else None,  # one embedding model, not two
            )
            analyse = reader
        selector = _fit_all(todo, labels, fields, tagger_obj, encoder, analyse, on_progress)
    finally:
        if reader is not None:
            reader.close()
        for item in owned:
            close_backend(item)

    from docrooter.tagger.element_tags import tagger_fingerprint

    meta = selector.meta
    meta.update(
        {
            "profile": profile,
            "max_pages": max_pages,
            "labeler": LABELER,
            "tagger": {
                "spec": _spec_of(tagger),
                "fingerprint": tagger_fingerprint(tagger_obj),
                "source": getattr(tagger_obj, "source", None),
            },
            "corpus_manifest_sha256": hashlib.sha256(corpus.manifest_path.read_bytes()).hexdigest(),
            "fields": {family: [f.id for f in fs] for family, fs in sorted(fields.items())},
            "fields_using_description": sorted(
                f.id for fs in fields.values() for f in fs if not f.query
            ),
        }
    )
    meta["documents"]["skipped_no_fields"] = dict(sorted(skipped.items()))
    # only the fields of a family that ended with a dial were calibrated; select() guarantees
    # recall for these (id and fingerprint) and for nothing else
    calibrated = calibration_of(
        [f for family, fs in fields.items() if family in selector.dials for f in fs]
    )
    return Selector(
        field=selector.field,
        field_fallback=selector.field_fallback,
        query=selector.query,
        query_fallback=selector.query_fallback,
        dials=selector.dials,
        embedder=_spec_of(embedder),
        meta=meta,
        calibrated_fields=calibrated,
    )


def _fit_all(
    todo: list[tuple[ManifestRecord, str, str]],
    labels: Mapping[str, Mapping[str, Mapping[str, str]]],
    fields: Mapping[str, list[Field]],
    tagger: Any,
    encoder: Any,
    analyse: Analyse,
    on_progress: Progress | None,
) -> Selector:
    """Score every document (calib-fit first), fit the combiners, then the dials."""
    field_rows, query_rows = _Rows(), _Rows()
    crc_scored: list[tuple[str, str, Field, np.ndarray, np.ndarray]] = []
    checks: dict[str, dict[str, Any]] = {}
    counts: dict[str, Counter[str]] = {"fit": Counter(), "crc": Counter()}
    section_types: Counter[str] = Counter()  # the doc type each one's sections were read as
    for done, (record, family, half) in enumerate(todo, 1):
        analysis = analyse(record)
        sections = getattr(analysis, "sections", None)
        section_types[str(getattr(sections, "doc_type", None) or "none")] += 1
        ctx = prepare(analysis, tagger=tagger, encoder=encoder, family=family)
        if not (ctx.tagged and ctx.has_sims):
            raise DocrooterError(
                f"{record.id}: the full features need every content element tagged and "
                "embedded; the tagger left some untagged"
            )
        leaves, checks[record.id] = label_leaves(ctx, labels.get(record.id, {}), record.id)
        counts[half][family] += 1
        for field in fields[family]:
            topics = set(field.topics)
            y = np.array([leaf in topics for leaf in leaves], dtype=np.float64)
            _, X = need_features(ctx, field)
            if half == "fit":
                field_rows.add(X, y, record.id)
                _, Xq = need_features(ctx, _query_of(field))
                query_rows.add(Xq, y, record.id)
            else:
                crc_scored.append((record.id, family, field, X, y))
        if on_progress is not None:
            on_progress(done, len(todo), record.id, half)

    fit_warnings: list[str] = []
    field_combiner = _fit("field", Combiner(), field_rows, fit_warnings)
    field_fallback = _fit(
        "field_fallback", Combiner(feature_names=FALLBACK_FEATURES), field_rows, fit_warnings
    )
    query = _fit("query", Combiner(), query_rows, fit_warnings)
    query_fallback = _fit(
        "query_fallback", Combiner(feature_names=FALLBACK_FEATURES), query_rows, fit_warnings
    )
    units = [
        CalibUnit(
            doc_id=doc_id,
            family=family,
            field_id=f.id,
            probs=field_combiner.predict_proba(X) if len(y) else np.zeros(0),
            relevant=y.astype(bool),
        )
        for doc_id, family, f, X, y in crc_scored
    ]
    dials = calibrate_dial(units, combiner=field_combiner)
    for family in sorted(set(counts["crc"]) - set(dials)):
        fit_warnings.append(f"no dial for {family}: no calib-crc unit has a relevant element")

    meta: dict[str, Any] = {
        "documents": {
            "fit": dict(sorted(counts["fit"].items())),
            "crc": dict(sorted(counts["crc"].items())),
        },
        "label_mismatch": {
            "content": sum(c["content"] for c in checks.values()),
            "no_label": sum(c["no_label"] for c in checks.values()),
            "missing": sum(c["missing"] for c in checks.values()),
            "hash_mismatch": sum(c["hash_mismatch"] for c in checks.values()),
            "per_document": checks,
        },
        "class_balance": {
            "field": field_rows.balance(),
            "query": query_rows.balance(),
            "note": "rows are not reweighted; the combiners fit the calib-fit class balance",
        },
        "dials": {
            family: {"n": d.n, "n_docs": d.n_docs, "n_excluded": d.n_excluded}
            for family, d in sorted(dials.items())
        },
        "section_doc_types": dict(sorted(section_types.items())),
        "warnings": fit_warnings,
    }
    return Selector(
        field=field_combiner,
        field_fallback=field_fallback,
        query=query,
        query_fallback=query_fallback,
        dials=dials,
        meta=meta,
    )
