"""``process()`` and ``process_many()``: resolve a configuration into engines and backends, run
them on documents, and return :class:`~docrooter.rooter.analysis.Analysis` objects (spec §7).

The order of a run: the configuration decisions (``route`` events), loading (element mode
unless ``layout="none"``), document type, its escalation, sections (with the taxonomy of the
final document type), their escalation, then summaries and, last, the element embeddings.

Ruling (spec §7.5): document-type escalation runs before the sections are normalised, so when it
changes the type, the sections are normalised once, with the new type's taxonomy, by the primary
``sections`` backend. That is the spec's "re-normalised with the new taxonomy", without the
discarded first pass.

Ruling (E9): an escalation backend that raises (a network error, a rejected key, a missing
extra) never aborts the analysis. The primary result is kept and the route log gets a
``warning`` decision that names the backend and the error's type, never its message (which may
carry a key or the request body).
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from docrooter import events
from docrooter.backends import registry as backend_registry
from docrooter.backends.base import Backend, close_backend
from docrooter.classify.classifier import Classifier, _is_trained
from docrooter.classify.documents import classify_document
from docrooter.classify.normalize import (
    DOC_TYPES,
    GENERIC_TAXONOMY,
    NormalizationResult,
    _section_taxonomy_name,
    default_view,
    label_sections,
    normalize_sections,
    section_classifier,
)
from docrooter.classify.results import DocTypeResult
from docrooter.classify.strategies import Greedy
from docrooter.document.load import load
from docrooter.document.models import Document
from docrooter.elements.models import Box
from docrooter.embed import embed, require_encoder
from docrooter.errors import DocrooterError
from docrooter.layout import registry as layout_registry
from docrooter.layout.base import LayoutEngine
from docrooter.ocr import registry as ocr_registry
from docrooter.ocr.base import OcrEngine, OcrLine
from docrooter.rooter.analysis import Analysis
from docrooter.rooter.config import RouterConfig
from docrooter.summarize import Summary, summarize
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy

if TYPE_CHECKING:
    from docrooter.pdf.routing import RouteDecision

__all__ = ["process", "process_many"]

_log = logging.getLogger(__name__)

# Tasks that also take an object instead of a name: the caller owns it (never closed here).
_INSTANCE_TASKS = ("ocr", "layout", "doc_type", "sections", "escalate")
OCR_UNAVAILABLE = "ocr_unavailable"


def process(
    source: str | os.PathLike[str],
    *,
    config: RouterConfig | str | os.PathLike[str] | None = None,
    profile: str | None = None,
    on_event: events.EventSink | None = None,
    **overrides: Any,
) -> Analysis:
    """Load, classify and summarise one document as the configuration says.

    The configuration is ``config`` (a :class:`RouterConfig`, or the path of a TOML/JSON
    file), ``profile`` and ``overrides`` (any :class:`RouterConfig` field), with
    ``overrides`` first, then the file, then the profile (``balanced`` by default). A
    ``RouterConfig`` cannot be combined with ``profile``; ``overrides`` still apply over it.

    ``ocr`` may also be an :class:`~docrooter.ocr.OcrEngine`, ``layout`` a
    :class:`~docrooter.layout.LayoutEngine`, ``doc_type`` and ``sections`` a backend or a
    :class:`~docrooter.Classifier`, and ``escalate`` a backend: the caller owns these and they
    are never closed here. What this call creates from names (engines and backends, each
    loaded at most once, shared between tasks with the same spec) is closed before it returns.

    Every configuration decision (OCR engine, layout, backends) is a document-scope
    ``RouteDecision``, emitted as a ``route`` event and, in element mode, put at the head of
    ``Document.meta["routes"]`` (the page decisions follow). Escalations are recorded there in
    every mode (spec §7.5). ``ocr="auto"`` with no engine installed records a warning decision
    and adds ``"ocr_unavailable"`` to ``meta["warnings"]`` of a PDF.

    ``embeddings`` (a backend spec such as ``embedding:BAAI/bge-small-en-v1.5``, needs element
    mode) puts one vector per element in ``Analysis.embeddings``; see :func:`~docrooter.embed`.
    An ``escalate`` backend that raises leaves the primary result in place and adds a warning
    to the route log (E9 ruling above).

    ``on_event`` receives the stages of :func:`~docrooter.load`,
    :func:`~docrooter.classify_document` and :func:`~docrooter.normalize_sections`, plus
    ``route``, ``escalate``, ``summarize`` and ``embed``, then one ``done``.
    """
    cfg, given = _resolve(config, profile, overrides)
    with events.pipeline_run(on_event):
        engines = _Engines(cfg, given)
        try:
            return _run(source, cfg, engines)
        finally:
            engines.close()  # before ``done``: released by the time a listener sees it


def process_many(
    sources: Iterable[str | os.PathLike[str]],
    *,
    config: RouterConfig | str | os.PathLike[str] | None = None,
    profile: str | None = None,
    on_event: events.EventSink | None = None,
    **overrides: Any,
) -> Iterator[Analysis]:
    """:func:`process` over ``sources``, yielding one :class:`Analysis` at a time.

    One OCR engine, one layout engine and one set of backends serve every document; they are
    created on first use and closed when the iterator is exhausted or closed (use
    ``contextlib.closing`` when stopping early). Each document is its own event run. The
    configuration is checked when this is called, before any document is read. ``sources`` is
    an iterable of paths or Markdown texts; a single path or text is a ``TypeError`` (a ``str``
    would be read as one source per character, and ``bytes`` as ints).
    """
    if isinstance(sources, (str, bytes, bytearray, os.PathLike)):
        raise TypeError(
            "process_many takes an iterable of sources, not a single path or text: pass a list, "
            "e.g. process_many([path]), or call process(path) for one document"
        )
    cfg, given = _resolve(config, profile, overrides)
    return _many(sources, cfg, given, on_event)


def _many(
    sources: Iterable[str | os.PathLike[str]],
    cfg: RouterConfig,
    given: dict[str, object],
    on_event: events.EventSink | None,
) -> Iterator[Analysis]:
    engines = _Engines(cfg, given)
    try:
        for source in sources:
            with events.pipeline_run(on_event):
                analysis = _run(source, cfg, engines)
            yield analysis
    finally:
        engines.close()


# --- configuration -------------------------------------------------------------------------


def _resolve(
    config: RouterConfig | str | os.PathLike[str] | None,
    profile: str | None,
    overrides: dict[str, Any],
) -> tuple[RouterConfig, dict[str, object]]:
    """The resolved configuration, and the engine or backend objects the caller passed."""
    plain: dict[str, Any] = {}
    given: dict[str, object] = {}
    for key, value in overrides.items():
        if key in _INSTANCE_TASKS and value is not None and not isinstance(value, str):
            given[key] = _check_instance(key, value)
            plain[key] = _instance_name(value)
        else:
            plain[key] = value
    if isinstance(config, RouterConfig):
        if profile is not None:
            raise TypeError("pass either a RouterConfig or a profile, not both")
        cfg = RouterConfig.model_validate({**config.model_dump(), **plain}) if plain else config
    else:
        cfg = RouterConfig.resolve(profile=profile, file=config, **plain)
    if cfg.embeddings != "none" and cfg.layout == "none":
        raise ValueError(
            f"embeddings={cfg.embeddings!r} needs element mode, and layout='none' (the fast "
            "profile) produces no elements: set layout='heuristic' or a layout engine name"
        )
    return cfg, given


def _check_instance(task: str, value: object) -> object:
    is_backend = callable(getattr(value, "score", None))  # the optional members may be absent
    if task == "ocr":
        ok = isinstance(value, OcrEngine)
    elif task == "layout":
        ok = isinstance(value, LayoutEngine)
    elif task == "escalate":
        ok = is_backend and not isinstance(value, Classifier)
    else:
        ok = is_backend or isinstance(value, Classifier)
    if not ok:
        kinds = {
            "ocr": "an engine name or an OcrEngine",
            "layout": "an engine name or a LayoutEngine",
            "escalate": "a backend spec or a Backend",
        }
        wanted = kinds.get(task, "a backend spec, a Backend or a Classifier")
        raise TypeError(f"{task} must be {wanted}, got {type(value).__name__}")
    return value


def _instance_name(value: object) -> str:
    """How a caller's object appears in the resolved configuration."""
    if isinstance(value, Classifier):
        backend = value.backend
        return backend if isinstance(backend, str) else _instance_name(backend)
    return str(getattr(value, "name", None) or type(value).__name__)


def _decision(task: str, engine: str, reason: str, **metrics: float) -> RouteDecision:
    from docrooter.pdf.routing import RouteDecision  # the PDF modules load on first use

    return RouteDecision(
        scope="document", target=None, task=task, engine=engine, reason=reason, metrics=metrics
    )


def _emit(decisions: list[RouteDecision]) -> None:
    """Report decisions as one ``route`` stage with a progress event each."""
    if not decisions:
        return
    stage = events.start("route", scope="document")
    if stage.active:
        for decision in decisions:
            stage.progress(**decision.model_dump())
    stage.end(decisions=len(decisions))


class _LazyOcr:
    """The registered OCR engine ``name``, created on the first page that needs OCR: creating
    one imports its package (about a second for ocrmac), and born-digital PDFs rarely need it.
    ``close`` closes the engine if it was created."""

    def __init__(self, name: str, lang: str) -> None:
        self.name = name
        self._lang = lang
        self._engine: OcrEngine | None = None

    def recognize(self, image: bytes, *, width_px: int, height_px: int, page: Box) -> list[OcrLine]:
        if self._engine is None:
            self._engine = ocr_registry.get_ocr_engine(self.name, lang=self._lang)
        return self._engine.recognize(image, width_px=width_px, height_px=height_px, page=page)

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()


class _Engines:
    """The engines and backends of a run: each created from its name on first use, at most
    once, and closed by :meth:`close`. Objects the caller passed are used as they are and
    never closed."""

    def __init__(self, cfg: RouterConfig, given: dict[str, object]) -> None:
        self.cfg = cfg
        self.given = given
        self._owned: list[object] = []
        self._ocr: tuple[OcrEngine | None, RouteDecision] | None = None
        self._layout: tuple[str | LayoutEngine, RouteDecision] | None = None
        self._backends: dict[str, Backend] = {}
        self._doc_types: Taxonomy | None = None
        self.ocr_unavailable = False

    def ocr(self) -> tuple[OcrEngine | None, RouteDecision]:
        if self._ocr is None:
            self._ocr = self._make_ocr()
        return self._ocr[0], self._ocr[1].model_copy(deep=True)

    def _make_ocr(self) -> tuple[OcrEngine | None, RouteDecision]:
        given = self.given.get("ocr")
        if given is not None:
            engine = given  # checked to be an OcrEngine
            assert isinstance(engine, OcrEngine)
            return engine, _decision("ocr", self.cfg.ocr, "engine passed by the caller")
        name = self.cfg.ocr
        if name == "none":
            return None, _decision("ocr", "none", "ocr=none: pages that need OCR keep their text")
        reason = f"ocr={name}"
        if name == "auto":
            resolved = ocr_registry.resolve_auto_ocr()
            if resolved is None:
                self.ocr_unavailable = True
                return None, _decision(
                    "ocr",
                    "none",
                    "warning: ocr=auto and no OCR engine is installed (pip install "
                    "'docrooter[ocr]', 'docrooter[ocr-mac]' on macOS, or the tesseract "
                    "binary); pages that need OCR keep their text layer",
                )
            name = resolved
            reason = "ocr=auto: the first installed engine (tesseract, ocrmac on macOS, rapidocr)"
        known = ocr_registry.available_ocr_engines()
        if name not in known:  # checked now; the engine itself is created on first use
            raise DocrooterError(f"unknown OCR engine {name!r}; available: {', '.join(known)}")
        engine = _LazyOcr(name, self.cfg.ocr_lang)
        self._owned.append(engine)
        return engine, _decision("ocr", name, reason)

    def layout(self) -> tuple[str | LayoutEngine, RouteDecision]:
        if self._layout is None:
            self._layout = self._make_layout()
        return self._layout[0], self._layout[1].model_copy(deep=True)

    def _make_layout(self) -> tuple[str | LayoutEngine, RouteDecision]:
        given = self.given.get("layout")
        if given is not None:
            assert isinstance(given, LayoutEngine)
            return given, _decision("layout", self.cfg.layout, "engine passed by the caller")
        name = self.cfg.layout
        if name == "heuristic":
            return name, _decision("layout", name, "layout=heuristic: the element rules")
        # A missing extra raises here (spec §6: never fall back silently).
        engine = layout_registry.get_layout_engine(name)
        self._owned.append(engine)
        return engine, _decision("layout", name, f"layout={name}: regions type the lines")

    def backend(self, task: str) -> Backend | Classifier:
        given = self.given.get(task)
        if given is not None:
            return cast("Backend | Classifier", given)
        spec = str(getattr(self.cfg, task))
        if spec not in self._backends:
            backend = backend_registry.resolve_backend(spec)
            self._backends[spec] = backend
            self._owned.append(backend)
        return self._backends[spec]

    def doc_types(self) -> Taxonomy:
        if self._doc_types is None:
            self._doc_types = builtin_taxonomy(DOC_TYPES)
        return self._doc_types

    def close(self) -> None:
        owned, self._owned = self._owned, []
        self._ocr = self._layout = None
        self._backends = {}
        for item in owned:
            try:
                close_backend(item)  # calls close() when there is one
            except Exception:  # one failing close must not keep the others open
                _log.warning("closing %s failed", type(item).__name__, exc_info=True)


# --- one document --------------------------------------------------------------------------


def _is_pdf(source: str | os.PathLike[str]) -> bool:
    if isinstance(source, str) and "\n" in source:
        return False  # Markdown text
    return Path(source).suffix.lower() == ".pdf"


def _run(source: str | os.PathLike[str], cfg: RouterConfig, engines: _Engines) -> Analysis:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    is_pdf = _is_pdf(source)
    element_mode = cfg.layout != "none"

    decisions: list[RouteDecision] = []
    options: dict[str, Any] = {}
    if is_pdf:
        if cfg.text != "auto":
            options["engine"] = cfg.text
        if cfg.max_pages is not None:
            options["max_pages"] = cfg.max_pages
        if element_mode:
            ocr, ocr_decision = engines.ocr()
            layout, layout_decision = engines.layout()
            decisions += [_text_decision(cfg.text), ocr_decision, layout_decision]
            options.update(ocr=ocr, layout=layout, ocr_dpi=cfg.ocr_dpi, tables=cfg.tables)
    for task in ("doc_type", "sections", "escalate", "embeddings"):
        spec = getattr(cfg, task)
        if spec != "none":
            how = "passed by the caller" if task in engines.given else f"{task}={spec}"
            decisions.append(_decision(task, spec, how))
    if cfg.embeddings != "none":
        require_encoder(engines.backend("embeddings"))  # before the document is read
    _emit(decisions)

    clock = time.perf_counter()
    doc = load(source, elements=element_mode, **options)
    timings["load"] = time.perf_counter() - clock
    if element_mode:
        doc.meta["routes"] = [d.model_dump() for d in decisions] + doc.meta.get("routes", [])
        if is_pdf and engines.ocr_unavailable:
            doc.meta.setdefault("warnings", []).append(OCR_UNAVAILABLE)

    escalations: list[RouteDecision] = []
    escalation_failed = False  # a backend that raised is not asked again for this document
    doc_type: DocTypeResult | None = None
    primary_type: DocTypeResult | None = None
    if cfg.doc_type != "none":
        clock = time.perf_counter()
        doc_type = primary_type = _classify(doc, engines.backend("doc_type"), engines.doc_types())
        timings["doc_type"] = time.perf_counter() - clock
        if _escalates(cfg, doc_type.confidence) and doc_type.best.stopped_reason != "empty_text":
            clock = time.perf_counter()
            stage = events.start("escalate", task="doc_type", engine=cfg.escalate)
            try:
                escalated = _classify(doc, engines.backend("escalate"), engines.doc_types())
            except Exception as exc:  # the primary result stands (E9 ruling)
                stage.end(error=type(exc).__name__)
                escalations.append(_escalation_failed(cfg, "doc_type", exc))
                escalation_failed = True
            else:
                stage.end(type=_leaf(escalated), confidence=round(escalated.confidence, 4))
                escalations += _doc_type_escalation(cfg, doc_type, escalated)
                doc_type = escalated  # kept whenever escalation fires (spec §7.5)
            timings["escalate"] = time.perf_counter() - clock

    sections: NormalizationResult | None = None
    if cfg.sections != "none":
        clock = time.perf_counter()
        sections = _normalize(doc, engines.backend("sections"), _leaf(doc_type))
        timings["sections"] = time.perf_counter() - clock
        if primary_type is not doc_type:
            before = _normalize_taxonomy(_leaf(primary_type), engines.doc_types())
            if before != sections.taxonomy:
                escalations.append(
                    _decision(
                        "sections",
                        cfg.sections,
                        f"taxonomy {sections.taxonomy} of the escalated document type "
                        f"(the primary type's was {before}): every section normalised with it",
                    )
                )
        if cfg.escalate != "none" and not escalation_failed:
            clock = time.perf_counter()
            sections, relabelled = _escalate_sections(doc, sections, cfg, engines)
            if relabelled:
                escalations.append(relabelled)
                spent = time.perf_counter() - clock
                timings["escalate"] = timings.get("escalate", 0.0) + spent

    if escalations:
        _emit(escalations)
        doc.meta["routes"] = doc.meta.get("routes", []) + [d.model_dump() for d in escalations]

    summaries: dict[str, Summary] = {}
    if cfg.summaries != "none":
        clock = time.perf_counter()
        stage = events.start("summarize", method=cfg.summaries)
        summaries = summarize(doc, method=cfg.summaries)
        stage.end(summaries=len(summaries))
        timings["summaries"] = time.perf_counter() - clock

    analysis = Analysis(
        document=doc, doc_type=doc_type, sections=sections, summaries=summaries, config=cfg
    )
    if cfg.embeddings != "none":
        clock = time.perf_counter()
        stage = events.start("embed", unit="element", engine=cfg.embeddings)
        encoder = cast("Backend", engines.backend("embeddings"))  # checked to encode
        analysis.embeddings = embed(analysis, encoder, unit="element")
        stage.end(vectors=len(analysis.embeddings.ids), dim=analysis.embeddings.dim)
        timings["embeddings"] = time.perf_counter() - clock
    timings["total"] = time.perf_counter() - started
    analysis.timings = {key: round(value, 4) for key, value in timings.items()}
    return analysis


def _text_decision(text: str) -> RouteDecision:
    if text != "auto":
        return _decision("text", text, f"text={text}")
    from docrooter.pdf.load import _auto_engine

    return _decision("text", _auto_engine(), "text=auto: pymupdf if installed, else pdfplumber")


def _leaf(result: DocTypeResult | None) -> str | None:
    return result.best.node_ids[-1] if result is not None and result.best.node_ids else None


def _classify(doc: Document, backend: Backend | Classifier, doc_types: Taxonomy) -> DocTypeResult:
    """:func:`classify_document` with ``backend`` (``Greedy`` for a trained model, which
    cannot score ``Flat``); a caller's ``Classifier`` classifies as it is configured."""
    if isinstance(backend, Classifier):
        return backend.classify_documents([doc], threshold=0.0)[0]
    strategy = Greedy() if _is_trained(backend) else None
    return classify_document(doc, doc_types, backend=backend, strategy=strategy)


def _normalize(
    doc: Document, backend: Backend | Classifier, doc_type: str | None
) -> NormalizationResult:
    """Sections normalised with the taxonomy of ``doc_type``; the generic section taxonomy when
    there is no type (as the playground does), never a second document-type pass."""
    taxonomy = None if doc_type else GENERIC_TAXONOMY
    if isinstance(backend, Classifier):
        return normalize_sections(doc, doc_type=doc_type, taxonomy=taxonomy, classifier=backend)
    return normalize_sections(doc, doc_type=doc_type, taxonomy=taxonomy, backend=backend)


def _normalize_taxonomy(doc_type: str | None, doc_types: Taxonomy) -> str:
    """The section taxonomy :func:`_normalize` uses for ``doc_type``."""
    return _section_taxonomy_name(doc_type, doc_types) if doc_type else GENERIC_TAXONOMY


def _escalates(cfg: RouterConfig, confidence: float) -> bool:
    return cfg.escalate != "none" and confidence < cfg.escalate_below


def _doc_type_escalation(
    cfg: RouterConfig, primary: DocTypeResult, escalated: DocTypeResult
) -> list[RouteDecision]:
    """Both results, and why the second one is kept."""
    return [
        _decision(
            "doc_type",
            cfg.doc_type,
            f"confidence {primary.confidence:.2f} below escalate_below {cfg.escalate_below}: "
            f"{_leaf(primary) or 'no type'} escalated",
            confidence=round(primary.confidence, 4),
        ),
        _decision(
            "doc_type",
            cfg.escalate,
            f"escalated result kept: {_leaf(escalated) or 'no type'} (confidences of different "
            "backends are not compared)",
            confidence=round(escalated.confidence, 4),
        ),
    ]


def _escalation_failed(cfg: RouterConfig, task: str, error: Exception) -> RouteDecision:
    """The warning for an escalation backend that raised: its name and the error's type only.
    The message is left out on purpose, since it can hold a key or the body of the request."""
    return _decision(
        task,
        cfg.escalate,
        f"warning: escalation backend {cfg.escalate} raised {type(error).__name__}: "
        "primary result kept",
    )


def _escalate_sections(
    doc: Document, result: NormalizationResult, cfg: RouterConfig, engines: _Engines
) -> tuple[NormalizationResult, RouteDecision | None]:
    """Re-label, in one batch with the escalation backend, the labelled sections whose score
    is below ``escalate_below``; their new labels are kept."""
    low = [
        sid
        for sid, norm in result.sections.items()
        if norm.canonical is not None and norm.score < cfg.escalate_below
    ]
    if not low:
        return result, None
    stage = events.start("escalate", task="sections", engine=cfg.escalate, sections=len(low))
    try:
        backend = engines.backend("escalate")
        assert not isinstance(backend, Classifier)  # escalate takes a backend only
        taxonomy = builtin_taxonomy(result.taxonomy)  # the rooter only uses built-in taxonomies
        view = default_view(taxonomy)
        with section_classifier(
            taxonomy, backend, result.doc_type, view, engines.doc_types()
        ) as classifier:
            relabelled = label_sections(doc, [doc.sections[sid] for sid in low], classifier, view)
    except Exception as exc:  # the primary labels stand (E9 ruling)
        stage.end(error=type(exc).__name__)
        return result, _escalation_failed(cfg, "sections", exc)
    changed = sum(
        1 for sid, norm in relabelled.items() if norm.canonical != result.sections[sid].canonical
    )
    stage.end(sections=len(relabelled), changed=changed)
    updated = result.model_copy(update={"sections": {**result.sections, **relabelled}})
    decision = _decision(
        "sections",
        cfg.escalate,
        f"{len(low)} section(s) scored below escalate_below {cfg.escalate_below}: re-labelled "
        f"by the escalation backend ({changed} changed)",
        sections=float(len(low)),
        changed=float(changed),
    )
    return updated, decision
