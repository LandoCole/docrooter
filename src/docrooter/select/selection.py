"""``select()``: which elements a downstream LLM should read for a need, at a completeness
(spec §6, §7).

**Needs.** A :class:`~docrooter.select.fields.Field`, a list of fields of one family (read as
one need: the union of their topics), or a free-text query.

**Scoring.** Every content element (:data:`~docrooter.select.relevance.CONTENT_KINDS`) gets
P(relevant) from a :class:`~docrooter.select.relevance.Combiner` over the spec §5 features. The
:class:`Selector` holds four combiners: the full field and query combiners, and their fallbacks
on ``("sim", "section_hit")`` alone. The full one is used when every content element has topic
tags (from the analysis, or from ``tagger``) and an embedder gives ``sim``; otherwise the
fallback is. The elements' tags are completed by ``tagger`` when it is given; the analysis is
never modified.

**The guarantee.** ``guarantee=True`` only when the need is fields of one family, the full field
combiner was used, the selector has a dial for that family, the document's type maps to that
family (spec §0.6), there is no budget, and every field of the need is one the dial was
calibrated on (:attr:`Selector.calibrated_fields`: the field's id and a fingerprint of its
topics and functions; a selector that records none, as in the old format, calibrated none).
A field outside it gets an estimate with ``uncalibrated_field=True``. The dial's λ̂(c) is
then the threshold and ``expected_recall`` is c: E[recall] ≥ c on documents like the calibration
set. A dial capped by its calibration size returns the whole document (``capped=True``, recall
trivially 1).

**Without the guarantee** (a query, the no-tagger or no-embedder fallback, no dial for the
family, a doc-type mismatch, an uncalibrated field) the cut is the model's own estimate: the
shortest prefix by p, ties kept together, whose share of sum(p) reaches c.
``expected_recall`` is that share, an estimate, and ``threshold`` the p of the last element taken.

**Doc-type mismatch.** When the analysis's doc type maps to another family than the field's
(no doc type maps to ``generic``, as in :func:`~docrooter.process`), the content elements are
re-tagged within the field's family (with ``tagger``; without one, the fallback is used), and
``doc_type_mismatch=True`` drops the guarantee: the dial was calibrated on correctly typed
documents.

**Budget.** With ``budget`` (tokens), elements are taken by p, each with its expansion, while
the total stays within the budget; one that does not fit is skipped and smaller ones are tried.
If the most relevant element alone exceeds the budget, it is returned alone with
``over_budget=True``; if it fits alone but nothing fits with its expansion, it is returned
alone too. ``expected_recall`` is the estimated share of sum(p) selected.

**Nothing relevant.** ``relevant_expected`` is sum(p) over the content elements. Below 0.5 the
document is taken to hold nothing relevant. On the guarantee path nothing changes: the dial's
threshold is applied (it usually keeps nothing), ``expected_recall`` is still c (what is
promised), and a capped dial still returns the whole document with ``expected_recall=1.0``.
Without the guarantee, ``expected_recall`` is ``None`` (undefined, not 0 or 1) and the
estimate and budget cuts select nothing, never the whole document.

**Expansion** (spec §6), after the cut: each selected element brings the element just before it
when that is in the same section, its section's heading and first paragraph, and, for a table
or a caption, its caption or table. Then every table or caption brought in brings its partner,
so tables always arrive whole with their captions. Headings are read and counted, though never
scored.

**Features.** :func:`prepare` reads one document once (its index, tags, section labels and
element embeddings) into a :class:`DocContext`, and :func:`need_features` gives the spec §5
feature matrix for a need from it. :func:`select` scores exactly these features, and fitting
(``docrooter.select.fit``) must build its training rows with the same two functions, so the
combiners and dials see at select time what they were fitted on.

**Tokens and chunks.** ``tokens`` is :func:`~docrooter.views.estimate_tokens` over the selected
elements, and ``token_share`` its share of the content and heading tokens of the document.
``chunks`` is :func:`~docrooter.chunking.chunk` over each contiguous run of selected elements
(page furniture between them does not break a run), numbered in reading order.

**c = 1.0** returns every content element and heading, with ``guarantee=True``, and needs no
selector, tagger or embedder.

Everything is linear in the document: each element is scored, expanded and counted once, with
per-section lookups precomputed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel

from docrooter.backends.base import close_backend
from docrooter.chunking import Chunk, chunk, compose, document_of, element_parts
from docrooter.document.models import Document
from docrooter.elements.models import Element, TagResult
from docrooter.errors import DocrooterError
from docrooter.select.crc import DialTable
from docrooter.select.fields import Field, field_fingerprint
from docrooter.select.relevance import CONTENT_KINDS, FEATURE_NAMES, Combiner, features
from docrooter.summarize import heading_titles
from docrooter.taxonomies import family_for_doc_type
from docrooter.views import estimate_tokens

if TYPE_CHECKING:
    from docrooter.rooter.analysis import Analysis

__all__ = [
    "FALLBACK_FEATURES",
    "NEED_INSTRUCTION",
    "SELECTOR_FILE",
    "DocContext",
    "Selection",
    "Selector",
    "need_features",
    "need_texts",
    "prepare",
    "readout",
    "select",
    "select_scored",
    "similarities",
]

SELECTOR_FILE = "selector.json"
SELECTOR_FORMAT = "docrooter-selector"
SELECTOR_VERSION = 1
FALLBACK_FEATURES: tuple[str, ...] = ("sim", "section_hit")
"""The features of the fallback combiners (spec §5: no tagger)."""
NEED_INSTRUCTION = "Given a need, retrieve the passages of a document that serve it"
"""The instruction of the need side of ``sim``, in the embedder recipe's query prompt."""
NO_RELEVANT_BELOW = 0.5  # sum(p) under which the document is taken to hold nothing relevant
HEADING_KINDS = frozenset({"heading", "title"})
_TOL = 1e-12
_ROLES = ("field", "field_fallback", "query", "query_fallback")


class Selection(BaseModel):
    """What to read for a need (module docstring).

    ``need`` is the field id (ids joined with ``,`` for a list) or the query. ``threshold`` is
    the relevance cut (``None`` for c = 1.0, a capped dial or a budget). ``expected_recall`` is
    the CRC target when ``guarantee`` (1.0 when capped), else the model's estimate, and
    ``None`` without the guarantee when no relevant content is expected (``relevant_expected``
    < 0.5). ``uncalibrated_field`` is ``True`` when everything else would have carried the
    guarantee but a field of the need is not one the dial was calibrated on (it is then an
    estimate, ``guarantee=False``). ``relevant_expected`` is the expected
    number of relevant elements, sum(p) (``None`` at c = 1.0, where nothing is scored).
    """

    need: str
    completeness: float
    threshold: float | None
    element_ids: list[str]  # in reading order
    tokens: int
    token_share: float  # tokens / the document's content and heading tokens
    expected_recall: float | None
    guarantee: bool
    chunks: list[Chunk]
    capped: bool = False
    doc_type_mismatch: bool = False
    relevant_expected: float | None = None
    over_budget: bool = False
    uncalibrated_field: bool = False


def readout(selection: Selection) -> str:
    """The one-line readout of a selection, e.g. "≈ 90% of relevant content kept · 3,240 tokens
    (12%) · guaranteed on calibration-like documents". The CLI and the playground both print
    this, so their wording cannot drift."""
    parts: list[str] = []
    if selection.expected_recall is None:
        parts.append("no relevant content expected")
    else:
        kept = f"≈ {selection.expected_recall:.0%} of relevant content kept"
        parts.append(kept if selection.guarantee else f"{kept} (estimate)")
    parts.append(f"{selection.tokens:,} tokens ({selection.token_share:.0%})")
    if selection.capped:
        parts.append("dial capped by calibration size")
    elif selection.guarantee:
        parts.append("guaranteed on calibration-like documents")
    if selection.doc_type_mismatch:
        parts.append("doc type mismatch")
    if selection.uncalibrated_field:
        parts.append("field not calibrated (estimate)")
    if selection.over_budget:
        parts.append("the most relevant element alone exceeds the budget")
    return " · ".join(parts)


# --- the selector ------------------------------------------------------------------------------


class Selector:
    """The fitted relevance combiners and the completeness dials, saved as ``selector.json``.

    ``field`` scores fields on every feature, and its ``dials`` (one per element family) were
    calibrated on its probabilities; ``field_fallback`` and ``query_fallback`` use
    :data:`FALLBACK_FEATURES` only; ``query`` scores free-text queries. ``embedder`` is the
    spec of the embedding backend ``sim`` was fitted with (``None`` when not recorded):
    :func:`select` loads it when the caller passes no embedder. ``meta`` is what the fit
    recorded (``docrooter.select.fit``: profile, tagger, corpus, document counts, label checks,
    warnings); it does not affect selection.

    The attributes are read-only, so a dial cannot be paired with another combiner after the
    check in the constructor; :func:`select` checks the pairing again before it guarantees.
    """

    def __init__(
        self,
        *,
        field: Combiner,
        field_fallback: Combiner,
        query: Combiner,
        query_fallback: Combiner,
        dials: dict[str, DialTable],
        embedder: str | None = None,
        meta: dict[str, Any] | None = None,
        calibrated_fields: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        if tuple(field.feature_names) != FEATURE_NAMES:
            raise ValueError(
                f"the field combiner must use all the features {FEATURE_NAMES}, "
                f"got {field.feature_names}"
            )
        for name, combiner in (
            ("field_fallback", field_fallback),
            ("query_fallback", query_fallback),
        ):
            if tuple(combiner.feature_names) != FALLBACK_FEATURES:
                raise ValueError(
                    f"the {name} combiner must use {FALLBACK_FEATURES}, "
                    f"got {combiner.feature_names}"
                )
        for family, dial in dials.items():
            if dial.family != family:
                raise ValueError(f"the dial under {family!r} is for family {dial.family!r}")
            dial.check_combiner(field)
        self._field = field
        self._field_fallback = field_fallback
        self._query = query
        self._query_fallback = query_fallback
        self._dials = dict(dials)
        self._embedder = embedder
        self._meta = json.loads(json.dumps(meta or {}))  # a JSON-safe copy
        self._calibrated = _check_calibrated(calibrated_fields)

    @property
    def field(self) -> Combiner:
        return self._field

    @property
    def field_fallback(self) -> Combiner:
        return self._field_fallback

    @property
    def query(self) -> Combiner:
        return self._query

    @property
    def query_fallback(self) -> Combiner:
        return self._query_fallback

    @property
    def dials(self) -> dict[str, DialTable]:
        """Family -> its dial (a copy of the mapping)."""
        return dict(self._dials)

    @property
    def embedder(self) -> str | None:
        return self._embedder

    @property
    def calibrated_fields(self) -> dict[str, dict[str, str]]:
        """Family -> {field id: fingerprint} of the fields the dial was calibrated on (a copy;
        empty for a selector that recorded none, whose every field is then an estimate)."""
        return {family: dict(fields) for family, fields in self._calibrated.items()}

    def is_calibrated(self, field: Field) -> bool:
        """Whether the dial of ``field``'s family was calibrated on this very field: the same
        id and the same fingerprint of topics and functions."""
        known = self._calibrated.get(field.family, {})
        return known.get(field.id) == field_fingerprint(field)

    @property
    def meta(self) -> dict[str, Any]:
        """What the fit recorded (a copy)."""
        result: dict[str, Any] = json.loads(json.dumps(self._meta))
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": SELECTOR_FORMAT,
            "version": SELECTOR_VERSION,
            "embedder": self.embedder,
            "combiners": {role: getattr(self, role).to_dict() for role in _ROLES},
            "dials": {family: dial.to_dict() for family, dial in sorted(self.dials.items())},
            "calibrated_fields": {
                family: dict(sorted(fields.items()))
                for family, fields in sorted(self.calibrated_fields.items())
            },
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Selector:
        """Read :meth:`to_dict`'s form. Every dial is checked against the field combiner, so a
        refit without recalibrating raises ``ValueError``."""
        if data.get("format") != SELECTOR_FORMAT or data.get("version") != SELECTOR_VERSION:
            raise ValueError(
                f"not a {SELECTOR_FORMAT} v{SELECTOR_VERSION}: "
                f"{data.get('format')!r} v{data.get('version')!r}"
            )
        combiners = {role: Combiner.from_dict(data["combiners"][role]) for role in _ROLES}
        return cls(
            field=combiners["field"],
            field_fallback=combiners["field_fallback"],
            query=combiners["query"],
            query_fallback=combiners["query_fallback"],
            dials={family: DialTable.from_dict(d) for family, d in data["dials"].items()},
            embedder=data.get("embedder"),
            meta=data.get("meta") or {},
            calibrated_fields=data.get("calibrated_fields"),
        )

    def save(self, path: str | Path) -> Path:
        """Write ``selector.json``: to ``path``, or into it when it is a directory (such as a
        tagger bundle). Returns the file written."""
        target = _selector_file(Path(path))
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Selector:
        """Read ``selector.json``: the file ``path``, or the one in the directory ``path``."""
        target = _selector_file(Path(path))
        try:
            text = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise DocrooterError(
                f"no selector at {target}; make one with `docrooter fit-selector`"
            ) from None
        return cls.from_dict(json.loads(text))


def _check_calibrated(
    value: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    if value is None:
        return {}
    problem = ValueError("calibrated_fields must map each family to {field id: fingerprint}")
    if not isinstance(value, Mapping):
        raise problem
    out: dict[str, dict[str, str]] = {}
    for family, fields in value.items():
        if not isinstance(family, str) or not isinstance(fields, Mapping):
            raise problem
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in fields.items()):
            raise problem
        out[family] = dict(fields)
    return out


def _selector_file(path: Path) -> Path:
    return path / SELECTOR_FILE if path.is_dir() else path


# --- needs and similarity ----------------------------------------------------------------------


def _parse_need(need: Field | Sequence[Field] | str) -> tuple[list[Field], str | None]:
    """``(fields, None)`` for a field or a list of fields of one family, ``([], query)`` for a
    query; ``ValueError`` otherwise."""
    if isinstance(need, str):
        if not need.strip():
            raise ValueError("a query need must not be blank")
        return [], need
    fields = [need] if isinstance(need, Field) else list(need)
    if not fields or not all(isinstance(f, Field) for f in fields):
        raise ValueError("a need is a Field, a non-empty list of Fields, or a query string")
    families = sorted({f.family for f in fields})
    if len(families) > 1:
        raise ValueError(f"the fields of one need must share a family; got {families}")
    return fields, None


def need_texts(need: Field | Sequence[Field] | str) -> list[str]:
    """The need side of ``sim``: each field's ``query`` (else its ``description``), or the
    query itself."""
    fields, query = _parse_need(need)
    if query is not None:
        return [query]
    return [f.query or f.description for f in fields]


def _query_prompt(encoder: Any) -> str:
    """The recipe's query prompt with :data:`NEED_INSTRUCTION`, or bare without a recipe."""
    prompt = getattr(getattr(encoder, "recipe", None), "query_prompt", None)
    return prompt.format(instruction=NEED_INSTRUCTION) if isinstance(prompt, str) else ""


def _unit_rows(vectors: Any, rows: int) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != rows:
        raise DocrooterError(f"the embedder returned shape {array.shape} for {rows} texts")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms > 0.0, norms, 1.0)


def _text_vectors(texts: Sequence[str], encoder: Any) -> np.ndarray:
    """The texts encoded bare (as :func:`docrooter.embed` encodes documents), unit rows."""
    return _unit_rows(encoder.encode(list(texts), prompt=""), len(texts))


def _need_sims(docs: np.ndarray, need: Field | Sequence[Field] | str, encoder: Any) -> np.ndarray:
    """The cosine of each row of ``docs`` to the need text (encoded with the recipe's query
    prompt); for a list of fields, the highest over them."""
    if docs.shape[0] == 0:
        return np.zeros(0)
    wanted = need_texts(need)
    needs = _unit_rows(encoder.encode(wanted, prompt=_query_prompt(encoder)), len(wanted))
    if docs.shape[1] != needs.shape[1]:
        raise DocrooterError(
            f"the embedder gave {docs.shape[1]}-d texts but {needs.shape[1]}-d needs"
        )
    return np.asarray((docs @ needs.T).max(axis=1), dtype=np.float64)


def similarities(
    texts: Sequence[str], need: Field | Sequence[Field] | str, encoder: Any
) -> np.ndarray:
    """``sim`` per text: the cosine between the text (encoded bare, as :func:`docrooter.embed`
    encodes documents) and the need text (encoded with the recipe's query prompt); for a list
    of fields, the highest over them."""
    if not texts:
        return np.zeros(0)
    return _need_sims(_text_vectors(texts, encoder), need, encoder)


# --- the document index ------------------------------------------------------------------------


class _Index:
    """Positions (in ``doc.elements``) of what can be selected, and the per-element and
    per-section lookups expansion needs, built in one pass."""

    def __init__(self, doc: Document) -> None:
        self.doc = doc
        elements = doc.elements
        self.selectable = [
            i
            for i, el in enumerate(elements)
            if el.kind in CONTENT_KINDS or el.kind in HEADING_KINDS
        ]
        self.content = [i for i in self.selectable if elements[i].kind in CONTENT_KINDS]
        self.rank = {pos: r for r, pos in enumerate(self.selectable)}
        self.tokens = {pos: estimate_tokens(elements[pos].text) for pos in self.selectable}
        self.total_tokens = sum(self.tokens.values())
        self.heading: dict[str | None, int] = {}
        self.first_paragraph: dict[str | None, int] = {}
        self.previous: dict[int, int] = {}
        by_id = {elements[pos].id: pos for pos in self.selectable}
        self.partners: dict[int, list[int]] = {}
        last: int | None = None
        for pos in self.selectable:
            el = elements[pos]
            section = el.section_id
            if el.kind in HEADING_KINDS and section is not None:
                self.heading.setdefault(section, pos)
            if el.kind == "paragraph":
                self.first_paragraph.setdefault(section, pos)
            if last is not None and elements[last].section_id == section:
                self.previous[pos] = last
            last = pos
            for other_id in (el.caption_for, el.caption_id):
                other = by_id.get(other_id or "")
                if other is not None and "table" in (el.kind, elements[other].kind):
                    self._link(pos, other)

    def _link(self, a: int, b: int) -> None:
        for x, y in ((a, b), (b, a)):
            linked = self.partners.setdefault(x, [])
            if y not in linked:
                linked.append(y)

    def expansion(self, pos: int) -> list[int]:
        """``pos`` and what it brings (module docstring: expansion), closed under the
        table-caption pairing so that a table never arrives without its caption, or a caption
        without its table."""
        section = self.doc.elements[pos].section_id
        out = [pos]
        if pos in self.previous:
            out.append(self.previous[pos])
        if section in self.heading:
            out.append(self.heading[section])
        if section in self.first_paragraph:
            out.append(self.first_paragraph[section])
        for item in list(out):  # a partner's partner is the item itself: one step closes it
            out.extend(self.partners.get(item, ()))
        return out

    def expand(self, core: Sequence[int]) -> set[int]:
        return {p for pos in core for p in self.expansion(pos)}

    def section_labels(self, analysis: Analysis | Document) -> list[str | None]:
        """The normalised label of each content element's section: its own, else its nearest
        labelled ancestor's (a subsection of Risk Factors is still risk factors)."""
        norms = (
            analysis.sections.sections
            if not isinstance(analysis, Document) and analysis.sections is not None
            else {}
        )
        sections = self.doc.sections
        cache: dict[str, str | None] = {}

        def label_of(section_id: str | None) -> str | None:
            walked: list[str] = []
            found: str | None = None
            current = section_id
            while current is not None and current not in walked:
                if current in cache:
                    found = cache[current]
                    break
                norm = norms.get(current)
                if norm is not None and norm.canonical is not None:
                    found = norm.canonical
                    walked.append(current)
                    break
                walked.append(current)
                section = sections.get(current)
                current = section.parent_id if section is not None else None
            for seen in walked:
                cache[seen] = found
            return found

        return [label_of(self.doc.elements[pos].section_id) for pos in self.content]

    def sim_texts(self) -> tuple[list[int], list[str]]:
        """The content elements with text, by index into ``content``, and their heading path
        and body as :func:`docrooter.embed` composes them."""
        parts = element_parts(self.doc, lambda el: el.kind in CONTENT_KINDS)
        body = {part.element.id: part.body for part in parts}
        paths: dict[str | None, list[str]] = {}
        rows: list[int] = []
        texts: list[str] = []
        for j, pos in enumerate(self.content):
            el = self.doc.elements[pos]
            if el.id not in body:
                continue
            if el.section_id not in paths:
                paths[el.section_id] = heading_titles(self.doc, el.section_id)
            rows.append(j)
            texts.append(compose(paths[el.section_id], body[el.id]))
        return rows, texts


# --- features: one path for select() and for fitting ------------------------------------------


class DocContext:
    """One document read once for scoring many needs (:func:`prepare`): its index, the tags
    and section labels of its content elements, and their embeddings.

    ``element_ids`` are the content elements in reading order, the rows of
    :func:`need_features`. ``family`` is the element family the tags belong to, ``doc_family``
    the one the document's type maps to. ``tagged`` says whether every content element has a
    topic tag, and ``has_sims`` whether an encoder gave the element embeddings (else ``sim``
    is 0); the full combiners need both.
    """

    def __init__(
        self,
        *,
        index: _Index,
        family: str,
        doc_family: str,
        tags: list[dict[str, TagResult] | None],
        tagged: bool,
        sections: list[str | None],
        vectors: np.ndarray | None,
        rows: list[int],
        encoder: Any,
    ) -> None:
        self.index = index
        self.family = family
        self.doc_family = doc_family
        self.tags = tags
        self.tagged = tagged
        self.sections = sections
        self.vectors = vectors  # unit rows, one per entry of ``rows``
        self.rows = rows  # indices into the content elements that have text
        self.encoder = encoder
        self.element_ids = [index.doc.elements[pos].id for pos in index.content]

    @property
    def has_sims(self) -> bool:
        return self.vectors is not None


def prepare(
    analysis: Analysis | Document,
    *,
    tagger: Any = None,
    encoder: Any = None,
    family: str | None = None,
) -> DocContext:
    """Read ``analysis`` once for :func:`need_features`.

    ``family`` is the element family the tags must belong to; by default, the one the
    analysis's doc type maps to (``generic`` without one). The analysis's own tags are used
    when every content element has one and ``family`` is the document's family. Otherwise
    ``tagger`` (an ``element_tags`` spec, loaded and closed here, or an object with
    ``tag(elements, family)``) tags the content elements within ``family``; without a tagger,
    tags of another family are dropped. ``encoder`` (an object with ``encode``, which the
    caller keeps open until its last :func:`need_features` call) embeds each content element's
    heading path and body once. The analysis is never modified.
    """
    doc = document_of(analysis)
    index = _Index(doc)
    doc_family = family_for_doc_type(_doc_type(analysis) or "")
    family = family or doc_family
    tags, tagged = _content_tags(doc, index, tagger, retag=family != doc_family, family=family)
    vectors: np.ndarray | None = None
    rows: list[int] = []
    if encoder is not None:
        rows, texts = index.sim_texts()
        vectors = _text_vectors(texts, encoder) if texts else np.zeros((0, 0))
    return DocContext(
        index=index,
        family=family,
        doc_family=doc_family,
        tags=tags,
        tagged=tagged,
        sections=index.section_labels(analysis),
        vectors=vectors,
        rows=rows,
        encoder=encoder,
    )


def need_features(
    ctx: DocContext, need: Field | Sequence[Field] | str
) -> tuple[list[str], np.ndarray]:
    """``(content element ids, the full spec §5 feature matrix)`` for ``need`` on a prepared
    document, one row per content element in reading order. ``sim`` is 0 when the context has
    no embeddings (or the element has no text)."""
    fields, _ = _parse_need(need)
    index = ctx.index
    sims = np.zeros(len(index.content))
    if ctx.vectors is not None and ctx.rows:
        sims[ctx.rows] = _need_sims(ctx.vectors, need, ctx.encoder)
    X = features(
        [index.doc.elements[pos] for pos in index.content],
        ctx.tags,
        sims,
        ctx.sections,
        fields if fields else need,
    )
    return list(ctx.element_ids), X


# --- select ------------------------------------------------------------------------------------


def select(
    analysis: Analysis | Document,
    need: Field | Sequence[Field] | str,
    *,
    completeness: float = 0.9,
    budget: int | None = None,
    selector: Selector | str | Path | None = None,
    tagger: Any = None,
    embedder: Any = None,
) -> Selection:
    """The elements of ``analysis`` to read for ``need`` at ``completeness`` (module docstring).

    ``selector`` is a :class:`Selector` or the path of its ``selector.json`` (or of a directory
    holding one); by default it is looked for in the tagger's bundle directory. There is no
    built-in selector: below c = 1.0 without one, ``ValueError`` says how to fit one.

    ``tagger`` (an ``element_tags`` spec such as a bundle directory, or an object with
    ``tag(elements, family)``) tags the content elements that have no tags, and re-tags them on
    a doc-type mismatch. ``embedder`` (a backend with ``encode``, or a spec such as
    ``"embedding:BAAI/bge-small-en-v1.5"``) gives ``sim``; by default the selector's recorded
    embedder is loaded. **Without an embedder, sim is unknown:** the fallback combiner is used
    (on ``section_hit`` alone) and there is no guarantee. A tagger or embedder named by a spec
    is loaded here and closed afterwards.

    ``budget`` (tokens, at least 1) replaces the dial and never carries the guarantee; it
    cannot be combined with c = 1.0, which always means the whole document.
    """
    c = _check_target(completeness, budget)
    fields, query = _parse_need(need)
    label = query if query is not None else ",".join(f.id for f in fields)
    doc_family = family_for_doc_type(_doc_type(analysis) or "")
    field_family = fields[0].family if fields else None
    mismatch = field_family is not None and field_family != doc_family

    if c == 1.0:
        index = _Index(document_of(analysis))
        return _selection(
            index,
            set(index.selectable),
            label,
            c,
            threshold=None,
            expected=1.0,
            guarantee=True,
            mismatch=mismatch,
        )

    encoder: Any = None
    encoder_owned = False
    try:
        chosen = _resolve_selector(selector, tagger)
        spec = embedder if embedder is not None else chosen.embedder
        if spec is not None:
            encoder_owned = isinstance(spec, str)
            if encoder_owned:
                from docrooter.backends.registry import resolve_backend

                encoder = resolve_backend(spec)
            else:
                encoder = spec
        ctx = prepare(analysis, tagger=tagger, encoder=encoder, family=field_family or doc_family)
        _, X = need_features(ctx, need)
    finally:
        if encoder_owned and encoder is not None:
            close_backend(encoder)
    return select_scored(analysis, ctx, X, need, completeness=c, budget=budget, selector=chosen)


def _check_target(completeness: float, budget: int | None) -> float:
    c = float(completeness)
    if not 0.0 <= c <= 1.0:  # also refuses NaN
        raise ValueError(f"completeness must be in [0, 1], got {completeness}")
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise ValueError(f"budget must be a whole number of tokens >= 1, got {budget!r}")
        if c == 1.0:
            raise ValueError(
                "budget cannot be combined with completeness=1.0, which always returns the "
                "whole document; lower the completeness or drop the budget"
            )
    return c


def select_scored(
    analysis: Analysis | Document,
    ctx: DocContext,
    X: np.ndarray,
    need: Field | Sequence[Field] | str,
    *,
    completeness: float,
    budget: int | None = None,
    selector: Selector,
) -> Selection:
    """The cut of :func:`select` on features already computed: ``ctx`` from :func:`prepare` and
    ``X`` from :func:`need_features` for the same ``need`` (the family of ``ctx`` must be the
    need's). Moving the completeness for a need then costs a re-threshold, with no tagging or
    embedding. Raises as :func:`select` does for a bad ``completeness`` or ``budget``."""
    c = _check_target(completeness, budget)
    if c == 1.0:
        return select(analysis, need, completeness=1.0)
    fields, query = _parse_need(need)
    label = query if query is not None else ",".join(f.id for f in fields)
    field_family = fields[0].family if fields else None
    mismatch = field_family is not None and field_family != ctx.doc_family
    chosen = selector
    index = ctx.index
    full = ctx.tagged and ctx.has_sims
    calibrated = all(chosen.is_calibrated(f) for f in fields)
    if fields:
        combiner = chosen.field if full else chosen.field_fallback
    else:
        combiner = chosen.query if full else chosen.query_fallback
    p = combiner.predict_proba(X) if index.content else np.zeros(0)
    relevant = float(p.sum())
    nothing = relevant < NO_RELEVANT_BELOW
    dial = chosen.dials.get(field_family) if field_family is not None else None
    eligible = bool(fields) and full and not mismatch and dial is not None and budget is None
    guaranteed = eligible and calibrated
    uncalibrated = eligible and not calibrated  # would be guaranteed, but this field was not fitted

    capped = over_budget = False
    threshold: float | None = None
    expected: float | None
    if budget is not None:
        picked, over_budget = _fill_budget(index, p, budget, nothing)
        guarantee = False
        expected = None if nothing else _share(index, p, picked, relevant)
    elif guaranteed:
        assert dial is not None
        dial.check_combiner(chosen.field)  # raises: never a guarantee from a stale dial
        mask, capped = dial.select(p, c)
        threshold, _ = dial.threshold(c)
        guarantee = True
        expected = 1.0 if capped else c  # what is promised, whatever sum(p) is
        picked = index.expand([index.content[j] for j in np.flatnonzero(mask)])
    else:
        guarantee = False
        core, threshold, expected = ([], None, None) if nothing else _estimate_cut(p, c)
        picked = index.expand([index.content[j] for j in core])
    return _selection(
        index,
        picked,
        label,
        c,
        threshold=threshold,
        expected=expected,
        guarantee=guarantee,
        mismatch=mismatch,
        capped=capped,
        relevant=relevant,
        over_budget=over_budget,
        uncalibrated=uncalibrated,
    )


def _doc_type(analysis: Analysis | Document) -> str | None:
    """The analysis's document type leaf (``None`` for a bare document or no doc type)."""
    if isinstance(analysis, Document):
        return None
    result = analysis.doc_type
    if result is not None and result.best.node_ids:
        return result.best.node_ids[-1]
    return analysis.sections.doc_type if analysis.sections is not None else None


def _resolve_selector(selector: Selector | str | Path | None, tagger: Any) -> Selector:
    if isinstance(selector, Selector):
        return selector
    if selector is not None:
        return Selector.load(selector)
    bundle = Path(tagger) if isinstance(tagger, (str, Path)) else getattr(tagger, "path", None)
    if bundle is not None and (Path(bundle) / SELECTOR_FILE).is_file():
        return Selector.load(Path(bundle) / SELECTOR_FILE)
    raise ValueError(
        "completeness below 1.0 needs a fitted selector, and docrooter has no built-in one: "
        "make one with `docrooter fit-selector` and pass selector=, or keep it as "
        f"{SELECTOR_FILE} in the tagger bundle"
    )


def _content_tags(
    doc: Document, index: _Index, tagger: Any, *, retag: bool, family: str
) -> tuple[list[dict[str, TagResult] | None], bool]:
    """The tags of each content element and whether all have a topic. The analysis's own tags
    are used unless some are missing or ``retag``; then ``tagger`` tags the elements within
    ``family`` (a tagger loaded here from a spec is closed here, whatever happens). Without a
    tagger on a re-tag, the old tags (of another family) are dropped."""
    elements = doc.elements
    existing: list[dict[str, TagResult] | None] = [elements[pos].tags for pos in index.content]
    complete = all(t is not None and "topic" in t for t in existing)
    if (complete and not retag) or not index.content:
        return existing, complete or not index.content
    if tagger is None:
        return ([None] * len(existing) if retag else existing), False
    from docrooter.tagger.element_tags import content_elements, resolve_tagger

    owned = isinstance(tagger, (str, Path))
    resolved = resolve_tagger(str(tagger)) if owned else tagger
    try:
        items = content_elements(doc)
        results = resolved.tag(items, family)
        if len(results) != len(items):
            raise DocrooterError(
                f"the element tagger returned {len(results)} results for {len(items)} elements"
            )
    finally:
        if owned:
            close_backend(resolved)
    by_id = {item.element_id: dict(result) for item, result in zip(items, results, strict=True)}
    tags = [by_id.get(elements[pos].id) for pos in index.content]
    return tags, all(t is not None and "topic" in t for t in tags)


def _estimate_cut(p: np.ndarray, c: float) -> tuple[list[int], float | None, float]:
    """The shortest prefix by p (ties kept together) whose share of sum(p) reaches ``c``:
    ``(content indices, the p of the last one taken, the share)``. Needs sum(p) > 0."""
    total = float(p.sum())
    if c <= 0.0:
        return [], None, 0.0
    order = np.argsort(-p, kind="stable")
    shares = np.cumsum(p[order]) / total
    last = min(int(np.searchsorted(shares, c - _TOL, side="left")), len(p) - 1)
    threshold = float(p[order[last]])
    core = np.flatnonzero(p >= threshold)  # the ties of the last one come along
    return core.tolist(), threshold, float(p[core].sum()) / total


def _fill_budget(index: _Index, p: np.ndarray, budget: int, nothing: bool) -> tuple[set[int], bool]:
    """Elements by p, each with its expansion, while the total stays within ``budget``:
    ``(positions, over_budget)`` (module docstring: budget)."""
    if nothing or len(p) == 0:
        return set(), False
    order = np.argsort(-p, kind="stable")
    best = index.content[int(order[0])]
    if index.tokens[best] > budget:  # R5: the best alone, flagged; never filler instead
        return {best}, True
    picked: set[int] = set()
    used = 0
    for j in order:
        if used >= budget:
            break
        unit = set(index.expansion(index.content[int(j)])) - picked
        cost = sum(index.tokens[pos] for pos in unit)
        if used + cost <= budget:
            picked |= unit
            used += cost
    if picked:
        return picked, False
    return {best}, False  # it fits alone, but nothing fits with its expansion


def _share(index: _Index, p: np.ndarray, picked: set[int], relevant: float) -> float:
    """The estimated recall of ``picked``: its share of sum(p) over the content elements."""
    if relevant <= 0.0:
        return 0.0
    return float(sum(p[j] for j, pos in enumerate(index.content) if pos in picked)) / relevant


def _selection(
    index: _Index,
    picked: set[int],
    label: str,
    c: float,
    *,
    threshold: float | None,
    expected: float | None,
    guarantee: bool,
    mismatch: bool,
    capped: bool = False,
    relevant: float | None = None,
    over_budget: bool = False,
    uncalibrated: bool = False,
) -> Selection:
    positions = sorted(picked)
    tokens = sum(index.tokens[pos] for pos in positions)
    total = index.total_tokens
    return Selection(
        need=label,
        completeness=c,
        threshold=threshold,
        element_ids=[index.doc.elements[pos].id for pos in positions],
        tokens=tokens,
        token_share=tokens / total if total else 0.0,
        expected_recall=expected,
        guarantee=guarantee,
        chunks=_chunks(index, positions),
        capped=capped,
        doc_type_mismatch=mismatch,
        relevant_expected=relevant,
        over_budget=over_budget,
        uncalibrated_field=uncalibrated,
    )


def _chunks(index: _Index, positions: list[int]) -> list[Chunk]:
    """:func:`~docrooter.chunking.chunk` over each run of selected elements that are next to
    each other among the selectable ones, renumbered in reading order."""
    runs: list[list[Element]] = []
    previous_rank = -2
    for pos in positions:
        rank = index.rank[pos]
        if rank != previous_rank + 1:
            runs.append([])
        runs[-1].append(index.doc.elements[pos])
        previous_rank = rank
    out: list[Chunk] = []
    for run in runs:
        for piece in chunk(index.doc.model_copy(update={"elements": run})):
            out.append(piece.model_copy(update={"id": f"c{len(out)}"}))
    return out
