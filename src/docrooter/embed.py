"""Embed a document's elements or chunks for retrieval (spec §9).

:func:`embed` runs the texts through an :class:`~docrooter.backends.embedding.EmbeddingBackend`
(``encode``, the one model path the classifier uses too) and returns an :class:`EmbeddingSet`:
the ids, the vectors, and the model and revision that made them, so that queries can be embedded
with the same model later.

Each text is the heading path joined with `` > `` and the element's text (see
:mod:`docrooter.chunking`), encoded with the empty prompt. That is the document side of the
recipes' asymmetric prompts: ``Recipe.query_prompt`` ("Instruct: …\\nQuery: " for UltraBERT, the
search instruction for bge) is for the question, and ``Recipe.label_prompt`` is for taxonomy
labels. Embed a query with ``backend.encode([question], prompt=backend.recipe.query_prompt.format(
instruction=...))`` and rank by dot product: the vectors are L2-normalised.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from docrooter.backends.base import Backend, close_backend
from docrooter.backends.registry import resolve_backend
from docrooter.chunking import SKIPPED_KINDS, chunk, compose, document_of, element_parts
from docrooter.document.models import Document
from docrooter.elements.models import ElementKind
from docrooter.errors import BackendError, DocrooterError
from docrooter.summarize import heading_titles

if TYPE_CHECKING:
    from docrooter.rooter.analysis import Analysis

__all__ = ["EmbeddingSet", "embed", "require_encoder"]

Dtype = Literal["float16", "float32"]
DOCUMENT_PROMPT = ""  # documents are encoded bare; only queries and labels carry a prompt
_NPZ_FORMAT = 1
_ELEMENT_KINDS = frozenset(get_args(ElementKind))


class EmbeddingSet(BaseModel):
    """Vectors for ``ids`` (element ids, or chunk ids), row by row.

    ``model`` and ``revision`` are the backend's model id and the revision it loaded (``None``
    when unpinned), ``dim`` the vector length and ``dtype`` the storage type of ``vectors``,
    a ``(len(ids), dim)`` array of L2-normalised rows (float16 loses about three digits).
    JSON holds the vectors as nested lists; :meth:`to_npz` is the compact file form.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str
    revision: str | None
    dim: int
    dtype: Dtype
    ids: list[str]
    vectors: np.ndarray

    @field_validator("vectors", mode="before")
    @classmethod
    def _as_array(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value, dtype=info.data.get("dtype", "float32"))

    @model_validator(mode="after")
    def _consistent(self) -> EmbeddingSet:
        if self.vectors.dtype != np.dtype(self.dtype):
            self.vectors = self.vectors.astype(self.dtype)
        if self.vectors.ndim != 2 or self.vectors.shape[0] != len(self.ids):
            raise ValueError(
                f"vectors must have one row per id: {len(self.ids)} ids, "
                f"vectors of shape {self.vectors.shape}"
            )
        if self.vectors.shape[1] != self.dim:
            raise ValueError(f"dim is {self.dim} but the vectors have {self.vectors.shape[1]}")
        return self

    @field_serializer("vectors")
    def _vectors_as_lists(self, vectors: np.ndarray) -> list[list[float]]:
        return vectors.tolist()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EmbeddingSet):
            return NotImplemented
        return (
            (self.model, self.revision, self.dim, self.dtype, self.ids)
            == (other.model, other.revision, other.dim, other.dtype, other.ids)
            and self.vectors.dtype == other.vectors.dtype
            and np.array_equal(self.vectors, other.vectors)
        )

    def to_npz(self, path: str | os.PathLike[str]) -> None:
        """Write the set to ``path`` as an ``.npz`` (read back without pickle by
        :meth:`from_npz`)."""
        meta = {
            "format": _NPZ_FORMAT,
            "model": self.model,
            "revision": self.revision,
            "dim": self.dim,
            "dtype": self.dtype,
        }
        with open(path, "wb") as handle:  # a file object: numpy would add ".npz" to a bare name
            np.savez(
                handle,
                vectors=self.vectors,
                ids=np.array(self.ids, dtype=str),
                meta=np.array(json.dumps(meta)),
            )

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> EmbeddingSet:
        """Read a set written by :meth:`to_npz`."""
        with np.load(path, allow_pickle=False) as data:
            if not {"vectors", "ids", "meta"} <= set(data.files):
                raise DocrooterError(f"{path} is not an EmbeddingSet file (see to_npz)")
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != _NPZ_FORMAT:
                raise DocrooterError(f"{path} has EmbeddingSet format {meta.get('format')!r}")
            return cls(
                model=meta["model"],
                revision=meta["revision"],
                dim=meta["dim"],
                dtype=meta["dtype"],
                ids=[str(i) for i in data["ids"]],
                vectors=data["vectors"],
            )


def embed(
    source: Analysis | Document,
    backend: str | Backend,
    *,
    unit: Literal["element", "chunk"] = "element",
    dtype: Dtype = "float16",
    kinds: Iterable[str] | None = None,
) -> EmbeddingSet:
    """Embed the elements (or the chunks) of ``source`` with an embedding ``backend``.

    ``backend`` is a spec such as ``"embedding:BAAI/bge-small-en-v1.5"`` (created here and closed
    afterwards; needs the ``models`` extra) or a backend with an ``encode`` method, which the
    caller keeps open. ``unit="element"`` embeds each element except page furniture and
    ``contents``, in reading order, under its element id; ``kinds`` replaces that default with
    exactly the element kinds named (furniture included). ``unit="chunk"`` embeds
    ``chunk(source)`` under the chunk ids. A heading is embedded as its heading path.
    """
    if unit not in ("element", "chunk"):
        raise ValueError(f"unit must be 'element' or 'chunk', got {unit!r}")
    if dtype not in ("float16", "float32"):
        raise ValueError(f"dtype must be 'float16' or 'float32', got {dtype!r}")
    wanted = _check_kinds(kinds)
    if unit == "chunk" and wanted is not None:
        raise ValueError("kinds selects elements: it needs unit='element'")

    doc = document_of(source)
    if unit == "chunk":
        chunks = chunk(doc)
        ids, texts = [c.id for c in chunks], [c.text for c in chunks]
    else:
        keep = (
            (lambda el: el.kind in wanted)
            if wanted is not None
            else (lambda el: el.kind not in SKIPPED_KINDS)
        )
        parts = element_parts(doc, keep)
        paths: dict[str | None, list[str]] = {}
        for part in parts:
            section_id = part.element.section_id
            if section_id not in paths:
                paths[section_id] = heading_titles(doc, section_id)
        ids = [p.element.id for p in parts]
        texts = [compose(paths[p.element.section_id], p.body) for p in parts]

    owned = isinstance(backend, str)
    encoder = resolve_backend(backend)  # a backend instance passes through
    try:
        require_encoder(encoder)
        vectors = _encode(encoder, texts)
        recipe = getattr(encoder, "recipe", None)
        return EmbeddingSet(
            model=str(
                getattr(encoder, "model_id", None)
                or getattr(encoder, "name", None)
                or type(encoder).__name__
            ),
            revision=getattr(recipe, "revision", None),
            dim=vectors.shape[1],
            dtype=dtype,
            ids=ids,
            vectors=vectors.astype(dtype),
        )
    finally:
        if owned:
            close_backend(encoder)


def require_encoder(backend: object) -> None:
    """Raise :class:`~docrooter.errors.BackendError` unless ``backend`` can ``encode`` texts."""
    if not callable(getattr(backend, "encode", None)):
        name = getattr(backend, "name", None) or type(backend).__name__
        raise BackendError(
            f"{name} cannot embed: it has no encode(); use an embedding backend such as "
            "'embedding:BAAI/bge-small-en-v1.5'"
        )


def _check_kinds(kinds: Iterable[str] | None) -> frozenset[str] | None:
    if kinds is None:
        return None
    wanted = frozenset([kinds] if isinstance(kinds, str) else kinds)
    unknown = sorted(wanted - _ELEMENT_KINDS)
    if unknown:
        raise ValueError(
            f"unknown element kind {unknown[0]!r}; valid kinds: {', '.join(sorted(_ELEMENT_KINDS))}"
        )
    return wanted


def _encode(backend: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = np.asarray(backend.encode(texts, prompt=DOCUMENT_PROMPT))
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise BackendError(
            f"{getattr(backend, 'name', type(backend).__name__)}.encode returned an array of "
            f"shape {vectors.shape} for {len(texts)} texts (expected one row per text)"
        )
    return vectors
