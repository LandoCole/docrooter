"""``ElementTagger``: the student that tags every element by topic and function (spec §9, §10).

One shared embedding body serves both facets:

- **topic:** a :class:`~docrooter.train.bundle.TrainedModel` over the merged ``element_topics``
  tree (root -> family -> leaf), with partial heads: a leaf with too few confident examples is
  left out of its family's head and is only ever predicted at the family level (Review Focus
  4). The tree is scored with Beam(3); with a ``family`` the reported topic is the best leaf
  within it, and ``family_probs`` keeps the unconstrained family distribution (§4.0 ruling 2);
- **function:** one-vs-rest logistic regression per function tag on the same embeddings
  (:mod:`docrooter.tagger.functions`).

The student's input is :func:`docrooter.teach.dataset.input_text`, cut to ``max_chars``
characters, at training and at inference alike (:func:`student_text`), so both see the same
bytes; the body then truncates it to its ``max_seq_len`` tokens.

Bundle layout::

    tagger.json        format, temperatures, input limits, function heads, training report
    taxonomy.json      the topic tree
    topic/             TrainedModel.save() (manifest.json, body/, heads/)
    functions/<tag>.joblib

``tagger.json`` is written last, through a temporary file: a bundle without it is incomplete.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Scores, to_probs
from docrooter.corpus.store import CorpusElement
from docrooter.elements.models import TagResult
from docrooter.errors import BackendError
from docrooter.taxonomy import Taxonomy
from docrooter.teach.dataset import input_text
from docrooter.train.bundle import TrainedModel

__all__ = [
    "BEAM_WIDTH",
    "CHARS_PER_TOKEN",
    "FORMAT",
    "FUNCTION_THRESHOLD",
    "ElementTagger",
    "student_input",
    "student_text",
]

FORMAT = "docrooter-element-tagger/1"
TAGGER_MANIFEST = "tagger.json"
TAXONOMY_FILE = "taxonomy.json"
TOPIC_DIR = "topic"
FUNCTIONS_DIR = "functions"
BEAM_WIDTH = 3  # spec §4.0 ruling 2: the whole tree is scored with Beam(3)
FUNCTION_THRESHOLD = 0.5  # a function tag is in ``labels`` from this probability on
# Inputs are cut to this many characters per token of ``max_seq_len`` before tokenising, so a
# huge element is never tokenised whole. Ordinary text runs 3-6 characters a token, so the
# body's own token cut still decides what the model sees.
CHARS_PER_TOKEN = 16
_SAFE_TAG = re.compile(r"[A-Za-z0-9_.-]+")
_DIGITS = 6


def student_input(text: str, max_chars: int) -> str:
    """A built student input (an :func:`~docrooter.teach.dataset.input_text`) cut to
    ``max_chars`` characters: the one cut applied wherever the student sees text (training,
    function rows, calibration, tagging), so all of them see the same bytes."""
    return text[:max_chars]


def student_text(element: CorpusElement, max_chars: int) -> str:
    """The student's input for ``element``: :func:`~docrooter.teach.dataset.input_text` (heading
    path, kind, text; a table as its Markdown) cut by :func:`student_input`."""
    return student_input(input_text(element), max_chars)


def _temper(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax rows re-tempered as the classifier does (``backends.base.to_probs``)."""
    return to_probs(Scores(probs, "softmax"), temperature=temperature, multi_label=False)


def temper_binary(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Independent probabilities with their logits divided by ``temperature``."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    if temperature == 1:
        return p
    logits = np.log(p) - np.log1p(-p)
    return 0.5 * (1.0 + np.tanh(logits / (2.0 * temperature)))  # the sigmoid, overflow-free


def _p(value: float) -> float:
    return round(min(1.0, max(0.0, float(value))), _DIGITS)


class ElementTagger:
    """Topic and function tags for corpus elements (module docstring).

    ``temperatures`` holds one temperature per facet, ``{"topic": t, "function": t}`` (1.0
    until :func:`~docrooter.tagger.train.calibrate_tagger` sets them). ``info`` is the training
    report saved with the bundle.
    """

    def __init__(
        self,
        topic: TrainedModel,
        functions: dict[str, Any],
        *,
        taxonomy: Taxonomy,
        max_chars: int,
        temperatures: dict[str, float] | None = None,
        info: dict[str, Any] | None = None,
        path: Path | None = None,
    ) -> None:
        if topic.taxonomy_hash != taxonomy.content_hash():
            raise ValueError("the topic model was trained on another taxonomy than the one given")
        if max_chars < 1:
            raise ValueError(f"max_chars must be >= 1, got {max_chars}")
        self.topic = topic
        self.functions = dict(functions)
        self.taxonomy = taxonomy
        self.max_chars = max_chars
        self.temperatures = {"topic": 1.0, "function": 1.0, **(temperatures or {})}
        self.info = dict(info or {})
        self._path = path
        recipe = topic.manifest["recipe"]
        self._prompt: str = recipe["query_prompt"].format(instruction=topic.manifest["instruction"])
        self._families = [n.id for n in taxonomy.children()]

    # --- properties -----------------------------------------------------------------------

    @property
    def path(self) -> Path | None:
        """Where the bundle was saved or loaded from, else ``None``."""
        return self._path

    @property
    def source(self) -> str:
        """``TagResult.source``: ``bundle:<directory name>``, or ``element-tagger`` unsaved."""
        return f"bundle:{self._path.name}" if self._path is not None else "element-tagger"

    @property
    def untrained_families(self) -> list[str]:
        """Families the root head never saw (too few examples), in taxonomy order: an element
        of one is reported at the family level with its (near-zero) family probability. Empty
        when there is no root head."""
        root = self.topic.manifest["heads"].get(Taxonomy.ROOT_ID)
        if root is None:
            return []
        return [f for f in self._families if f not in root["classes"]]

    @property
    def untrained(self) -> list[str]:
        """Topic leaves no head predicts, in taxonomy order: each is only ever reported at the
        level of its family (Review Focus 4)."""
        covered = {c for info in self.topic.manifest["heads"].values() for c in info["classes"]}
        return [leaf.id for leaf in self.taxonomy.leaves() if leaf.id not in covered]

    # --- tagging --------------------------------------------------------------------------

    def tag(
        self,
        elements: Sequence[CorpusElement],
        family: str | None = None,
        *,
        batch_size: int = 256,
    ) -> list[dict[str, TagResult]]:
        """``{"topic": TagResult, "function": TagResult}`` per element (``function`` only when
        the tagger has function heads).

        **Topic.** With a ``family`` (the document's, from its doc type), the labels are the
        best leaf within that family and ``probs`` its family head's distribution over its
        trained leaves (P(leaf | family)); ``confidence`` is the best one's probability. A
        family without a head (fewer than two trained leaves, or one the root head never saw:
        :attr:`untrained_families`) is reported as the family itself, with the family's
        unconstrained probability as its probability and confidence. Without a ``family`` the
        tree is scored with Beam(3): ``probs`` holds the joint probability of each surviving
        path's end node and the best path's end node is the label. ``family_probs`` is always
        the unconstrained family distribution (0.0 for a family the root head never saw).

        **Function.** ``probs`` holds every tag with a head (calibrated), ``labels`` those at
        :data:`FUNCTION_THRESHOLD` or above in taxonomy order, ``confidence`` the least certain
        of the per-tag decisions.
        """
        if family is not None and family not in self._families:
            raise ValueError(f"unknown element family {family!r}; use one of {self._families}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if family is None and Taxonomy.ROOT_ID not in self.topic.manifest["heads"]:
            raise ValueError("this tagger has no family head; pass the document's family")
        out: list[dict[str, TagResult]] = []
        for start in range(0, len(elements), batch_size):
            batch = elements[start : start + batch_size]
            vectors = self.embed([student_text(e, self.max_chars) for e in batch])
            topics = self._topics(vectors, family)
            functions = self._functions(vectors)
            for i, topic in enumerate(topics):
                tags = {"topic": topic}
                if functions is not None:
                    tags["function"] = functions[i]
                out.append(tags)
        return out

    def embed(self, texts: list[str]) -> np.ndarray:
        """The body's embeddings of student inputs, with the prompt the heads were fitted on."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(self.topic.as_zero_shot().encode(texts, prompt=self._prompt))

    def _head_probs(self, node_id: str, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
        """``(classes, tempered probabilities)`` of a node's head."""
        head = self.topic._head(node_id)
        classes = [str(c) for c in head.classes_]
        probs = _temper(np.asarray(head.predict_proba(vectors), dtype=float), self._t("topic"))
        return classes, probs

    def _t(self, facet: str) -> float:
        return float(self.temperatures.get(facet, 1.0))

    def _family_rows(self, vectors: np.ndarray) -> list[dict[str, float]] | None:
        if Taxonomy.ROOT_ID not in self.topic.manifest["heads"]:
            return None
        classes, probs = self._head_probs(Taxonomy.ROOT_ID, vectors)
        rows = []
        for row in probs:
            got = dict(zip(classes, row.tolist(), strict=True))
            rows.append({f: got.get(f, 0.0) for f in self._families})
        return rows

    def _leaf_rows(self, family: str, vectors: np.ndarray) -> list[dict[str, float]] | None:
        """P(leaf | family) per row, in taxonomy order, or ``None`` without a family head."""
        if family not in self.topic.manifest["heads"]:
            return None
        classes, probs = self._head_probs(family, vectors)
        order = [c.id for c in self.taxonomy.children(family) if c.id in set(classes)]
        rows = []
        for row in probs:
            got = dict(zip(classes, row.tolist(), strict=True))
            rows.append({leaf: got[leaf] for leaf in order})
        return rows

    def _topics(self, vectors: np.ndarray, family: str | None) -> list[TagResult]:
        family_rows = self._family_rows(vectors)
        if family is not None:
            leaf_rows = self._leaf_rows(family, vectors)
            out = []
            for i in range(len(vectors)):
                fam = family_rows[i] if family_rows is not None else None
                if leaf_rows is None:
                    p = fam[family] if fam is not None else 1.0
                    probs = {family: p}
                else:
                    probs = leaf_rows[i]
                out.append(self._topic_result(probs, fam))
            return out
        assert family_rows is not None  # checked in tag()
        tables = {f: self._leaf_rows(f, vectors) for f in self._families}
        return [
            self._beam(family_rows[i], {f: t[i] if t else None for f, t in tables.items()})
            for i in range(len(vectors))
        ]

    def _beam(
        self, families: dict[str, float], leaves: dict[str, dict[str, float] | None]
    ) -> TagResult:
        """Beam(3) over root -> family -> leaf for one element: the top families, each expanded
        to its trained leaves (``leaves[family]``, P(leaf | family)), or kept as a path ending
        at the family when it has no head, then the top paths by joint probability. Ties keep
        the taxonomy order."""
        ranked = sorted(families.items(), key=lambda item: -item[1])[:BEAM_WIDTH]
        paths: list[tuple[str, float]] = []
        for fam, p in ranked:
            row = leaves[fam]
            if row is None:
                paths.append((fam, p))
            else:
                paths.extend((leaf, p * q) for leaf, q in row.items())
        paths = sorted(paths, key=lambda item: -item[1])[:BEAM_WIDTH]
        return self._topic_result(dict(paths), families)

    def _topic_result(
        self, probs: dict[str, float], family_probs: dict[str, float] | None
    ) -> TagResult:
        best = max(probs, key=lambda k: probs[k])  # first of equals: taxonomy order
        return TagResult(
            labels=[best],
            probs={k: _p(v) for k, v in probs.items()},
            confidence=_p(probs[best]),
            source=self.source,
            family_probs=(
                {k: _p(v) for k, v in family_probs.items()} if family_probs is not None else None
            ),
        )

    def raw_function_probs(self, vectors: np.ndarray) -> np.ndarray:
        """Untempered P(yes), one column per :attr:`function_tags` entry."""
        columns = []
        for tag in self.function_tags:
            head = self.functions[tag]
            positive = list(head.classes_).index(1)
            columns.append(np.asarray(head.predict_proba(vectors), dtype=float)[:, positive])
        return np.stack(columns, axis=1)

    def _functions(self, vectors: np.ndarray) -> list[TagResult] | None:
        if not self.functions:
            return None
        tags = self.function_tags
        probs = temper_binary(self.raw_function_probs(vectors), self._t("function"))
        out = []
        for row in probs:
            values = dict(zip(tags, row.tolist(), strict=True))
            out.append(
                TagResult(
                    labels=[t for t in tags if values[t] >= FUNCTION_THRESHOLD],
                    probs={t: _p(v) for t, v in values.items()},
                    confidence=_p(min(max(v, 1.0 - v) for v in values.values())),
                    source=self.source,
                )
            )
        return out

    @property
    def function_tags(self) -> list[str]:
        """The function tags with a head, in the order they were trained (taxonomy order)."""
        return list(self.functions)

    # --- lifecycle and persistence --------------------------------------------------------

    def close(self) -> None:
        """Release the body (it reloads from the bundle on the next use, if saved)."""
        self.topic.close()

    def __enter__(self) -> ElementTagger:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def save(self, path: str | Path) -> None:
        """Write the bundle (module docstring); ``tagger.json`` last, atomically.

        Over an existing bundle, everything the bundle manages (``tagger.json`` first, then the
        taxonomy, ``topic/`` and ``functions/``) is removed before writing, so no stale file
        survives and an interrupted save leaves no ``tagger.json``. The body and heads are read
        into memory first, so a tagger loaded from ``path`` can be saved back over it.
        """
        joblib = require("joblib", "models")
        for tag in self.functions:
            if not _SAFE_TAG.fullmatch(tag):
                raise ValueError(f"function tag {tag!r} is not a safe file name")
        out = Path(path).resolve()
        self.topic.as_zero_shot()._load()  # the weights, before their files are removed
        for node_id in self.topic.manifest["heads"]:
            self.topic._head(node_id)
        _clear(out)
        (out / FUNCTIONS_DIR).mkdir(parents=True, exist_ok=True)
        self.topic.save(out / TOPIC_DIR)
        files = {}
        for tag, head in self.functions.items():
            files[tag] = f"{FUNCTIONS_DIR}/{tag}.joblib"
            joblib.dump(head, out / files[tag])
        _write_json(out / TAXONOMY_FILE, self.taxonomy.to_dict())
        from docrooter import __version__

        manifest = {
            "format": FORMAT,
            "docrooter_version": __version__,
            "taxonomy_hash": self.taxonomy.content_hash(),
            "temperatures": self.temperatures,
            "max_chars": self.max_chars,
            "beam_width": BEAM_WIDTH,
            "function_threshold": FUNCTION_THRESHOLD,
            "functions": files,
            "info": self.info,
        }
        _write_json(out / TAGGER_MANIFEST, manifest)
        self._path = out

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> ElementTagger:
        """Open a bundle written by :meth:`save`; the body loads on first use."""
        joblib = require("joblib", "models")
        root = Path(path).resolve()
        try:
            manifest = json.loads((root / TAGGER_MANIFEST).read_text(encoding="utf-8"))
            taxonomy = Taxonomy.from_dict(
                json.loads((root / TAXONOMY_FILE).read_text(encoding="utf-8"))
            )
        except FileNotFoundError as exc:
            raise BackendError(
                f"no element-tagger bundle at {root}: {exc.filename} not found"
            ) from None
        except (OSError, ValueError) as exc:
            raise BackendError(f"cannot read the element-tagger bundle at {root}: {exc}") from exc
        if manifest.get("format") != FORMAT:
            raise BackendError(
                f"{root} holds format {manifest.get('format')!r}; this docrooter reads {FORMAT!r}"
            )
        topic = TrainedModel.load(root / TOPIC_DIR, device=device)
        functions = {tag: joblib.load(root / file) for tag, file in manifest["functions"].items()}
        return cls(
            topic,
            functions,
            taxonomy=taxonomy,
            max_chars=int(manifest["max_chars"]),
            temperatures={k: float(v) for k, v in manifest["temperatures"].items()},
            info=manifest.get("info") or {},
            path=root,
        )


def _clear(root: Path) -> None:
    """Remove the files a bundle manages under ``root`` (``tagger.json`` first); others stay."""
    for name in (TAGGER_MANIFEST, TAGGER_MANIFEST + ".tmp", TAXONOMY_FILE, TAXONOMY_FILE + ".tmp"):
        (root / name).unlink(missing_ok=True)
    for name in (TOPIC_DIR, FUNCTIONS_DIR):
        if (root / name).is_dir():
            shutil.rmtree(root / name)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
