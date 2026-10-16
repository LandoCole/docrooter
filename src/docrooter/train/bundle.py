"""``TrainedModel``: a shared embedding body plus per-node heads, saved as a bundle directory.

Bundle layout::

    manifest.json
    body/                      SentenceTransformer.save() of the (fine-tuned) body
    heads/<sha1(node)[:12]>.joblib
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Candidate, Scores, close_backend
from docrooter.errors import BackendError

if TYPE_CHECKING:
    from docrooter.backends.embedding import EmbeddingBackend

MANIFEST = "manifest.json"
BODY_DIR = "body"
# Recipe fields that describe how to load a model from the hub, not how to talk to it.
_LOAD_FIELDS = ("revision", "trust_remote_code")


def head_file(node_id: str) -> str:
    """Bundle-relative path of a node's head."""
    return f"heads/{hashlib.sha1(node_id.encode('utf-8')).hexdigest()[:12]}.joblib"


class TrainedModel:
    """Per-node logistic-regression heads over a shared embedding body (``softmax`` scores).

    ``supports(node)`` is True only for nodes with a head; the classifier sends other nodes to
    its fallback, by default :meth:`as_zero_shot` (the same body used as an embedding model).
    Texts are embedded with the query prompt and instruction used at training time, whatever
    ``instruction`` the caller passes to :meth:`score`.
    """

    max_candidates: int | None = None
    temperature: float = 1.0

    def __init__(
        self,
        manifest: dict[str, Any],
        heads: dict[str, Any],
        body: EmbeddingBackend,
        *,
        path: Path | None = None,
    ) -> None:
        self.manifest = manifest
        self.name = f"trained:{manifest['base_model']}"
        self._heads = dict(heads)  # node -> fitted head, filled lazily for a loaded bundle
        self._body = body
        self._path = path
        self._parent: dict[str, str] = {
            child: node for node, info in manifest["heads"].items() for child in info["classes"]
        }
        recipe = manifest["recipe"]
        self._prompt: str = recipe["query_prompt"].format(instruction=manifest["instruction"])

    @property
    def path(self) -> Path | None:
        """Where the bundle was saved or loaded from (absolute), else ``None``."""
        return self._path

    @property
    def _device(self) -> str | None:
        """The body's requested device (``None``: chosen by ``select_device`` on first use).

        Reports look for this attribute to tell that a backend runs on a device.
        """
        return getattr(self._body, "_device", None)

    @property
    def taxonomy_hash(self) -> str:
        """Content hash of the taxonomy the model was trained on."""
        return self.manifest["taxonomy_hash"]

    # --- Backend protocol -----------------------------------------------------------------

    def supports(self, node_id: str) -> bool:
        return node_id in self.manifest["heads"]

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        ids = [c.node_id for c in candidates]
        node_id = self._parent.get(ids[0]) if ids else None
        classes = self.manifest["heads"][node_id]["classes"] if node_id is not None else []
        if node_id is None or len(set(ids)) != len(ids) or set(ids) != set(classes):
            where = "no head covers them"
            if node_id is not None:
                where = f"node {node_id!r} has a head for {sorted(classes)}"
            raise BackendError(
                f"{self.name}: candidates {ids} do not match a trained head ({where})"
            )
        if not texts:
            return Scores(np.zeros((0, len(ids))), "softmax")
        head = self._head(node_id)
        column = {str(c): i for i, c in enumerate(head.classes_)}
        probs = head.predict_proba(self._body.encode(list(texts), prompt=self._prompt))
        return Scores(np.asarray(probs, dtype=float)[:, [column[i] for i in ids]], "softmax")

    def tokenizer(self) -> Callable[[str], int] | None:
        return self._body.tokenizer()

    def as_zero_shot(self) -> EmbeddingBackend:
        """The body as a zero-shot embedding backend (base recipe prompts and temperature).

        The same instance every time, sharing this model's weights; nothing loads until used.
        """
        return self._body

    def close(self) -> None:
        """Release the body (also the :meth:`as_zero_shot` backend) and reloadable heads.

        A saved or loaded model reloads both from its bundle on its next use. A body fine-tuned
        in memory and never saved cannot be reloaded: using the model after ``close`` then
        raises ``BackendError``, so save it first if you still need it.
        """
        if self._path is not None:
            self._heads.clear()  # reloaded from the bundle on demand
        elif self.manifest.get("train_metrics", {}).get("triplets"):
            self._body._unrecoverable = (
                f"{self.name} was fine-tuned in memory and closed before it was saved; "
                "its fine-tuned weights are gone (save the model before closing it)"
            )
        close_backend(self._body)

    def _head(self, node_id: str) -> Any:
        if node_id not in self._heads:
            if self._path is None:
                raise BackendError(f"{self.name}: no head was fitted for node {node_id!r}")
            joblib = require("joblib", "models")
            self._heads[node_id] = joblib.load(self._path / self.manifest["heads"][node_id]["file"])
        return self._heads[node_id]

    # --- persistence ----------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write the bundle: ``manifest.json``, ``body/`` and ``heads/*.joblib``."""
        joblib = require("joblib", "models")
        out = Path(path).resolve()
        (out / "heads").mkdir(parents=True, exist_ok=True)
        self._body._load().save(str(out / BODY_DIR))
        for node_id, info in self.manifest["heads"].items():
            joblib.dump(self._head(node_id), out / info["file"])
        (out / MANIFEST).write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._path = out
        self._reload_body_from(out / BODY_DIR)

    def _reload_body_from(self, body_dir: Path) -> None:
        """Make the body reload from ``body_dir`` after :meth:`close`, not from the hub (where
        a fine-tuned body's weights do not exist). The weights in memory are unchanged."""
        from docrooter.backends.embedding import EmbeddingBackend

        if isinstance(self._body, EmbeddingBackend):
            self._body.model_id = str(body_dir)
            self._body.recipe = self._body.recipe.model_copy(update={"revision": None})

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | None = None, revision: str | None = None
    ) -> TrainedModel:
        """Open a bundle written by :meth:`save`; heads and body load on first use.

        Everything comes from the bundle, so it works offline. The body runs remote code only
        when the base model's recipe required it (the bundle is local and was written by
        :meth:`save`).
        """
        from docrooter.backends.embedding import EmbeddingBackend

        if revision is not None:
            raise BackendError(f"trained bundles have no revisions (got @{revision})")
        root = Path(path).resolve()
        try:
            manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise BackendError(f"no trained bundle at {root}: {MANIFEST} not found") from None
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError(f"cannot read {root / MANIFEST}: {exc}") from exc
        recipe = dict(manifest["recipe"])
        trust = recipe.pop("trust_remote_code")
        for field in _LOAD_FIELDS:
            recipe.pop(field, None)
        body = EmbeddingBackend(
            str(root / BODY_DIR), device=device, trust_remote_code=True if trust else None, **recipe
        )
        return cls(manifest, {}, body, path=root)
