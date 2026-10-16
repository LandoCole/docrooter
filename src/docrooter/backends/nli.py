"""NLI backend: zero-shot classification by textual entailment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Candidate, Scores, _row_softmax
from docrooter.backends.loading import select_device
from docrooter.backends.recipes import Recipe, get_recipe
from docrooter.errors import BackendError

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


def _label_indices(label2id: dict[str, int], model_id: str) -> tuple[int, int | None]:
    """Index of the entailment logit and, if the model has one, of the explicit "other" logit.

    "Other" is a contradiction or not-entailment label. Models with neither (for example a
    three-way model without a contradiction label) return ``None``; the caller then pools the
    remaining logits.
    """
    entail = other = None
    for key, index in label2id.items():
        name = key.lower()
        negated = name.startswith(("not", "non")) and "entail" in name
        if "entail" in name and not negated:
            entail = index
        elif negated or "contra" in name:
            other = index if other is None or "contra" in name else other
    if entail is None:
        raise BackendError(
            f"{model_id} has no entailment label in its config (labels: {sorted(label2id)}); "
            "it is not an NLI model"
        )
    return entail, other


class NLIBackend:
    """Scores ``text`` against each candidate as premise and "This text is about <label>."

    Single-label scoring (``softmax``) takes a row softmax of the entailment logits over the
    candidates; with ``multi_label=True`` each pair gets its own softmax over [entailment,
    other] (``independent``). The model loads on the first call to ``score`` or ``tokenizer``.
    """

    max_candidates: int | None = None

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        hypothesis_template: str = "This text is about {}.",
        revision: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        if "{}" not in hypothesis_template:
            raise BackendError(
                f"hypothesis_template needs a '{{}}' placeholder: {hypothesis_template!r}"
            )
        self.model_id = model
        self.name = f"nli:{model}"
        self.recipe: Recipe = get_recipe(model, revision=revision, batch_size=batch_size)
        self.temperature = self.recipe.temperature
        self.hypothesis_template = hypothesis_template
        self._device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._device_in_use = "cpu"
        self._entail = 0
        self._other: int | None = None

    def _load(self) -> Any:
        if self._model is None:
            transformers = require("transformers", "models")
            device = select_device(self._device)
            # No attn_implementation: DeBERTa-v2 has its own attention and rejects "sdpa".
            kwargs: dict[str, Any] = {"revision": self.recipe.revision}
            if device.split(":")[0] != "cuda":
                kwargs["dtype"] = "float32"  # transformers 5 would otherwise keep the checkpoint's
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, revision=self.recipe.revision
            )
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **kwargs
            )
            self._entail, self._other = _label_indices(dict(model.config.label2id), self.model_id)
            model.to(device)
            model.eval()
            self._tokenizer, self._model, self._device_in_use = tokenizer, model, device
        return self._model

    def _pair_logits(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        """Entailment-vs-other logits, shape ``(n_pairs, 2)``, batched."""
        model = self._load()
        torch = require("torch", "models")
        rows: list[np.ndarray] = []
        for start in range(0, len(premises), self.recipe.batch_size):
            stop = start + self.recipe.batch_size
            enc = self._tokenizer(
                premises[start:stop],
                hypotheses[start:stop],
                truncation="only_first",
                max_length=self.recipe.max_seq_len,
                padding=True,
                return_tensors="pt",
            ).to(self._device_in_use)
            with torch.inference_mode():
                logits = model(**enc).logits
            rows.append(np.asarray(logits.float().cpu(), dtype=np.float64))
        logits = np.concatenate(rows)
        entail = logits[:, self._entail]
        if self._other is not None:
            other = logits[:, self._other]
        else:
            rest = np.delete(logits, self._entail, axis=1)
            peak = rest.max(axis=1)
            other = peak + np.log(np.exp(rest - peak[:, None]).sum(axis=1))
        return np.stack([entail, other], axis=1)

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        kind = "independent" if multi_label else "softmax"
        n_texts, n_cands = len(texts), len(candidates)
        if not n_texts or not n_cands:
            return Scores(np.zeros((n_texts, n_cands)), kind)
        hypotheses = [self.hypothesis_template.format(c.text) for c in candidates]
        premises = [t for t in texts for _ in candidates]
        pairs = self._pair_logits(premises, hypotheses * n_texts)
        if multi_label:
            probs = _row_softmax(pairs)[:, 0]
        else:
            probs = _row_softmax(pairs[:, 0].reshape(n_texts, n_cands)).reshape(-1)
        return Scores(probs.reshape(n_texts, n_cands), kind)

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        self._load()
        tok = self._tokenizer
        return lambda text: len(tok.encode(text, add_special_tokens=False))
