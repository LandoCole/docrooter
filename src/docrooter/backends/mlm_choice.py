"""Masked-LM multiple-choice backend: the model fills in the letter of the right option."""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from docrooter._extras import require
from docrooter.backends.base import Candidate, Scores, _row_softmax
from docrooter.backends.loading import (
    after_model_load,
    attn_implementation,
    prepare_model_load,
    release_memory,
    run_with_cpu_fallback,
    trust_decision,
)
from docrooter.backends.recipes import Recipe, get_recipe
from docrooter.errors import BackendError

LETTERS: tuple[str, ...] = tuple(string.ascii_uppercase)

# Most tokens one forward batch may hold; MLM logits are (batch, length, vocab) floats.
_BATCH_TOKEN_BUDGET = 16384


def build_choice_prompt(
    text: str,
    options: list[str],
    *,
    instruction: str,
    answer: str,
    markers: Sequence[str] = LETTERS,
) -> str:
    """The multiple-choice prompt: instruction, text, lettered options, then ``Answer:``.

    ``answer`` goes after ``Answer: `` (the mask token for an MLM, ``""`` for an LLM, which
    then ends the prompt with ``Answer:`` and no trailing space).
    """
    if len(options) > len(markers):
        raise BackendError(f"{len(options)} options but only {len(markers)} markers")
    lines = [instruction, f"Text: {text}", "Options:"]
    lines += [f"{marker}. {option}" for marker, option in zip(markers, options, strict=False)]
    lines.append(f"Answer: {answer}" if answer else "Answer:")
    return "\n".join(lines)


class MLMChoiceBackend:
    """Zero-shot classification by asking a masked LM which lettered option fits.

    Experimental: the prompt is a fixed design choice that the benchmark validates. Scores are
    a softmax over the mask-position logits of the option letters (``softmax`` kind). The
    model cannot judge options independently, so ``multi_label=True`` is ignored and the
    result is still a softmax. At most 26 candidates fit; more raise ``BackendError`` (the
    classifier routes larger sets to a fallback before calling). The model loads on first use.
    """

    max_candidates: int | None = 26

    def __init__(
        self,
        model: str = "dmis-lab/UltraBERT-380M",
        *,
        revision: str | None = None,
        device: str | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.model_id = model
        self.name = f"mlm-choice:{model}"
        self.recipe: Recipe = get_recipe(model, revision=revision)
        # Letter probabilities are used as they are, not sharpened by an embedding temperature.
        self.temperature = 1.0
        self._device = device
        # A caller-chosen revision is not the pinned one the recipe vouches for.
        unpinned = revision is not None and revision != get_recipe(model).revision
        self.trust_remote_code = trust_decision(
            model,
            self.recipe.model_copy(update={"revision": None}) if unpinned else self.recipe,
            trust_remote_code,
        )
        self._tokenizer: Any = None
        self._model: Any = None
        self._device_in_use = "cpu"
        self._marker_ids: dict[str, int] = {}

    def _load(self) -> Any:
        if self._model is None:
            device = prepare_model_load(self._device)
            transformers = require("transformers", "models")
            kwargs: dict[str, Any] = {
                "revision": self.recipe.revision,
                "trust_remote_code": self.trust_remote_code,
                "attn_implementation": attn_implementation(device),
            }
            if device.split(":")[0] != "cuda":
                kwargs["dtype"] = "float32"
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.recipe.revision,
                trust_remote_code=self.trust_remote_code,
            )
            if not getattr(tokenizer, "mask_token", None):
                raise BackendError(f"{self.model_id} has no mask token; it is not a masked LM")
            model = transformers.AutoModelForMaskedLM.from_pretrained(self.model_id, **kwargs)
            self._device_in_use = device  # before the move, so an MPS OOM there is caught
            model.to(device)
            model.eval()
            after_model_load(device)
            self._tokenizer, self._model = tokenizer, model
        return self._model

    def _marker_id(self, letter: str) -> int:
        """Token id of ``letter``: the first of ``" A"`` and ``"A"`` that is a single token."""
        if letter not in self._marker_ids:
            for form in (f" {letter}", letter):
                ids = self._tokenizer.encode(form, add_special_tokens=False)
                if len(ids) == 1:
                    self._marker_ids[letter] = ids[0]
                    break
            else:
                raise BackendError(
                    f"{self.model_id} has no single-token form of option marker {letter!r}"
                )
        return self._marker_ids[letter]

    def _fit(self, text: str, prompt_for: Callable[[str], str]) -> str:
        """The prompt for ``text``, with ``text`` cut by tokens until the prompt fits."""
        tok, limit = self._tokenizer, self.recipe.max_seq_len
        ids = tok.encode(text, add_special_tokens=False)
        keep = len(ids)
        while True:
            prompt = prompt_for(text if keep == len(ids) else tok.decode(ids[:keep]).strip())
            excess = len(tok.encode(prompt, add_special_tokens=True)) - limit
            if excess <= 0:
                return prompt
            if keep == 0:
                raise BackendError(
                    f"the options alone exceed the {limit}-token limit of {self.model_id}"
                )
            keep = max(0, keep - excess)  # decoding can re-tokenise slightly longer; loop

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        return run_with_cpu_fallback(
            self,
            "scoring",
            lambda: self._score(
                texts, candidates, instruction=instruction, multi_label=multi_label
            ),
        )

    def _score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        n_cands = len(candidates)
        if self.max_candidates is not None and n_cands > self.max_candidates:
            raise BackendError(
                f"mlm-choice takes at most {self.max_candidates} candidates, got {n_cands}"
            )
        if not len(texts) or not n_cands:
            return Scores(np.zeros((len(texts), n_cands)), "softmax")
        model = self._load()
        torch = require("torch", "models")
        tok = self._tokenizer
        mask_token = tok.mask_token
        options = [c.text for c in candidates]
        instruction = instruction or self.recipe.default_instruction
        marker_ids = [self._marker_id(letter) for letter in LETTERS[:n_cands]]

        def prompt_for(text: str) -> str:
            return build_choice_prompt(text, options, instruction=instruction, answer=mask_token)

        prompts = [self._fit(t, prompt_for) for t in texts]
        lengths = [len(tok.encode(p, add_special_tokens=True)) for p in prompts]

        values = np.zeros((len(texts), n_cands), dtype=np.float64)
        order = sorted(range(len(prompts)), key=lengths.__getitem__)
        batch_size = self.recipe.batch_size_for(self._device_in_use)
        start = 0
        while start < len(order):
            stop = start + 1  # lengths ascend, so the batch's longest is its last member
            while (
                stop < len(order)
                and stop - start < batch_size
                and (stop - start + 1) * lengths[order[stop]] <= _BATCH_TOKEN_BUDGET
            ):
                stop += 1
            batch = order[start:stop]
            enc = tok(
                [prompts[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=self.recipe.max_seq_len,
                return_tensors="pt",
            )
            ids = np.asarray(enc["input_ids"])
            rows, cols = np.nonzero(ids == tok.mask_token_id)
            if len(rows) != len(batch) or (rows != np.arange(len(batch))).any():
                raise BackendError("expected exactly one mask token in every prompt")
            with torch.inference_mode():
                logits = model(**enc.to(self._device_in_use)).logits
            at_mask = logits[rows.tolist(), cols.tolist()][:, marker_ids]
            values[batch] = _row_softmax(np.asarray(at_mask.float().cpu(), dtype=np.float64))
            start = stop
        return Scores(values, "softmax")

    def close(self) -> None:
        """Drop the model and tokenizer and release their memory; the next call reloads."""
        self._model = self._tokenizer = None
        self._marker_ids.clear()
        release_memory()

    def supports(self, node_id: str) -> bool:
        return True

    def tokenizer(self) -> Callable[[str], int] | None:
        self._load()
        tok = self._tokenizer
        return lambda text: len(tok.encode(text, add_special_tokens=False))
