"""Per-model recipes: prompts, sequence length, trust settings and fine-tuning defaults."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from docrooter.runtime import default_batch_size


class Recipe(BaseModel):
    """How to talk to one model. Defaults suit a plain sentence-embedding model.

    ``batch_size`` is the encoding (and fine-tuning) batch; ``None`` means 32 on CUDA and,
    elsewhere, 16 within the default 8 GiB memory budget (see :meth:`batch_size_for`).
    """

    model_config = ConfigDict(extra="forbid")

    revision: str | None = None
    trust_remote_code: bool = False
    query_prompt: str = "{instruction}\n"
    label_prompt: str = ""
    default_instruction: str = "Classify this document section"
    max_seq_len: int = 512
    temperature: float = 0.05
    batch_size: int | None = None
    finetune_lr: float = 2e-5
    finetune_epochs: int = 1

    def batch_size_for(self, device: str) -> int:
        """The batch size to use on ``device``: ``batch_size`` if set, else the device default."""
        if self.batch_size is not None:
            return self.batch_size
        return default_batch_size(device)


_RECIPES: dict[str, Recipe] = {
    "dmis-lab/UltraBERT-Embed-150M": Recipe(
        revision="367d5343b5a0164d57b810cedf5de4b8983d5984",
        trust_remote_code=True,
        query_prompt="Instruct: {instruction}\nQuery: ",
        label_prompt="Label: ",
        max_seq_len=1024,
        finetune_lr=5e-5,
    ),
    "dmis-lab/UltraBERT-380M": Recipe(
        revision="65bbbfeefb03bc93b9965d1d511708dabcee8511",
        trust_remote_code=True,
        max_seq_len=1024,
        finetune_lr=3e-5,
    ),
    "BAAI/bge-small-en-v1.5": Recipe(
        query_prompt="Represent this sentence for searching relevant passages: ",
        max_seq_len=512,
    ),
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": Recipe(max_seq_len=512, temperature=1.0),
}


def register_recipe(model_id: str, recipe: Recipe) -> None:
    """Register (or replace) the recipe used for ``model_id``."""
    _RECIPES[model_id] = recipe


def get_recipe(model_id: str, **overrides: Any) -> Recipe:
    """Recipe for ``model_id`` (defaults if unregistered) with ``overrides`` applied.

    Overrides whose value is ``None`` are ignored, so callers can forward optional arguments
    unconditionally. Unknown field names raise ``pydantic.ValidationError``. The registered
    recipe is never modified.
    """
    base = _RECIPES.get(model_id, Recipe())
    changes = {k: v for k, v in overrides.items() if v is not None}
    if not changes:
        return base.model_copy()
    return Recipe.model_validate({**base.model_dump(), **changes})
