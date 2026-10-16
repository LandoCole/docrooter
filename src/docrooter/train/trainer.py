"""Training: contrastive fine-tuning of a shared embedding body plus one linear head per node.

The data helpers (:func:`node_groups`, :func:`trainable_nodes`, :func:`build_triplets`) are pure
Python. :func:`train` needs the ``models`` extra.
"""

from __future__ import annotations

import logging
import math
import random
import time
import warnings
from typing import TYPE_CHECKING, Any

from docrooter._extras import require
from docrooter.backends.recipes import get_recipe
from docrooter.taxonomy import Taxonomy

if TYPE_CHECKING:
    from docrooter.backends.embedding import EmbeddingBackend
    from docrooter.train.bundle import TrainedModel
    from docrooter.train.dataset import Dataset

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "dmis-lab/UltraBERT-Embed-150M"
WARMUP_RATIO = 0.05
MAX_GRAD_NORM = 1.0

Triplet = tuple[str, str, str]


def node_groups(taxonomy: Taxonomy, dataset: Dataset) -> dict[str, list[tuple[str, str]]]:
    """``node -> [(text, child taken next), ...]`` for every node some example passes through.

    The same pairs as :meth:`Dataset.node_examples` gives per node (``Taxonomy.ROOT_ID``
    included), built in one pass over the resolved examples.
    """
    groups: dict[str, list[tuple[str, str]]] = {}
    for example in dataset.resolve(taxonomy):
        parent = Taxonomy.ROOT_ID
        for child in example.path:
            groups.setdefault(parent, []).append((example.text, child))
            parent = child
    return groups


def _internal_nodes(taxonomy: Taxonomy) -> list[str]:
    """``ROOT_ID`` then every node with children, in taxonomy order."""
    return [Taxonomy.ROOT_ID, *(n.id for n in taxonomy.nodes.values() if n.children)]


def _trainable(
    taxonomy: Taxonomy, groups: dict[str, list[tuple[str, str]]], min_examples_per_child: int
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for node_id in _internal_nodes(taxonomy):
        child_ids = [c.id for c in taxonomy.children(node_id)]
        if len(child_ids) < 2 or node_id not in groups:
            continue
        counts = dict.fromkeys(child_ids, 0)
        for _, child in groups[node_id]:
            counts[child] += 1
        if min(counts.values()) >= min_examples_per_child:
            out[node_id] = counts
    return out


def trainable_nodes(
    taxonomy: Taxonomy, dataset: Dataset, min_examples_per_child: int = 4
) -> dict[str, dict[str, int]]:
    """``node -> child -> example count`` for the nodes that get a head.

    A node qualifies when it has at least 2 children and every child has at least
    ``min_examples_per_child`` examples: the classifier offers a head all of a node's children,
    so a head must know them all. Other nodes are left to the zero-shot fallback.
    """
    return _trainable(taxonomy, node_groups(taxonomy, dataset), min_examples_per_child)


def _node_triplets(pairs: list[tuple[str, str]], rng: random.Random) -> list[Triplet]:
    """One ``(anchor, positive, negative)`` per distinct example text at a node."""
    distinct: dict[str, dict[str, None]] = {}
    for text, child in pairs:
        distinct.setdefault(child, {})[text] = None
    by_child = {child: list(texts) for child, texts in distinct.items()}
    children = list(by_child)
    triplets = []
    for child, texts in by_child.items():
        siblings = [c for c in children if c != child]
        if len(texts) < 2 or not siblings:
            continue
        for i, anchor in enumerate(texts):
            j = rng.randrange(len(texts) - 1)
            positive = texts[j + 1 if j >= i else j]  # any other text of the same child
            negative = rng.choice(by_child[rng.choice(siblings)])
            triplets.append((anchor, positive, negative))
    return triplets


def build_triplets(
    taxonomy: Taxonomy, dataset: Dataset, *, max_triplets: int = 20000, seed: int = 0
) -> list[Triplet]:
    """Contrastive triplets ``(anchor, positive, negative)``, deterministic for a ``seed``.

    At every internal node, each example there is an anchor; the positive is another example
    taking the same child and the negative one taking a sibling child (a hard negative by
    construction). Anchors without a positive or a negative are skipped. Above ``max_triplets``
    the nodes contribute in turn, so every node gets an equal share of the cap (or all it has).
    """
    return _triplets(taxonomy, node_groups(taxonomy, dataset), max_triplets, seed)


def _triplets(
    taxonomy: Taxonomy, groups: dict[str, list[tuple[str, str]]], max_triplets: int, seed: int
) -> list[Triplet]:
    rng = random.Random(seed)
    pools = []
    for node_id in _internal_nodes(taxonomy):
        if node_id in groups:
            pool = _node_triplets(groups[node_id], rng)
            rng.shuffle(pool)
            if pool:
                pools.append(pool)
    out: list[Triplet] = []
    k = 0
    while pools and len(out) < max_triplets:  # round k takes the k-th triplet of every node
        pools = [p for p in pools if k < len(p)]
        out.extend(pool[k] for pool in pools[: max_triplets - len(out)])
        k += 1
    rng.shuffle(out)
    return out


def _finetune(
    body: EmbeddingBackend,
    triplets: list[Triplet],
    *,
    prompt: str,
    epochs: int,
    lr: float,
    seed: int,
    gradient_checkpointing: bool = True,
) -> None:
    """Fine-tune ``body``'s SentenceTransformer in place with MultipleNegativesRankingLoss.

    Mirrors ``SentenceTransformerTrainer`` defaults (AdamW, linear decay after
    ``WARMUP_RATIO`` warmup, gradient clipping at 1.0) without its ``accelerate`` dependency.
    Each batch holds three columns (anchor, positive, negative), every text prefixed with the
    query ``prompt``. The batch is the body's (recipe ``batch_size``, else 16, 32 on CUDA),
    and texts are truncated to the body's ``max_seq_len``. With ``gradient_checkpointing``
    (when the model supports it) activations are recomputed in the backward pass instead of
    kept: a batch of 16 triplets at 512 tokens otherwise holds about 10 GB of activations.
    """
    torch = require("torch", "models")
    transformers = require("transformers", "models")
    losses = require("sentence_transformers.sentence_transformer.losses", "models")
    st_util = require("sentence_transformers.util", "models")

    model = body._load()
    model.max_seq_length = body.recipe.max_seq_len
    batch_size = body.batch_size
    transformer = _hf_model(model) if gradient_checkpointing else None
    if transformer is not None:
        transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    torch.manual_seed(seed)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    rows = [tuple(prompt + text for text in triplet) for triplet in triplets]
    steps_per_epoch = math.ceil(len(rows) / batch_size)
    total = steps_per_epoch * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=math.ceil(total * WARMUP_RATIO), num_training_steps=total
    )
    rng = random.Random(seed)
    model.train()
    try:
        for _ in range(epochs):
            order = rows[:]
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch = order[start : start + batch_size]
                features = [
                    st_util.batch_to_device(model.preprocess(list(column)), model.device)
                    for column in zip(*batch, strict=True)
                ]
                loss = loss_fn(features, None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
    finally:
        model.eval()
        if transformer is not None:
            transformer.gradient_checkpointing_disable()


def _finetune_with_fallback(body: EmbeddingBackend, triplets: list[Triplet], **kwargs: Any) -> None:
    """:func:`_finetune`; on an MPS out-of-memory error, again from the base weights with half
    the batch, then on the CPU (with that half batch). Each retry logs a warning."""
    from docrooter.backends.loading import is_mps_oom, release_memory, warn_cpu_fallback

    for attempt in ("mps", "half batch", "cpu"):
        try:
            _finetune(body, triplets, **kwargs)
            return
        except Exception as exc:
            if attempt == "cpu" or not is_mps_oom(exc, body._device_in_use):
                raise
        # Outside the handler, so the failed attempt's activations can be freed. The body
        # reloads from its base model: a partly fine-tuned one would be a different start.
        body.close()
        if attempt == "mps":
            half = max(1, body.batch_size // 2)
            logger.warning(
                "fine-tuning %s ran out of MPS memory at batch %d; retrying with batch %d",
                body.model_id,
                body.batch_size,
                half,
            )
            body.recipe = body.recipe.model_copy(update={"batch_size": half})
        else:
            warn_cpu_fallback(body.name, "fine-tuning")
            body._device = "cpu"
        release_memory()


def _hf_model(model: Any) -> Any:
    """The Hugging Face model inside a SentenceTransformer, if it supports checkpointing."""
    try:
        transformer = model[0].auto_model
    except (AttributeError, IndexError, TypeError):
        return None
    return transformer if getattr(transformer, "supports_gradient_checkpointing", False) else None


def train(
    taxonomy: Taxonomy,
    dataset: Dataset,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    finetune_body: bool = True,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    min_examples_per_child: int = 4,
    max_triplets: int = 20000,
    device: str | None = None,
    seed: int = 0,
    instruction: str | None = None,
    max_seq_len: int = 512,
    gradient_checkpointing: bool = True,
) -> TrainedModel:
    """Train a :class:`TrainedModel`: a shared embedding body and one head per trainable node.

    With ``finetune_body`` the body is first fine-tuned contrastively on
    :func:`build_triplets`. Then each node from :func:`trainable_nodes` gets a
    ``LogisticRegression`` on the body's embeddings of its examples. ``epochs``, ``lr`` and
    ``batch_size`` default to the base model's recipe. Texts are embedded with the recipe's
    query prompt around ``instruction`` (default: the recipe's), at training and at inference.

    Texts are truncated to ``max_seq_len`` tokens (capped at the recipe's own limit) for
    fine-tuning, for the heads' embeddings and, through the saved recipe, at inference: long
    training texts would otherwise allocate attention quadratic in the recipe's limit.
    ``gradient_checkpointing`` (default on) recomputes activations during fine-tuning instead
    of storing them, which bounds its memory (about 6 GB instead of 10 GB on CPU for
    UltraBERT at batch 16); turn it off for speed when memory is plentiful.
    """
    from docrooter import __version__
    from docrooter.backends.embedding import EmbeddingBackend
    from docrooter.train.bundle import TrainedModel, head_file

    linear_model = require("sklearn.linear_model", "models")
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    started = time.perf_counter()
    seq_len = min(max_seq_len, get_recipe(base_model).max_seq_len)
    body = EmbeddingBackend(base_model, device=device, batch_size=batch_size, max_seq_len=seq_len)
    recipe = body.recipe
    instruction = instruction or recipe.default_instruction
    prompt = recipe.query_prompt.format(instruction=instruction)
    groups = node_groups(taxonomy, dataset)
    trainable = _trainable(taxonomy, groups, min_examples_per_child)
    if not trainable:
        warnings.warn(
            f"no node has at least 2 children with {min_examples_per_child}+ examples each; "
            "the model has no heads and will defer every decision to its zero-shot fallback",
            UserWarning,
            stacklevel=2,
        )

    used_epochs, used_lr, n_triplets = 0, None, 0
    if finetune_body:
        triplets = _triplets(taxonomy, groups, max_triplets, seed)
        if triplets:
            used_epochs = epochs if epochs is not None else recipe.finetune_epochs
            used_lr = lr if lr is not None else recipe.finetune_lr
            n_triplets = len(triplets)
            _finetune_with_fallback(
                body,
                triplets,
                prompt=prompt,
                epochs=used_epochs,
                lr=used_lr,
                seed=seed,
                gradient_checkpointing=gradient_checkpointing,
            )
            recipe = body.recipe  # a retry may have halved the batch
            body.name = f"embedding:{base_model} (fine-tuned)"
        else:
            warnings.warn(
                "finetune_body=True but the dataset yields no triplets (no node has two examples "
                "of one child and one of another); skipping body fine-tuning",
                UserWarning,
                stacklevel=2,
            )

    heads: dict[str, Any] = {}
    manifest_heads: dict[str, dict[str, Any]] = {}
    for node_id in trainable:
        pairs = groups[node_id]
        X = body.encode([text for text, _ in pairs], prompt=prompt)
        head = linear_model.LogisticRegression(max_iter=1000, random_state=seed)
        head.fit(X, [child for _, child in pairs])
        heads[node_id] = head
        manifest_heads[node_id] = {
            "file": head_file(node_id),
            "classes": [str(c) for c in head.classes_],
            "n": len(pairs),
        }

    node_counts = {}
    for node_id in _internal_nodes(taxonomy):
        if node_id in groups:
            counts: dict[str, int] = {}
            for _, child in groups[node_id]:
                counts[child] = counts.get(child, 0) + 1
            node_counts[node_id] = counts

    manifest = {
        "docrooter_version": __version__,
        "base_model": base_model,
        "base_revision": recipe.revision,
        "taxonomy_hash": taxonomy.content_hash(),
        "instruction": instruction,
        "recipe": recipe.model_dump(),
        "heads": manifest_heads,
        "node_counts": node_counts,
        "train_metrics": {
            "triplets": n_triplets,
            "epochs": used_epochs,
            "lr": used_lr,
            "device": body._device_in_use,
            "batch_size": body.batch_size,
            "seconds": round(time.perf_counter() - started, 3),
        },
    }
    return TrainedModel(manifest, heads, body)
