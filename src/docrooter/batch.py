"""Batched LLM prompts: few chat calls per document stage, each option listed once.

Per-item calls repeat the whole option list for every section, so the options, not the
document, dominate the prompt. The sections stage is batched in two steps by default: one call
sorts every section into a group of the taxonomy (:func:`group_prompt`, the groups listed once
with a one-line gloss), then one call per group that received sections chooses among that
group's leaves only (:func:`section_prompt` with ``leaf_ids``). The flat variant lists every
leaf in one call. A batched call sends:

- **a system turn** with the instruction and the options once, each as ``id: label: definition``,
  every group followed by its leaves (indented);
- **a user turn** with the numbered views (``[1] ...``) of the sections, or the one document
  view.

The reply is a JSON object: ``{"<n>": "<leaf id>", ...}`` for sections, ``{"doc_type": "<id>"}``
for the document. The prompts are also what the token estimates of non-LLM backends are built
from, since they are what an LLM would be sent: :meth:`BatchPrompt.parts` estimates the
instructions, the options and the content separately, and :meth:`BatchPrompt.estimate` is their
sum.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from docrooter.ladder import FULL_CAP, FULL_RESERVE_SHARE
from docrooter.taxonomy import Taxonomy
from docrooter.views import estimate_tokens, tokens_for_words

MAX_BATCH = 40
"""The most sections one batched call carries."""
MAX_CALL_TOKENS = FULL_CAP
"""The default size limit of one call's prompt (instructions, options and views): the 16,000
tokens of the ``full`` cap (spec ruling 6.1)."""

SECTION_INSTRUCTION = (
    "You label the sections of one document. For each numbered section, choose the single "
    "best option id from the list below.\n"
    'Reply with a JSON object mapping each section number to one option id, such as {"1": '
    '"<id>", "2": "<id>"}. Use only the ids listed.'
)
GROUP_INSTRUCTION = (
    "You sort the sections of one document into groups. For each numbered section, choose the "
    "single best group id from the list below.\n"
    'Reply with a JSON object mapping each section number to one group id, such as {"1": '
    '"<id>", "2": "<id>"}. Use only the ids listed.'
)
GLOSS_WORDS = 16
"""The most words of a group's description its gloss keeps."""
DOCUMENT_INSTRUCTION = (
    "You classify one whole document. Choose the single best document type id from the list "
    "below.\n"
    'Reply with a JSON object {"doc_type": "<id>"}. Use only the ids listed.'
)
DOC_TYPE_KEY = "doc_type"
RETRY_NOTE = (
    "\nYour previous reply named an id that is not one of the listed ids. Answer again, with "
    "listed ids only."
)
"""Added to the instructions of a re-ask, so the model knows why it is asked again."""


@dataclass(frozen=True)
class BatchPrompt:
    """The chat messages of one batched call and the parts they are made of.

    ``items`` are the content blocks of the user turn (one per numbered section, or the
    document view), joined by blank lines.
    """

    instructions: str
    options: str
    items: tuple[str, ...]

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": f"{self.instructions}\n\n{self.options}"},
            {"role": "user", "content": "\n\n".join(self.items)},
        ]

    def parts(self) -> dict[str, int]:
        """Estimated prompt tokens of the ``instructions``, the ``options`` and the ``content``
        (the rest of :meth:`estimate`, so the parts sum to it)."""
        instructions = estimate_tokens(self.instructions)
        options = estimate_tokens(self.options)
        content = self.estimate() - instructions - options
        return {"instructions": instructions, "options": options, "content": content}

    def estimate(self) -> int:
        """The estimated prompt tokens of :attr:`messages`, as the backend estimates a call
        that reports no usage."""
        return messages_tokens(self.messages)

    def retry(self) -> BatchPrompt:
        """This prompt for a re-ask: the instructions say the last reply named no listed id."""
        return dataclasses.replace(self, instructions=self.instructions + RETRY_NOTE)


def messages_tokens(messages: Sequence[Mapping[str, str]]) -> int:
    """The estimated prompt tokens of chat ``messages``: each message's text, estimated."""
    return sum(estimate_tokens(m["content"]) for m in messages)


def options_listing(taxonomy: Taxonomy, leaf_ids: Collection[str] | None = None) -> str:
    """``Options:`` then every leaf (or those in ``leaf_ids``) as ``id: label: definition``,
    each under its groups, in taxonomy order; a group lists only the groups and leaves below it
    that are kept."""
    keep = {leaf.id for leaf in taxonomy.leaves()} if leaf_ids is None else set(leaf_ids)
    for leaf_id in list(keep):
        keep.update(node.id for node in taxonomy.path(leaf_id))
    lines = ["Options:"]
    stack = [(node, 0) for node in reversed(taxonomy.children())]
    while stack:
        node, depth = stack.pop()
        if node.id not in keep:
            continue
        text = f"{node.id}: {node.label}"
        if node.description:
            text += f": {node.description}"
        lines.append("  " * depth + text)
        stack.extend((child, depth + 1) for child in reversed(taxonomy.children(node.id)))
    return "\n".join(lines)


def groups_listing(taxonomy: Taxonomy) -> str:
    """``Options:`` then every top-level node as ``id: label: gloss``, the gloss being the first
    sentence of its description, at most :data:`GLOSS_WORDS` words."""
    lines = ["Options:"]
    for node in taxonomy.children():
        text = f"{node.id}: {node.label}"
        if gloss := _gloss(node.description or ""):
            text += f": {gloss}"
        lines.append(text)
    return "\n".join(lines)


def _gloss(description: str) -> str:
    first = description.strip().split(". ")[0].rstrip(".")
    words = first.split()
    return " ".join(words[:GLOSS_WORDS]) + ("…" if len(words) > GLOSS_WORDS else "")


def section_prompt(
    views: Mapping[str, str], taxonomy: Taxonomy, leaf_ids: Collection[str] | None = None
) -> BatchPrompt:
    """The batched sections call: ``views`` maps each section's number to its view. The options
    are every leaf, or only ``leaf_ids`` (one group's leaves, in the two-step stage)."""
    items = tuple(f"[{n}] {view}" for n, view in views.items())
    return BatchPrompt(SECTION_INSTRUCTION, options_listing(taxonomy, leaf_ids), items)


def group_prompt(views: Mapping[str, str], taxonomy: Taxonomy) -> BatchPrompt:
    """The first call of the two-step sections stage: each section's group."""
    items = tuple(f"[{n}] {view}" for n, view in views.items())
    return BatchPrompt(GROUP_INSTRUCTION, groups_listing(taxonomy), items)


def group_leaves(taxonomy: Taxonomy, group_id: str) -> list[str]:
    """The leaves under ``group_id`` (itself, for a leaf), in taxonomy order."""
    out: list[str] = []
    stack = [taxonomy.node(group_id)]
    while stack:
        node = stack.pop()
        children = taxonomy.children(node.id)
        if not children:
            out.append(node.id)
        stack.extend(reversed(children))
    return out


def document_prompt(
    view: str, taxonomy: Taxonomy, leaf_ids: Sequence[str] | None = None
) -> BatchPrompt:
    """The batched document call: the document ``view`` and the doc types offered."""
    return BatchPrompt(DOCUMENT_INSTRUCTION, options_listing(taxonomy, leaf_ids), (view,))


def chunks(items: Sequence[Any], size: int = MAX_BATCH) -> list[Sequence[Any]]:
    """``items`` in consecutive runs of at most ``size``."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def call_budget(max_call_tokens: int = MAX_CALL_TOKENS) -> int:
    """The estimated prompt tokens a call is packed to: ``max_call_tokens`` less
    :data:`~docrooter.ladder.FULL_RESERVE_SHARE` (15%) of it, which covers the error of the
    estimate, so the measured prompt stays within ``max_call_tokens`` (13,600 of 16,000).
    ``ValueError`` below 1."""
    if max_call_tokens < 1:
        raise ValueError(f"max_call_tokens must be at least 1, got {max_call_tokens}")
    return max_call_tokens - math.ceil(round(FULL_RESERVE_SHARE * max_call_tokens, 6))


def cut_to_tokens(text: str, max_tokens: int) -> str:
    """The head of ``text`` whose :func:`~docrooter.views.estimate_tokens` is at most
    ``max_tokens``: whole words, with the text's own spacing ("" when not one word fits)."""
    words = max(0, math.floor(max_tokens / 1.3))
    while words and tokens_for_words(words) > max_tokens:
        words -= 1
    while tokens_for_words(words + 1) <= max_tokens:
        words += 1
    if not words:
        return ""
    for i, match in enumerate(re.finditer(r"\S+", text), start=1):
        if i == words:
            return text[: match.end()]
    return text  # it has no more words than fit


def pack(
    views: Mapping[str, str],
    prompt_for: Callable[[dict[str, str]], BatchPrompt],
    budget: int,
    size: int = MAX_BATCH,
) -> list[dict[str, str]]:
    """``views`` (by section number, in order) packed into calls: each call carries at most
    ``size`` views, and its prompt, also when asked again (:meth:`BatchPrompt.retry`), is
    estimated at most ``budget`` tokens. A view too large to fit a call alone is cut head-first
    (:func:`cut_to_tokens`) and sent alone; compare a call's views with ``views`` to see which.
    ``ValueError`` when the instructions and options leave no room for a view."""
    overhead = prompt_for({}).retry().estimate()
    room = budget - overhead
    if room < tokens_for_words(2):  # "[n]" and one word
        raise ValueError(
            f"a {budget}-token call budget leaves no room for a section view after the "
            f"{overhead}-token instructions and options; raise max_call_tokens"
        )
    calls: list[dict[str, str]] = []
    current: dict[str, str] = {}
    used = 0
    for n, view in views.items():
        cost = estimate_tokens(f"[{n}] {view}")  # an item's words, so the sum bounds the join
        if cost > room:  # too large for any call: cut, and sent alone
            if current:
                calls.append(current)
                current, used = {}, 0
            prefix = f"[{n}] "
            calls.append({n: cut_to_tokens(prefix + view, room)[len(prefix) :]})
            continue
        if current and (len(current) >= size or used + cost > room):
            calls.append(current)
            current, used = {}, 0
        current[n] = view
        used += cost
    if current:
        calls.append(current)
    return calls


def reply_tokens(reply: Mapping[str, str]) -> int:
    """The estimated completion tokens of a JSON reply."""
    return estimate_tokens(json.dumps(dict(reply)))


def valid_id(value: object, valid: Mapping[str, str]) -> str | None:
    """``value`` as one of the ``valid`` ids (keyed by lower case), or ``None``. Surrounding
    spaces, quotes and case are forgiven."""
    if not isinstance(value, str):
        return None
    return valid.get(value.strip().strip("\"'").strip().lower())


def section_reply_max_tokens(n: int) -> int:
    """A completion budget for a reply naming ``n`` section ids."""
    return 32 + 24 * n


__all__ = [
    "DOCUMENT_INSTRUCTION",
    "DOC_TYPE_KEY",
    "GROUP_INSTRUCTION",
    "MAX_BATCH",
    "MAX_CALL_TOKENS",
    "RETRY_NOTE",
    "SECTION_INSTRUCTION",
    "BatchPrompt",
    "call_budget",
    "chunks",
    "cut_to_tokens",
    "document_prompt",
    "group_leaves",
    "messages_tokens",
    "group_prompt",
    "groups_listing",
    "options_listing",
    "pack",
    "reply_tokens",
    "section_prompt",
    "section_reply_max_tokens",
    "valid_id",
]
