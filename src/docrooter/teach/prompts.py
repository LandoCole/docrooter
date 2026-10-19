"""The teacher's prompts (spec §7, §13): wording, input limits and the prompt hash.

Every engine builds its messages here, so the prompt hash in a label's provenance describes
what the model saw. A multiple-choice prompt has two parts:

- the **prefix** (the system turn): the instruction and, for a family-wide prompt (spec §4.0
  ruling 5), every option of the family, lettered A, B, ... This part is the same for every
  element of a family, so the local engines prefill it once and cache it;
- the **suffix** (the user turn): the element's context and text, then the options to choose
  from. With a family prefix the suffix only names the shortlisted letters, and the answer is
  restricted to them; without one the element's own options (at most 8) are lettered A..H here.

A yes/no prompt puts the question in the prefix and the element in the suffix; P(yes) is the
softmax over the "Yes" and "No" tokens.

Element text is cut to :data:`MAX_TEXT_TOKENS` tokens of the teacher's tokenizer and the
context to :data:`MAX_CONTEXT_TOKENS` (Review Focus 3). A table's text is its description
then its Markdown (:func:`table_text`), so the cut keeps the description and the first rows.

Changing any wording here changes :func:`prompt_hash`, so cached labels made with the old
prompt no longer match (Review Focus 2); bump :data:`PROMPT_VERSION` as well when the meaning
changes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence

PROMPT_VERSION = "teach-v1"

MAX_TEXT_TOKENS = 384  # spec §7.1: element text, in teacher tokens
MAX_CONTEXT_TOKENS = 96  # document type, heading path and kind
MAX_ITEM_OPTIONS = 8  # options lettered per element (A..H) when there is no family prefix
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

CHOOSE_SYSTEM = "{instruction}\n\nAnswer with the letter of one option only."
CHOOSE_FAMILY_SYSTEM = (
    "{instruction}\n\nOptions:\n{options_block}\n\nAnswer with the letter of one option only."
)
CHOOSE_SUFFIX = "{context}Text:\n{text}\n\n{choices}Answer with one letter: {letters}."
CHOOSE_ITEM_OPTIONS = "Options:\n{options_block}\n\n"
YES_NO_SYSTEM = "{question}\n\nAnswer Yes or No only."
YES_NO_SUFFIX = "{context}Text:\n{text}\n\nAnswer Yes or No."
YES, NO = "Yes", "No"

_TEMPLATES = (
    "CHOOSE_SYSTEM",
    "CHOOSE_FAMILY_SYSTEM",
    "CHOOSE_SUFFIX",
    "CHOOSE_ITEM_OPTIONS",
    "YES_NO_SYSTEM",
    "YES_NO_SUFFIX",
    "YES",
    "NO",
)


def options_block(options: Sequence[tuple[str, str]]) -> str:
    """The options as lettered lines, ``"A. label: definition"``, in order."""
    if len(options) > len(LETTERS):
        raise ValueError(f"at most {len(LETTERS)} options can be lettered, got {len(options)}")
    return "\n".join(f"{LETTERS[i]}. {text}" for i, (_, text) in enumerate(options))


def letter_list(letters: Sequence[str]) -> str:
    """``"A, B or C"``."""
    if len(letters) <= 1:
        return "".join(letters)
    return ", ".join(letters[:-1]) + " or " + letters[-1]


def table_text(description: str | None, markdown: str) -> str:
    """A table element's teacher text: its description (caption or title), then its cells as
    Markdown, so truncation keeps the description and the header row."""
    description = (description or "").strip()
    return f"{description}\n{markdown}" if description else markdown


def context_text(doc_type: str | None, heading_path: Sequence[str], kind: str) -> str:
    """``"Document type: …\\nSection: A > B\\nElement: paragraph"`` (spec §7.1)."""
    lines = []
    if doc_type:
        lines.append(f"Document type: {doc_type}")
    if heading_path:
        lines.append("Section: " + " > ".join(heading_path))
    lines.append(f"Element: {kind}")
    return "\n".join(lines)


def neutralise(text: str) -> str:
    """Break chat-template markers (``<|im_end|>`` and the like) in element text, so an
    element cannot close its turn or open the answer."""
    return text.replace("<|", "<​|").replace("|>", "|​>")


def truncate_tokens(
    text: str,
    limit: int,
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> str:
    """``text`` cut to its first ``limit`` tokens; unchanged when it fits."""
    ids = encode(text)
    return text if len(ids) <= limit else decode(list(ids)[:limit])


def truncate_chars(text: str, limit_tokens: int) -> str:
    """For engines without the teacher's tokenizer (``openai``): about 4 characters a token."""
    limit = 4 * limit_tokens
    return text if len(text) <= limit else text[:limit]


def choose_system(instruction: str, family_options: Sequence[tuple[str, str]] | None) -> str:
    if family_options is None:
        return CHOOSE_SYSTEM.format(instruction=instruction)
    return CHOOSE_FAMILY_SYSTEM.format(
        instruction=instruction, options_block=options_block(family_options)
    )


def choose_suffix(
    context: str,
    text: str,
    letters: Sequence[str],
    item_options: Sequence[tuple[str, str]] | None,
) -> str:
    """The element's turn. ``item_options`` are listed (lettered A..) when there is no family
    prefix; otherwise only ``letters`` are named."""
    choices = ""
    if item_options is not None:
        choices = CHOOSE_ITEM_OPTIONS.format(options_block=options_block(item_options))
    return CHOOSE_SUFFIX.format(
        context=_context_block(context), text=text, choices=choices, letters=letter_list(letters)
    )


def function_question(label: str, description: str) -> str:
    """The yes/no question for one function tag (spec §7.3): its label and definition, then
    the question. On the integration items this wording answered 4 of 4, where asking "Does
    this element have this function?" first answered a past-results sentence as forward-looking.
    """
    return f"Function: {label}. {description}\nDoes the element have this function?"


def yes_no_system(question: str) -> str:
    return YES_NO_SYSTEM.format(question=question)


def yes_no_suffix(context: str, text: str) -> str:
    return YES_NO_SUFFIX.format(context=_context_block(context), text=text)


def _context_block(context: str) -> str:
    """The context and a blank line, or nothing. The element's turn then always starts with a
    letter, so it tokenises the same apart from the prefix as joined to it."""
    context = context.strip()
    return f"{context}\n\n" if context else ""


def prompt_hash(instruction: str, options_block: str) -> str:
    """sha256 of the prompt version, every template above, the instruction (or yes/no
    question) and the family's lettered options block (``""`` when options are per element).
    Goes into each label's provenance and its :func:`~docrooter.teach.label_key`."""
    payload = {
        "version": PROMPT_VERSION,
        "templates": {name: globals()[name] for name in _TEMPLATES},
        "limits": [MAX_TEXT_TOKENS, MAX_CONTEXT_TOKENS, MAX_ITEM_OPTIONS],
        "instruction": instruction,
        "options": options_block,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
