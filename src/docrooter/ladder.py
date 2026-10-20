"""The context ladder: four named levels of context, each an exact view preset.

A level names how much of a section or a document is shown to a backend. Each rung shows
everything the rung below shows and adds one field, with a budget step at least as large as
that field, so a higher rung renders at least as many estimated tokens as a lower one and
loses nothing it showed. Budgets are view budgets in estimated tokens (1.3 per word) unless
the backend supplies a tokenizer; the OpenAI-compatible backend does not.

Section presets (:func:`section_view`, a :class:`~docrooter.views.SectionView`):

=========  ==========================================  ==========  ====================
Level      Fields, in order                            max_tokens  Other settings
=========  ==========================================  ==========  ====================
headings   path, title                                 32
terms      path, title, keyphrases                     64          8 key phrases
lead       path, title, keyphrases, lead               192         8 key phrases,
                                                                   3 lead sentences
full       path, title, keyphrases, text               2048        8 key phrases
=========  ==========================================  ==========  ====================

The section ``text`` starts with the section's lead, so ``full`` contains ``lead``.

Document presets (:func:`document_view`, a :class:`~docrooter.doc_view.DocumentView`); every
rung shows the contents to depth 2, at most 40 entries:

=========  ==========================================  ==========  ====================
Level      Fields, in order                            max_tokens  Other settings
=========  ==========================================  ==========  ====================
headings   title, stats, toc                           320
terms      title, stats, keyphrases, toc               448         12 pooled key phrases
lead       title, stats, keyphrases, lead, toc         640         12 pooled key phrases,
                                                                   120 lead words
full       title, stats, keyphrases, toc, text         12,576      12 pooled key phrases,
                                                       (see below) 120 lead words
=========  ==========================================  ==========  ====================

The contents can be long, so the fields a rung adds go before it: the contents take what is
left, which is never less than the rung below left them.

The document ``text`` is the whole body text, each word once: the lead (120 words, skipping
cover and notice boilerplate) first, then the rest in reading order. It is cut head-first, and
the rendered view (and ``DocTypeResult.truncated``) records the cut. Only the words a budget
could ever hold are gathered, so a long document is never counted whole.

``full_cap`` (default :data:`FULL_CAP`, 16,000) is the target size of the whole prompt, checked
against the API's ``prompt_tokens``; the benchmark reports the share of documents measured over
it. The ``full`` view's budget is ``full_cap`` less a reserve of :data:`FULL_RESERVE_TOKENS`
(1,024) plus :data:`FULL_RESERVE_SHARE` (15%) of ``full_cap``, which covers the instruction, the
options and the error of the estimate: 16,000 - 1,024 - 2,400 = 12,576 (see
:func:`full_view_budget`).

``level=None`` on the classification entry points keeps their old views; passing ``level=``
together with ``view=`` raises ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Literal

from docrooter.doc_view import DocumentView
from docrooter.views import SectionView

Level = Literal["headings", "terms", "lead", "full"]
LEVELS: tuple[Level, ...] = ("headings", "terms", "lead", "full")
FULL_CAP = 16000
"""The default target prompt size of the document ``full`` level (spec ruling 6.1)."""
FULL_RESERVE_TOKENS = 1024
"""Fixed part of the prompt reserve kept out of the ``full`` view's budget."""
FULL_RESERVE_SHARE = 0.15
"""Share of ``full_cap`` also kept out of the ``full`` view's budget."""

_SECTION_PRESETS: dict[str, SectionView] = {
    "headings": SectionView(fields=("path", "title"), max_tokens=32),
    "terms": SectionView(fields=("path", "title", "keyphrases"), max_tokens=64, keyphrases=8),
    "lead": SectionView(
        fields=("path", "title", "keyphrases", "lead"),
        max_tokens=192,
        keyphrases=8,
        lead_sentences=3,
    ),
    "full": SectionView(
        fields=("path", "title", "keyphrases", "text"), max_tokens=2048, keyphrases=8
    ),
}

_DOCUMENT_PRESETS: dict[str, DocumentView] = {
    "headings": DocumentView(
        fields=("title", "stats", "toc"), max_tokens=320, toc_depth=2, toc_entries=40
    ),
    "terms": DocumentView(
        fields=("title", "stats", "keyphrases", "toc"),
        max_tokens=448,
        keyphrases=12,
        toc_depth=2,
        toc_entries=40,
    ),
    "lead": DocumentView(
        fields=("title", "stats", "keyphrases", "lead", "toc"),
        max_tokens=640,
        keyphrases=12,
        lead_words=120,
        toc_depth=2,
        toc_entries=40,
    ),
    "full": DocumentView(
        fields=("title", "stats", "keyphrases", "toc", "text"),
        max_tokens=FULL_CAP,  # replaced by full_view_budget(full_cap)
        keyphrases=12,
        lead_words=120,
        toc_depth=2,
        toc_entries=40,
    ),
}


def section_view(level: Level) -> SectionView:
    """The :class:`SectionView` preset of ``level`` (a fresh copy)."""
    return _SECTION_PRESETS[_checked(level)].model_copy()


def document_view(level: Level, *, full_cap: int = FULL_CAP) -> DocumentView:
    """The :class:`DocumentView` preset of ``level`` (a fresh copy). ``full_cap`` is the target
    prompt size of the ``full`` level, whose view gets :func:`full_view_budget` of it; the other
    levels ignore it."""
    view = _DOCUMENT_PRESETS[_checked(level)]
    if level != "full":
        return view.model_copy()
    return view.model_copy(update={"max_tokens": full_view_budget(full_cap)})


def full_view_budget(full_cap: int = FULL_CAP) -> int:
    """The ``full`` document view's token budget: ``full_cap`` less the prompt reserve
    (``FULL_RESERVE_TOKENS + FULL_RESERVE_SHARE * full_cap``, rounded up). ``ValueError`` when
    nothing is left."""
    reserve = FULL_RESERVE_TOKENS + math.ceil(round(FULL_RESERVE_SHARE * full_cap, 6))
    budget = full_cap - reserve
    if budget < 1:
        raise ValueError(
            f"full_cap {full_cap} leaves no view budget after a {reserve}-token reserve"
        )
    return budget


def next_level(level: Level) -> Level | None:
    """The rung above ``level``; None above ``full``."""
    i = LEVELS.index(_checked(level))
    return LEVELS[i + 1] if i + 1 < len(LEVELS) else None


def check_level_and_view(level: Level | None, view: object | None) -> None:
    """Raise ``ValueError`` when both ``level`` and ``view`` are given, or ``level`` is unknown."""
    if level is not None and view is not None:
        raise ValueError("pass level= or view=, not both")
    if level is not None:
        _checked(level)


def _checked(level: str) -> Level:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; valid levels: {', '.join(LEVELS)}")
    return level
