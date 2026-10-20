"""The context ladder: four named levels of context, each an exact view preset.

A level names how much of a section or a document is shown to a backend. Each rung keeps the
previous rung's fields, in the same order, adds one more (``full`` swaps the lead for the whole
text, which contains it) and raises the token budget, so a higher rung renders at least as many
estimated tokens as a lower one. Budgets are view budgets (estimated tokens unless the backend
supplies a tokenizer); the instruction and options of a prompt come on top of them.

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

Document presets (:func:`document_view`, a :class:`~docrooter.doc_view.DocumentView`); every
rung shows the contents to depth 2, at most 40 entries:

=========  ==========================================  ==========  ====================
Level      Fields, in order                            max_tokens  Other settings
=========  ==========================================  ==========  ====================
headings   title, stats, toc                           320
terms      title, stats, toc, keyphrases               384         12 pooled key phrases
lead       title, stats, toc, keyphrases, lead         576         12 pooled key phrases,
                                                                   120 lead words
full       title, stats, toc, keyphrases, text         full_cap    12 pooled key phrases
                                                       (16,000)
=========  ==========================================  ==========  ====================

``full`` replaces the lead with the whole text (it contains the lead). For a document that is
the whole body text in reading order, cut head-first at ``full_cap`` tokens; the rendered view
records ``truncated`` when the cut happens. Only the words a budget could ever hold are
gathered, so a long document is never counted whole.

``level=None`` on the classification entry points keeps their old views; passing ``level=``
together with ``view=`` raises ``ValueError``.
"""

from __future__ import annotations

from typing import Literal

from docrooter.doc_view import DocumentView
from docrooter.views import SectionView

Level = Literal["headings", "terms", "lead", "full"]
LEVELS: tuple[Level, ...] = ("headings", "terms", "lead", "full")
FULL_CAP = 16000
"""The default token cap of the document ``full`` level (spec ruling 6.1)."""

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
        fields=("title", "stats", "toc", "keyphrases"),
        max_tokens=384,
        keyphrases=12,
        toc_depth=2,
        toc_entries=40,
    ),
    "lead": DocumentView(
        fields=("title", "stats", "toc", "keyphrases", "lead"),
        max_tokens=576,
        keyphrases=12,
        lead_words=120,
        toc_depth=2,
        toc_entries=40,
    ),
    "full": DocumentView(
        fields=("title", "stats", "toc", "keyphrases", "text"),
        max_tokens=FULL_CAP,
        keyphrases=12,
        toc_depth=2,
        toc_entries=40,
    ),
}


def section_view(level: Level) -> SectionView:
    """The :class:`SectionView` preset of ``level`` (a fresh copy)."""
    return _SECTION_PRESETS[_checked(level)].model_copy()


def document_view(level: Level, *, full_cap: int = FULL_CAP) -> DocumentView:
    """The :class:`DocumentView` preset of ``level`` (a fresh copy). ``full_cap`` is the token
    budget of the ``full`` level and is ignored by the others."""
    view = _DOCUMENT_PRESETS[_checked(level)]
    if level != "full":
        return view.model_copy()
    if full_cap < 1:
        raise ValueError(f"full_cap must be at least 1; got {full_cap}")
    return view.model_copy(update={"max_tokens": full_cap})


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
