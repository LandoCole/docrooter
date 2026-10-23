"""Heading validation (reject false headings) and hierarchy repair (fix levels)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from docrooter.document.models import RawHeading

_BASE_SCORE = 0.6
_LOW_CONFIDENCE = 0.7
_NUMBERING = re.compile(
    r"^("
    r"(\d+(\.\d+)*\.?|(Article|Section|Chapter|Part|Item|Appendix)\s+\w+)\b"
    # Roman numerals stay uppercase and well-formed so "Civil." or "Li. Chen" do not count.
    r"|(?-i:(?=[IVXLC])(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\.(?=\s|$)"
    r")",
    re.IGNORECASE,
)
_PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.IGNORECASE)
_URL_OR_EMAIL = re.compile(r"https?://|www\.|\S+@\S+\.\S+", re.IGNORECASE)
# A title-case token: letters, optionally joined by apostrophes or hyphens ("Company's").
_WORD = re.compile(r"[^\W\d_]+(?:['\u2019-][^\W\d_]+)*")
_CONNECTORS = frozenset({"and", "of", "the", "in", "for", "to", "a", "an", "on", "with"})
_TITLE_CASE_RATIO = 0.7
_MAX_WORDS = 20
_MIN_ALNUM_RATIO = 0.4
# Dotted-numeric prefix used to infer levels; parts are capped at 3 digits so years don't match.
_DOTTED = re.compile(r"^(\d{1,3}(?:\.\d{1,3})*)\.?(?:\s|$)")
_NUMBERED_FRACTION = 0.5


def _is_title_case_or_caps(title: str) -> bool:
    letters = [c for c in title if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        return True
    words = [w for w in _WORD.findall(title) if w.casefold() not in _CONNECTORS]
    if not words:
        return False
    upper = sum(1 for w in words if w[0].isupper())
    return upper / len(words) >= _TITLE_CASE_RATIO


def score_heading(title: str, *, repeat_count: int = 1) -> tuple[float, list[str]]:
    """Score how heading-like ``title`` is, in [0, 1], with the reasons that moved the score."""
    text = title.strip()
    score = _BASE_SCORE
    reasons: list[str] = []

    def adjust(delta: float, reason: str) -> None:
        nonlocal score
        score += delta
        reasons.append(reason)

    if _NUMBERING.match(text):
        adjust(0.25, "numbering_prefix")
    if _is_title_case_or_caps(text):
        adjust(0.1, "title_case")
    if len(text.split()) > _MAX_WORDS:
        adjust(-0.4, "too_long")
    if text.endswith((".", ",", ";")):
        adjust(-0.3, "sentence_punctuation")
    if _PAGE_NUMBER.match(text):
        adjust(-0.7, "page_number")
    if _URL_OR_EMAIL.search(text):
        adjust(-0.6, "url_or_email")
    chars = [c for c in text if not c.isspace()]
    if sum(c.isalnum() for c in chars) < _MIN_ALNUM_RATIO * len(chars) or not chars:
        adjust(-0.5, "low_alphanumeric")
    if text.startswith(("-", "*", "+", "|")):
        adjust(-0.6, "list_or_table_marker")
    if repeat_count >= 3:
        adjust(-0.5, "repeated_title")

    return round(min(1.0, max(0.0, score)), 4), reasons


def _normalise(title: str) -> str:
    return " ".join(title.casefold().split())


def validate_headings(raw: list[RawHeading], *, min_score: float = 0.5) -> list[RawHeading]:
    """Drop false headings, merging each into the previous block's text.

    A rejected heading's title is appended as a line, followed by its body, to the text of the
    previous kept block. When there is no previous block it becomes a level-0 "Preamble" block.
    Survivors get ``heading_score`` and, below 0.7, the ``low_heading_confidence`` flag. The
    level-0 preamble block is not a heading and passes through untouched. Inputs are not mutated.
    """
    counts = Counter(_normalise(h.title) for h in raw if h.level > 0)
    out: list[RawHeading] = []
    for heading in raw:
        if heading.level <= 0:
            out.append(heading)
            continue
        score, _ = score_heading(heading.title, repeat_count=counts[_normalise(heading.title)])
        if score >= min_score:
            flags = list(heading.flags)
            if score < _LOW_CONFIDENCE and "low_heading_confidence" not in flags:
                flags.append("low_heading_confidence")
            out.append(heading.model_copy(update={"heading_score": score, "flags": flags}))
            continue
        merged = "\n".join(part for part in (heading.title, heading.text) if part)
        if out:
            previous = out[-1]
            text = "\n".join(part for part in (previous.text, merged) if part)
            out[-1] = previous.model_copy(update={"text": text})
        else:
            out.append(
                RawHeading(
                    title="Preamble", level=0, text=merged, line=heading.line, page=heading.page
                )
            )
    return out


def repair_levels(raw: list[RawHeading], *, keep_flags: Sequence[str] = ()) -> list[RawHeading]:
    """Fix heading levels.

    When at least half of the headings carry a dotted-numeric prefix ("1", "1.2", "1.2.3"), those
    headings take the number of dot-separated parts as their level. Gaps are then closed: a
    heading goes one level below the nearest heading before it with a smaller level (1 when
    there is none), so a level never jumps by more than one and headings of one level stay
    siblings. A heading carrying one of ``keep_flags`` keeps its level exactly, and the
    headings after it nest under it by the same rule. Changed headings get the
    ``level_repaired`` flag. The level-0 preamble block is untouched. Inputs are not mutated.
    """
    headings = [h for h in raw if h.level > 0]
    numbered = sum(1 for h in headings if _DOTTED.match(h.title.strip()))
    use_numbering = bool(headings) and numbered / len(headings) >= _NUMBERED_FRACTION

    levels: list[int] = []
    kept: list[int | None] = []
    for heading in raw:
        keep = heading.level > 0 and any(flag in heading.flags for flag in keep_flags)
        kept.append(heading.level if keep else None)
        match = (
            _DOTTED.match(heading.title.strip()) if use_numbering and heading.level > 0 else None
        )
        levels.append(len(match.group(1).split(".")) if match and not keep else heading.level)
    return _relevel(raw, _stacked(levels, kept))


def _stacked(levels: list[int], kept: list[int | None]) -> list[int]:
    """Gap-free levels for ``levels``, keeping the ``kept`` ones.

    A stack holds ``(level in, level out)`` of the open headings. A heading closes those with a
    level in at or above its own and goes one below the one left open (1 when none is). A kept
    heading closes those with a level out at or above the one it keeps, and keeps it.
    """
    out: list[int] = []
    stack: list[tuple[int, int]] = []
    for level, keep in zip(levels, kept, strict=True):
        if level <= 0:
            out.append(level)
            continue
        if keep is not None:
            while stack and stack[-1][1] >= keep:
                stack.pop()
            new = keep
        else:
            while stack and stack[-1][0] >= level:
                stack.pop()
            new = stack[-1][1] + 1 if stack else 1
        stack.append((level if keep is None else keep, new))
        out.append(new)
    return out


def _relevel(raw: list[RawHeading], levels: list[int]) -> list[RawHeading]:
    """Each heading at its new level, flagged ``level_repaired`` when that changes it."""
    out: list[RawHeading] = []
    for heading, level in zip(raw, levels, strict=True):
        if level != heading.level:
            flags = list(heading.flags)
            if "level_repaired" not in flags:
                flags.append("level_repaired")
            heading = heading.model_copy(update={"level": level, "flags": flags})
        out.append(heading)
    return out
