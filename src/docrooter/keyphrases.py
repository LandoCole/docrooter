"""Corpus-aware keyphrase extraction (TF-IDF over 1-3-grams, no external dependencies)."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does
    doesn't doing don't down during each either else few for from further had hadn't has hasn't
    have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
    however i i'd i'll i'm i've if in into is isn't it it's its itself just let's may me might
    more most must mustn't my myself neither no nor not of off on once only or other ought our
    ours ourselves out over own per same shall shan't she she'd she'll she's should shouldn't so
    some such than that that's the their theirs them themselves then there there's these they
    they'd they'll they're they've this those through thus to too under until up upon us very
    via was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
    whether which while who who's whom whose why why's will with within without won't would
    wouldn't yet you you'd you'll you're you've your yours yourself yourselves
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")
_MAX_NGRAM = 3


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(text: str) -> list[tuple[str, ...]]:
    """Candidate n-grams (1-3 words) in order of occurrence, repeats included.

    A candidate never contains a stopword, which also guarantees it neither starts nor ends
    with one.
    """
    tokens = _tokenize(text)
    grams: list[tuple[str, ...]] = []
    for start in range(len(tokens)):
        for n in range(1, _MAX_NGRAM + 1):
            window = tokens[start : start + n]
            if len(window) < n or any(word in STOPWORDS for word in window):
                break  # a longer window starting here would contain the same stopword
            grams.append(tuple(window))
    return grams


def document_frequencies(corpus: Sequence[str]) -> Counter[tuple[str, ...]]:
    """Number of corpus texts containing each candidate n-gram."""
    df: Counter[tuple[str, ...]] = Counter()
    for text in corpus:
        df.update(set(_ngrams(text)))
    return df


def keyphrases_from_frequencies(
    text: str, df: Counter[tuple[str, ...]], n_docs: int, k: int = 8
) -> list[str]:
    """Top-``k`` keyphrases of ``text`` given precomputed corpus document frequencies."""
    return [phrase for phrase, _score in scored_keyphrases(text, df, n_docs, k)]


def scored_keyphrases(
    text: str, df: Counter[tuple[str, ...]], n_docs: int, k: int = 8
) -> list[tuple[str, float]]:
    """:func:`keyphrases_from_frequencies` with each phrase's score, best first."""
    if k <= 0:
        return []
    tf = Counter(_ngrams(text))
    scored = [
        (
            count * math.log((1 + n_docs) / (1 + df.get(gram, 0))) * (1 + 0.5 * (len(gram) - 1)),
            -index,  # earlier first occurrence wins ties
            " ".join(gram),
        )
        for index, (gram, count) in enumerate(tf.items())
    ]
    scored.sort(reverse=True)
    return drop_contained([(phrase, score) for score, _order, phrase in scored], k)


def drop_contained(ranked: Sequence[tuple[str, float]], k: int) -> list[tuple[str, float]]:
    """The first ``k`` of ``ranked`` (best first), skipping phrases inside an earlier kept one."""
    kept: list[tuple[str, float]] = []
    for phrase, score in ranked:
        if len(kept) == k:
            break
        padded = f" {phrase} "
        if any(padded in f" {other} " for other, _ in kept):
            continue
        kept.append((phrase, score))
    return kept


def extract_keyphrases(text: str, corpus: Sequence[str], k: int = 8) -> list[str]:
    """Distinctive 1-3-word phrases of ``text`` relative to ``corpus``.

    Score is ``tf * idf * (1 + 0.5 * (n_words - 1))``. A phrase contained in a higher-scored
    kept phrase is dropped.
    """
    return keyphrases_from_frequencies(text, document_frequencies(corpus), len(corpus), k)
