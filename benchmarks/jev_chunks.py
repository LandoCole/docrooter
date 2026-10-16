"""The naive Jev baseline for document types: raw page text, chunked, one Choice per chunk.

No docrooter structure is used. The text is pymupdf's ``page.get_text()`` for every page, split
into chunks of about ``chunk_tokens`` estimated tokens (words x 1.3). At most ``max_chunks``
evenly spaced chunks are sent, each as one Jev ``Choice`` over the taxonomy's leaves (candidate
text = full label path plus definition), and the per-chunk probabilities are averaged.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from docrooter.backends.base import Candidate, candidate_text
from docrooter.backends.jev import JevBackend
from docrooter.taxonomy import Taxonomy

TOKENS_PER_WORD = 1.3
CHUNK_TOKENS = 1500
MAX_CHUNKS = 8
INSTRUCTION = "What type of document is this?"
_WORD = re.compile(r"\S+\s*")


def estimate_tokens(text: str) -> int:
    """Words x 1.3, rounded."""
    return round(len(text.split()) * TOKENS_PER_WORD)


def chunk_indices(n: int, k: int) -> list[int]:
    """``k`` evenly spaced indexes into ``n`` chunks, first and last included (all when n <= k)."""
    if n <= k:
        return list(range(n))
    if k <= 1:
        return [0] if k == 1 else []
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def select_chunks(
    text: str, *, chunk_tokens: int = CHUNK_TOKENS, max_chunks: int = MAX_CHUNKS
) -> list[str]:
    """Split ``text`` into consecutive chunks of ``chunk_tokens / 1.3`` words (whitespace kept)
    and return at most ``max_chunks`` of them, evenly spaced."""
    words = _WORD.findall(text.strip())
    per_chunk = max(1, round(chunk_tokens / TOKENS_PER_WORD))
    chunks = ["".join(words[i : i + per_chunk]).strip() for i in range(0, len(words), per_chunk)]
    return [chunks[i] for i in chunk_indices(len(chunks), max_chunks)]


def mean_probs(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Per-key mean of probability dicts (a missing key counts as 0)."""
    if not rows:
        return {}
    keys = list(dict.fromkeys(k for row in rows for k in row))
    return {k: sum(row.get(k, 0.0) for row in rows) / len(rows) for k in keys}


def page_text(pdf_path: str | Path) -> str:
    """All pages' ``get_text()`` joined by newlines."""
    try:
        import pymupdf
    except ImportError:  # older installs only ship the fitz name
        import fitz as pymupdf  # type: ignore[no-redef]
    with pymupdf.open(str(pdf_path)) as doc:
        return "\n".join(page.get_text() for page in doc)


class _UsageClient:
    """Wraps an ``AsyncTypeSafeClient`` and records each response's input tokens by state."""

    def __init__(self, inner: Any, sink: list[tuple[str, int | None]]) -> None:
        self._inner = inner
        self._sink = sink

    async def __aenter__(self) -> _UsageClient:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self._inner.__aexit__(*exc)

    async def system_one(self, **kwargs: Any) -> Any:
        response = await self._inner.system_one(**kwargs)
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "input_tokens", None)
        state = kwargs.get("state")
        key = state.get("section", "") if isinstance(state, dict) else str(state)
        self._sink.append((key, int(tokens) if tokens is not None else None))
        return response


class UsageJevBackend(JevBackend):
    """``JevBackend`` that keeps ``(state text, input tokens)`` for every request in ``usage``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.usage: list[tuple[str, int | None]] = []

    def _client(self, sdk: Any) -> Any:
        return _UsageClient(super()._client(sdk), self.usage)

    def tokens_for(self, texts: Sequence[str]) -> int | None:
        """Summed SDK-reported input tokens of the requests whose state is one of ``texts``
        (``None`` when any of them reported none)."""
        wanted = set(texts)
        found = [t for key, t in self.usage if key in wanted]
        if not found or any(t is None for t in found):
            return None
        return sum(t for t in found if t is not None)


def leaf_candidates(taxonomy: Taxonomy) -> list[Candidate]:
    """One candidate per leaf: full label path plus definition (as docrooter's Flat sends)."""
    return [
        Candidate(n.id, candidate_text(n, taxonomy=taxonomy, full_path=True), tuple(n.examples))
        for n in taxonomy.leaves()
    ]


class ChunkResult(NamedTuple):
    leaf: str | None
    probs: dict[str, float]
    n_tokens: int
    tokens_estimated: bool
    extract_s: float
    classify_s: float
    n_chunks: int


def classify(
    pdf_path: str | Path,
    taxonomy: Taxonomy,
    *,
    backend: JevBackend,
    chunk_tokens: int = CHUNK_TOKENS,
    max_chunks: int = MAX_CHUNKS,
    instruction: str = INSTRUCTION,
) -> ChunkResult:
    """Classify one PDF from its raw text; ``leaf`` is ``None`` when the PDF has no text.

    ``n_tokens`` is the SDK-reported input token count when ``backend`` is a
    :class:`UsageJevBackend` that got usage back, else an estimate (words x 1.3 of each chunk,
    the instruction and every option text), flagged by ``tokens_estimated``.
    """
    start = time.perf_counter()
    chunks = select_chunks(page_text(pdf_path), chunk_tokens=chunk_tokens, max_chunks=max_chunks)
    extract_s = time.perf_counter() - start
    if not chunks:
        return ChunkResult(None, {}, 0, False, extract_s, 0.0, 0)
    candidates = leaf_candidates(taxonomy)
    if isinstance(backend, UsageJevBackend):
        backend.usage.clear()
    start = time.perf_counter()
    scores = backend.score(chunks, candidates, instruction=instruction)
    classify_s = time.perf_counter() - start
    rows = [
        {c.node_id: float(p) for c, p in zip(candidates, row, strict=True)} for row in scores.values
    ]
    probs = mean_probs(rows)
    leaf = max(probs, key=lambda k: probs[k])
    reported = backend.tokens_for(chunks) if isinstance(backend, UsageJevBackend) else None
    if reported is not None:
        n_tokens, estimated = reported, False
    else:
        per_request = estimate_tokens(" ".join([instruction, *(c.text for c in candidates)]))
        n_tokens = sum(estimate_tokens(c) + per_request for c in chunks)
        estimated = True
    return ChunkResult(leaf, probs, n_tokens, estimated, extract_s, classify_s, len(chunks))
