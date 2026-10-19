"""The training-corpus sources, in spec §5 order, and the order shortfalls are made up in."""

from __future__ import annotations

from benchmarks.training_corpus.sources import (
    arxiv,
    crs,
    edgar_ars,
    edgar_ex10,
    fed,
    fund_docs,
    govinfo_gao,
    rfc,
    uscourts,
)
from benchmarks.training_corpus.sources.base import Candidate, Source

ALL_SOURCES: tuple[Source, ...] = (
    edgar_ars.SOURCE,
    fund_docs.SOURCE,
    govinfo_gao.SOURCE,
    crs.SOURCE,
    fed.SOURCE,
    arxiv.SOURCE,
    uscourts.SOURCE,
    rfc.SOURCE,
    edgar_ex10.SOURCE,
)

# Finance first, so a shortfall keeps the corpus finance-heavy; RFCs (technical) last.
MAKEUP_ORDER: tuple[str, ...] = (
    "edgar_ars",
    "fund_docs",
    "fed",
    "edgar_ex10",
    "crs",
    "govinfo_gao",
    "arxiv",
    "uscourts",
    "rfc",
)

__all__ = ["ALL_SOURCES", "MAKEUP_ORDER", "Candidate", "Source"]
