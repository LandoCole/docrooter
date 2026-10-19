"""The training-corpus sources, and the order shortfalls are made up in.

``ALL_SOURCES`` run by default and add up to the spec's 1,100 documents. The EDGAR sources
(``OPTIONAL_SOURCES``) are kept but run only when named with ``--sources``: www.sec.gov refuses
the project User-Agent, so by controller ruling company IR annual reports replace Form ARS and
CUAD contracts replace exhibit-10 PDFs.
"""

from __future__ import annotations

from benchmarks.training_corpus.sources import (
    arxiv,
    crs,
    cuad,
    edgar_ars,
    edgar_ex10,
    fed,
    fund_docs,
    govinfo_gao,
    ir_annual_reports,
    rfc,
    uscourts,
)
from benchmarks.training_corpus.sources.base import Candidate, Source

ALL_SOURCES: tuple[Source, ...] = (
    ir_annual_reports.SOURCE,
    fund_docs.SOURCE,
    govinfo_gao.SOURCE,
    crs.SOURCE,
    fed.SOURCE,
    arxiv.SOURCE,
    uscourts.SOURCE,
    rfc.SOURCE,
    cuad.SOURCE,
)
OPTIONAL_SOURCES: tuple[Source, ...] = (edgar_ars.SOURCE, edgar_ex10.SOURCE)

# Finance first, so a shortfall keeps the corpus finance-heavy; RFCs (technical) last.
MAKEUP_ORDER: tuple[str, ...] = (
    "edgar_ars",
    "ir_annual_reports",
    "fund_docs",
    "fed",
    "edgar_ex10",
    "cuad",
    "crs",
    "govinfo_gao",
    "arxiv",
    "uscourts",
    "rfc",
)

__all__ = ["ALL_SOURCES", "MAKEUP_ORDER", "OPTIONAL_SOURCES", "Candidate", "Source"]
