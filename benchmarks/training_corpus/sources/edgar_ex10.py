"""SEC EDGAR exhibit-10 material contracts filed as PDF (spec §5, "Contracts").

Listing: EDGAR full-text search for ``"agreement"`` in 8-K, 10-K and 10-Q filings of the same
windows as Form ARS, keeping hits whose ``file_type`` is an ``EX-10`` exhibit and whose file is a
``.pdf``. EDGAR's official exhibit format is HTML or text, so PDF exhibits (unofficial copies) are
rare: the yield is expected to be low, and shortfalls are made up from other sources.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, safe_id
from benchmarks.training_corpus.sources.edgar_ars import (
    ARCHIVES,
    archive_url,
    company_name,
    paged_search,
)

NAME = "edgar_ex10"
LICENCE = "SEC public filing (exhibit 10); kept locally for research, never redistributed"
FORMS = "8-K,10-K,10-Q"


def parse_hits(payload: dict[str, Any]) -> list[Candidate]:
    """The PDF exhibit-10 documents among one page of full-text-search hits."""
    out: list[Candidate] = []
    for hit in payload.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        accession, _, filename = str(hit.get("_id", "")).partition(":")
        file_type = str(source.get("file_type") or "")
        if not file_type.startswith("EX-10") or not filename.lower().endswith(".pdf"):
            continue
        ciks = source.get("ciks") or []
        if not accession or not ciks:
            continue
        names = source.get("display_names") or [""]
        company = company_name(str(names[0]))
        description = str(source.get("file_description") or file_type)
        out.append(
            Candidate(
                id=safe_id("ex10", f"{accession}-{filename.rsplit('.', 1)[0]}"),
                url=archive_url(str(ciks[0]), accession, filename),
                source=NAME,
                licence=LICENCE,
                doc_type_hint="legal.contract",
                title=f"{company} {description}".strip(),
                publisher=company,
                extra={"accession": accession, "cik": str(ciks[0]), "exhibit": file_type},
            )
        )
    return out


def candidates(net: Net) -> Iterator[Candidate]:
    return paged_search(net, FORMS, '"agreement"', parse_hits)


SOURCE = Source(
    name=NAME,
    target=50,
    cap=75,
    probe_url=f"{ARCHIVES}/",
    candidates=candidates,
)
