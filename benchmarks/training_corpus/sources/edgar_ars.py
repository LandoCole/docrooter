"""SEC EDGAR Form ARS: glossy annual reports furnished as PDF (research brief §3.1).

Listing: EDGAR full-text search (``efts.sec.gov/LATEST/search-index``, 100 hits a page) for
``forms=ARS`` in the March-April proxy-season peak of 2026, 2025 and 2024. Only a filing's main
document is kept (``file_type`` equal to the form, so no exhibits), only if it is a ``.pdf``
(some ARS main documents are HTML), and one per accession. The PDF lives at
``www.sec.gov/Archives/edgar/data/<cik>/<accession without dashes>/<file>``.

EDGAR allows 10 requests/s; we use at most 5 (one bucket for all SEC hosts).
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Iterator
from typing import Any

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, safe_id

NAME = "edgar_ars"
LICENCE = "SEC public filing (Form ARS); kept locally for research, never redistributed"
FTS = "https://efts.sec.gov/LATEST/search-index"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
WINDOWS = (("2026-03-01", "2026-04-30"), ("2025-03-01", "2025-04-30"), ("2024-03-01", "2024-04-30"))
PAGE = 100
MAX_FROM = 10_000  # full-text search does not page past 10,000 hits


def search_url(form: str, query: str, start: str, end: str, offset: int) -> str:
    params = {
        "q": query,
        "forms": form,
        "dateRange": "custom",
        "startdt": start,
        "enddt": end,
        "from": str(offset),
    }
    return f"{FTS}?{urllib.parse.urlencode(params)}"


def archive_url(cik: str, accession: str, filename: str) -> str:
    return f"{ARCHIVES}/{int(cik)}/{accession.replace('-', '')}/{filename}"


def company_name(display_name: str) -> str:
    """``"WATSCO INC  (WSO, WSO-B)  (CIK 0000105016)"`` -> ``"WATSCO INC"``."""
    return display_name.split("  (", 1)[0].split(" (CIK", 1)[0].strip()


def parse_hits(payload: dict[str, Any], form: str = "ARS") -> list[Candidate]:
    """The PDF main documents among one page of full-text-search hits, one per accession."""
    out: list[Candidate] = []
    seen: set[str] = set()
    for hit in payload.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        accession, _, filename = str(hit.get("_id", "")).partition(":")
        if source.get("file_type") != form or not filename.lower().endswith(".pdf"):
            continue
        ciks = source.get("ciks") or []
        if not accession or not ciks or accession in seen:
            continue
        seen.add(accession)
        names = source.get("display_names") or [""]
        company = company_name(str(names[0]))
        period = str(source.get("period_ending") or "")[:4]
        out.append(
            Candidate(
                id=safe_id("ars", accession),
                url=archive_url(str(ciks[0]), accession, filename),
                source=NAME,
                licence=LICENCE,
                doc_type_hint="financial.annual_report",
                title=f"{company} annual report {period}".strip(),
                publisher=company,
                extra={
                    "accession": accession,
                    "cik": str(ciks[0]),
                    "filed": source.get("file_date"),
                },
            )
        )
    return out


def paged_search(net: Net, form: str, query: str, parse: Any) -> Iterator[Candidate]:
    seen: set[str] = set()
    for start, end in WINDOWS:
        offset = 0
        while offset < MAX_FROM:
            payload = json.loads(net.listing(search_url(form, query, start, end, offset)))
            hits = payload.get("hits", {}).get("hits", [])
            for candidate in parse(payload):
                if candidate.id not in seen:
                    seen.add(candidate.id)
                    yield candidate
            total = int(payload.get("hits", {}).get("total", {}).get("value", 0))
            offset += PAGE
            if not hits or offset >= total:
                break


def candidates(net: Net) -> Iterator[Candidate]:
    return paged_search(net, "ARS", '"annual report"', parse_hits)


SOURCE = Source(
    name=NAME,
    target=300,
    cap=450,
    probe_url=f"{ARCHIVES}/",
    candidates=candidates,
)
