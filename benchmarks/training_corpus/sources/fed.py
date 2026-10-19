"""Federal Reserve Board publications (research brief §3.3).

There is no document API and the RSS feeds hold only the latest 15 items, so the listing reads
the Board's own index pages (robots.txt allows them; 1 request/s):

- FEDS working papers: ``/econres/feds/<year>.htm``; an entry ``id="2025110"`` is FEDS 2025-110,
  whose PDF is ``/econres/feds/files/2025110pap.pdf``;
- IFDP papers: ``/econres/ifdp/<year>.htm``; ``id="20251429"`` is IFDP 1429, PDF
  ``/econres/ifdp/files/ifdp1429.pdf``;
- FOMC minutes: the PDF links on ``/monetarypolicy/fomccalendars.htm``.

The three streams are interleaved. Doc types follow the series (controller ruling, L5 fix 1):
FEDS and IFDP working papers are research papers (element family ``research_paper``); FOMC
minutes and every other Board report (Monetary Policy Report, Financial Stability Report,
testimony) are government reports (family ``generic``). See :func:`doc_type_for_series`.

Licence: public domain unless otherwise indicated (Board disclaimer); seals, logos and marked
third-party material excepted.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, round_robin

NAME = "fed"
LICENCE = (
    "Federal Reserve Board publication: public domain unless otherwise indicated; not redistributed"
)
BASE = "https://www.federalreserve.gov"
YEARS = (2025, 2024, 2023, 2022)
PUBLISHER = "Board of Governors of the Federal Reserve System"

_ENTRY = re.compile(
    r'id="(?P<num>\d{7,8})".*?<h5><a href="(?P<href>[^"]+)">(?P<title>.*?)</a></h5>', re.S
)
_MINUTES = re.compile(r'href="(/monetarypolicy/files/fomcminutes(\d{8})\.pdf)"')


RESEARCH_SERIES = frozenset({"FEDS", "IFDP"})


def doc_type_for_series(series: str) -> str:
    """The doc type of a Board publication series: the FEDS and IFDP working-paper series are
    ``research.research_paper``; any other series (FOMC minutes, the Monetary Policy Report,
    the Financial Stability Report, testimony) is ``government.government_report``."""
    if series.upper() in RESEARCH_SERIES:
        return "research.research_paper"
    return "government.government_report"


def _title(raw: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", raw)).split())


def parse_series(page: str, series: str) -> list[Candidate]:
    """FEDS (``series="feds"``) or IFDP (``"ifdp"``) entries of one year page."""
    out = []
    for match in _ENTRY.finditer(page):
        num = match.group("num")
        if series == "feds" and len(num) == 7:
            url = f"{BASE}/econres/feds/files/{num}pap.pdf"
            doc_id = f"fed-feds-{num}"
        elif series == "ifdp" and len(num) == 8:
            url = f"{BASE}/econres/ifdp/files/ifdp{int(num[4:])}.pdf"
            doc_id = f"fed-ifdp-{int(num[4:])}"
        else:
            continue
        out.append(
            Candidate(
                id=doc_id,
                url=url,
                source=NAME,
                licence=LICENCE,
                doc_type_hint=doc_type_for_series(series),
                title=_title(match.group("title")),
                publisher=PUBLISHER,
                extra={"series": series.upper()},
            )
        )
    return out


def parse_minutes(page: str) -> list[Candidate]:
    out, seen = [], set()
    for path, date in _MINUTES.findall(page):
        if date in seen:
            continue
        seen.add(date)
        day = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        out.append(
            Candidate(
                id=f"fed-fomcminutes-{date}",
                url=f"{BASE}{path}",
                source=NAME,
                licence=LICENCE,
                doc_type_hint=doc_type_for_series("FOMC minutes"),
                title=f"Minutes of the Federal Open Market Committee {day}",
                publisher=PUBLISHER,
                extra={"series": "FOMC minutes"},
            )
        )
    return sorted(out, key=lambda c: c.id, reverse=True)


def _series(net: Net, series: str) -> Iterator[Candidate]:
    for year in YEARS:
        page = net.listing(f"{BASE}/econres/{series}/{year}.htm").decode("utf-8", "replace")
        yield from parse_series(page, series)


def _minutes(net: Net) -> Iterator[Candidate]:
    page = net.listing(f"{BASE}/monetarypolicy/fomccalendars.htm").decode("utf-8", "replace")
    yield from parse_minutes(page)


def candidates(net: Net) -> Iterator[Candidate]:
    return round_robin(_series(net, "feds"), _series(net, "ifdp"), _minutes(net))


SOURCE = Source(
    name=NAME,
    target=60,
    cap=220,
    probe_url=f"{BASE}/econres/",
    candidates=candidates,
)
