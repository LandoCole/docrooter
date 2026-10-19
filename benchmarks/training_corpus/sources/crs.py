"""Congressional Research Service reports via the Congress.gov API (research brief §3.2).

Listing: ``api.congress.gov/v3/crsreport`` (250 a request, newest first), ``contentType``
``Reports`` only, active or archived (no two-page "In Focus" resources or blog posts). The
PDF is the public product link
``www.congress.gov/crs_external_products/<prefix>/PDF/<id>/<id>.<version>.pdf``, where ``prefix``
is the id's letters (``R``, ``RL``, ``RS``); www.congress.gov asks for a 2 s crawl delay, which
robots.txt handling applies.

Key: ``CONGRESS_API_KEY`` or ``DATA_GOV_API_KEY`` from the process environment, else
``DEMO_KEY`` (api.data.gov's demo key, accepted by api.congress.gov; one request lists 250).
"""

from __future__ import annotations

import json
import re
import urllib.parse
from collections.abc import Iterator

from benchmarks.training_corpus._net import DEMO_KEY, Net
from benchmarks.training_corpus.sources.base import Candidate, Source, env_key, safe_id
from benchmarks.training_corpus.sources.govinfo_gao import with_key

NAME = "crs"
LICENCE = (
    "US government work (CRS report; copyright status unverified, may contain third-party "
    "material); not redistributed"
)
API = "https://api.congress.gov/v3/crsreport"
PRODUCTS = "https://www.congress.gov/crs_external_products"
LIMIT = 250
MAX_PAGES = 3


def api_key() -> str:
    return env_key("CONGRESS_API_KEY", "DATA_GOV_API_KEY") or DEMO_KEY


def product_pdf(report_id: str, version: int) -> str:
    prefix = re.match(r"[A-Za-z]+", report_id)
    letters = prefix.group(0).upper() if prefix else report_id
    return f"{PRODUCTS}/{letters}/PDF/{report_id}/{report_id}.{version}.pdf"


def list_url(offset: int) -> str:
    query = urllib.parse.urlencode({"offset": offset, "limit": LIMIT, "format": "json"})
    return f"{API}?{query}"


def parse_reports(payload: dict) -> list[Candidate]:
    out = []
    for item in payload.get("CRSReports", []):
        report_id = str(item.get("id", ""))
        if item.get("contentType") != "Reports":
            continue
        if not re.fullmatch(r"[A-Z]{1,3}\d{3,6}", report_id):
            continue
        version = int(item.get("version") or 1)
        out.append(
            Candidate(
                id=safe_id("crs", report_id),
                url=product_pdf(report_id, version),
                source=NAME,
                licence=LICENCE,
                doc_type_hint="government.government_report",
                title=str(item.get("title") or ""),
                publisher="Congressional Research Service",
                extra={
                    "version": version,
                    "publish_date": item.get("publishDate"),
                    "status": item.get("status"),
                },
            )
        )
    return out


def candidates(net: Net) -> Iterator[Candidate]:
    key = api_key()
    for page in range(MAX_PAGES):
        payload = json.loads(net.listing(with_key(list_url(page * LIMIT), key)))
        reports = payload.get("CRSReports", [])
        yield from parse_reports(payload)
        if len(reports) < LIMIT:
            break


SOURCE = Source(
    name=NAME,
    target=100,
    cap=200,
    probe_url=f"{PRODUCTS}/",
    candidates=candidates,
)
