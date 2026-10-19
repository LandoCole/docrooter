"""govinfo GAOREPORTS: Government Accountability Office reports (research brief §3.2).

Listing: the govinfo ``published`` endpoint for collection ``GAOREPORTS`` issued 2004-2008 (the
born-digital end of the collection; earlier reports are more often scans), up to 1,000 packages
a request, ``docClass`` ``REPORT`` only (no Comptroller General decisions). The PDF is the public
content link ``www.govinfo.gov/content/pkg/<package>/pdf/<package>.pdf``, which needs no key.

The listing needs an api.data.gov key: ``GOVINFO_API_KEY`` or ``DATA_GOV_API_KEY`` from the
process environment, else ``DEMO_KEY`` (documented by govinfo for trying the API; its hourly limit
fits, because one request lists 1,000 packages).
"""

from __future__ import annotations

import json
import random
import urllib.parse
from collections.abc import Iterator

from benchmarks.training_corpus._net import DEMO_KEY, Net
from benchmarks.training_corpus.sources.base import Candidate, Source, env_key, safe_id

NAME = "govinfo_gao"
LICENCE = (
    "US government work: GAO products are not protected by copyright in the US "
    "(third-party images excepted); not redistributed"
)
API = "https://api.govinfo.gov"
CONTENT = "https://www.govinfo.gov/content/pkg"
WINDOW = ("2004-01-01", "2008-12-31")
PAGE_SIZE = 1000
MAX_PAGES = 3


def api_key() -> str:
    return env_key("GOVINFO_API_KEY", "DATA_GOV_API_KEY") or DEMO_KEY


def with_key(url: str, key: str) -> str:
    joiner = "&" if "?" in url else "?"
    return f"{url}{joiner}{urllib.parse.urlencode({'api_key': key})}"


def content_pdf(package_id: str, granule_id: str | None = None) -> str:
    return f"{CONTENT}/{package_id}/pdf/{granule_id or package_id}.pdf"


def published_url(collection: str, start: str, end: str) -> str:
    query = urllib.parse.urlencode(
        {"offsetMark": "*", "pageSize": PAGE_SIZE, "collection": collection}
    )
    return f"{API}/published/{start}/{end}?{query}"


def parse_packages(payload: dict) -> list[Candidate]:
    out = []
    for package in payload.get("packages", []):
        package_id = str(package.get("packageId", ""))
        if package.get("docClass") != "REPORT" or not package_id.startswith("GAOREPORTS-"):
            continue
        out.append(
            Candidate(
                id=safe_id("gao", package_id.removeprefix("GAOREPORTS-")),
                url=content_pdf(package_id),
                source=NAME,
                licence=LICENCE,
                doc_type_hint="government.government_report",
                title=str(package.get("title") or ""),
                publisher="U.S. Government Accountability Office",
                extra={"package_id": package_id, "date_issued": package.get("dateIssued")},
            )
        )
    return out


def candidates(net: Net) -> Iterator[Candidate]:
    key = api_key()
    url: str | None = published_url("GAOREPORTS", *WINDOW)
    for _ in range(MAX_PAGES):
        if not url:
            break
        payload = json.loads(net.listing(with_key(url, key)))
        page = parse_packages(payload)
        random.Random(f"0:{NAME}:{url}").shuffle(page)  # spread each page across years
        yield from page
        url = payload.get("nextPage")


SOURCE = Source(
    name=NAME,
    target=150,
    cap=250,
    probe_url=f"{CONTENT}/",
    candidates=candidates,
)
