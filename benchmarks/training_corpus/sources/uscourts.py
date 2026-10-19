"""govinfo USCOURTS: federal appellate opinions, in place of CourtListener (research brief §3.6).

Listing: one govinfo ``search`` request (POST, up to 1,000 results) for
``collection:(USCOURTS) AND courttype:(Appellate)``, newest first. Appellate granules are
opinions; district-court packages are mostly orders. Results are interleaved across circuits so
no one court dominates. The PDF is the public granule content link
``www.govinfo.gov/content/pkg/<package>/pdf/<granule>.pdf``, which needs no key.

Key: as for GAO (``GOVINFO_API_KEY`` / ``DATA_GOV_API_KEY``, else ``DEMO_KEY``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, round_robin, safe_id
from benchmarks.training_corpus.sources.govinfo_gao import (
    API,
    CONTENT,
    api_key,
    content_pdf,
    with_key,
)

NAME = "uscourts"
LICENCE = "US federal court opinion (government work; verify per document); not redistributed"
QUERY = {
    "query": "collection:(USCOURTS) AND courttype:(Appellate)",
    "pageSize": 1000,
    "offsetMark": "*",
    "sorts": [{"field": "publishdate", "sortOrder": "DESC"}],
}


def parse_results(payload: dict) -> list[Candidate]:
    """Opinion candidates, interleaved across courts (each court's results keep their order)."""
    by_court: dict[str, list[Candidate]] = defaultdict(list)
    for result in payload.get("results", []):
        package_id = str(result.get("packageId", ""))
        granule_id = str(result.get("granuleId") or "")
        if not package_id.startswith("USCOURTS-") or not granule_id:
            continue
        court = package_id.split("-")[1] if package_id.count("-") >= 2 else "?"
        authors = result.get("governmentAuthor") or []
        by_court[court].append(
            Candidate(
                id=safe_id("uscourts", granule_id.removeprefix("USCOURTS-")),
                url=content_pdf(package_id, granule_id),
                source=NAME,
                licence=LICENCE,
                doc_type_hint="legal.court_opinion",
                title=str(result.get("title") or ""),
                publisher=str(authors[0]) if authors else "United States Courts",
                extra={"granule_id": granule_id, "date_issued": result.get("dateIssued")},
            )
        )
    return list(round_robin(*(by_court[court] for court in sorted(by_court))))


def candidates(net: Net) -> Iterator[Candidate]:
    body = json.dumps(QUERY, sort_keys=True).encode()
    payload = json.loads(net.listing(with_key(f"{API}/search", api_key()), body=body))
    yield from parse_results(payload)


SOURCE = Source(
    name=NAME,
    target=100,
    cap=150,
    probe_url=f"{CONTENT}/",
    candidates=candidates,
)
