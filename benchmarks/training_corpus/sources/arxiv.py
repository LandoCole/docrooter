"""arXiv papers under CC BY 4.0 or CC0 only, licence read from metadata (research brief §3.5).

Listing: OAI-PMH ``ListRecords`` in the ``arXiv`` metadata format (the only one carrying the
licence), following ``resumptionToken``, for the finance-flavoured sets ``q-fin``, ``econ`` and
``stat`` plus some ``cs`` for diversity, interleaved. A record is kept only when its
``<license>`` is exactly the CC BY 4.0 or CC0 1.0 URI (http or https): no arXiv default licence,
no share-alike, no non-commercial variants, no missing licence. Authors are kept in the manifest
for attribution.

Limits: one request every 3 s across arXiv's machines (OAI-PMH), and the arxiv.org robots.txt
crawl delay of 15 s for PDFs (``https://arxiv.org/pdf/<id>``).
"""

from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source, round_robin, safe_id

NAME = "arxiv"
OAI = "https://oaipmh.arxiv.org/oai"
PDF = "https://arxiv.org/pdf"
LICENCES = {
    "creativecommons.org/licenses/by/4.0/": "CC-BY-4.0",
    "creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
}
# (set, from, until): finance first, a little cs for layout diversity.
HARVESTS = (
    ("q-fin", "2025-01-01", "2025-06-30"),
    ("econ", "2025-01-01", "2025-06-30"),
    ("stat", "2025-01-01", "2025-01-31"),
    ("cs", "2025-01-01", "2025-01-03"),
)
MAX_PAGES = 20

_OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
_ARXIV_NS = "{http://arxiv.org/OAI/arXiv/}"


def licence_name(uri: str) -> str | None:
    """``CC-BY-4.0`` or ``CC0-1.0`` for those licence URIs (http or https), else ``None``."""
    uri = uri.strip()
    for scheme in ("http://", "https://"):
        if uri.startswith(scheme):
            return LICENCES.get(uri[len(scheme) :])
    return None


def first_url(set_spec: str, start: str, end: str) -> str:
    query = {"verb": "ListRecords", "metadataPrefix": "arXiv", "set": set_spec}
    query.update({"from": start, "until": end})
    return f"{OAI}?{urllib.parse.urlencode(query)}"


def next_url(token: str) -> str:
    return f"{OAI}?{urllib.parse.urlencode({'verb': 'ListRecords', 'resumptionToken': token})}"


def parse_page(xml: bytes) -> tuple[list[Candidate], str | None]:
    """The kept records of one ListRecords response, and its resumption token (if any)."""
    root = ET.fromstring(xml)
    out = []
    for meta in root.iter(f"{_ARXIV_NS}arXiv"):
        arxiv_id = (meta.findtext(f"{_ARXIV_NS}id") or "").strip()
        licence = licence_name(meta.findtext(f"{_ARXIV_NS}license") or "")
        if not arxiv_id or licence is None:
            continue
        authors = []
        for author in meta.iter(f"{_ARXIV_NS}author"):
            names = [
                (author.findtext(f"{_ARXIV_NS}forenames") or "").strip(),
                (author.findtext(f"{_ARXIV_NS}keyname") or "").strip(),
            ]
            authors.append(" ".join(n for n in names if n))
        title = " ".join((meta.findtext(f"{_ARXIV_NS}title") or "").split())
        out.append(
            Candidate(
                id=safe_id("arxiv", arxiv_id),
                url=f"{PDF}/{arxiv_id}",
                source=NAME,
                licence=licence,
                doc_type_hint="research.research_paper",
                title=title,
                publisher="arXiv",
                extra={
                    "arxiv_id": arxiv_id,
                    "authors": authors,
                    "categories": (meta.findtext(f"{_ARXIV_NS}categories") or "").split(),
                },
            )
        )
    token_el = root.find(f".//{_OAI_NS}resumptionToken")
    token = (token_el.text or "").strip() if token_el is not None else ""
    return out, token or None


def harvest(net: Net, set_spec: str, start: str, end: str) -> Iterator[Candidate]:
    url: str | None = first_url(set_spec, start, end)
    for _ in range(MAX_PAGES):
        if url is None:
            return
        records, token = parse_page(net.listing(url))
        yield from records
        url = next_url(token) if token else None


def candidates(net: Net) -> Iterator[Candidate]:
    seen: set[str] = set()
    for candidate in round_robin(*(harvest(net, *h) for h in HARVESTS)):
        if candidate.id not in seen:  # cross-listed papers appear in several sets
            seen.add(candidate.id)
            yield candidate


SOURCE = Source(
    name=NAME,
    target=150,
    cap=250,
    probe_url=f"{PDF}/",
    candidates=candidates,
)
