"""Training-corpus fetcher (v0.5 L4): exclusion, rate limits, licence and document filters,
splits and resume. Offline: every response comes from a fake transport; no PDF is real except
the one test that runs the isolated page counter on a PDF it makes itself."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from email.message import Message
from pathlib import Path

import pytest

from benchmarks.training_corpus import _net, fetch
from benchmarks.training_corpus.sources import (
    ALL_SOURCES,
    MAKEUP_ORDER,
    OPTIONAL_SOURCES,
    arxiv,
    cuad,
    edgar_ars,
    ir_annual_reports,
)
from benchmarks.training_corpus.sources.base import Candidate, Source
from docrooter.taxonomies import family_for_doc_type

UA = "docrooter-research/0.5 (research; contact via repository)"


def _pdf(tag: str) -> bytes:
    return b"%PDF-1.7\n% " + tag.encode() + b"\n" + b"0" * 64 + b"\n%%EOF\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- fakes ---------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Every test is offline: a real request would fail the test, not reach a server. Only
    loopback (a local ``http.server`` fixture on 127.0.0.1) is allowed."""
    real_open = urllib.request.OpenerDirector.open

    def guarded(self, fullurl, *args, **kwargs):
        url = fullurl if isinstance(fullurl, str) else fullurl.full_url
        if urllib.parse.urlsplit(url).hostname == "127.0.0.1":
            return real_open(self, fullurl, *args, **kwargs)
        raise AssertionError("unit tests must not use the network")

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", guarded)


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.t += seconds


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = io.BytesIO(body)
        self.status = 200
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value

    def read(self, n: int = -1) -> bytes:
        return self._body.read(n)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class FakeTransport:
    """``routes``: url -> body bytes, or (status, body, headers). Unknown URLs are 404s.
    Records (url, time, user agent, method) for every request."""

    def __init__(self, routes: dict, clock: FakeClock | None = None) -> None:
        self.routes = routes
        self.clock = clock or FakeClock()
        self.calls: list[tuple[str, float, str, str]] = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.calls.append((url, self.clock(), request.get_header("User-agent"), request.method))
        route = self.routes.get(url)
        if route is None:
            raise urllib.error.HTTPError(url, 404, "Not Found", Message(), None)
        status, body, headers = route if isinstance(route, tuple) else (200, route, {})
        if status >= 300:
            message = Message()
            for key, value in headers.items():
                message[key] = value
            raise urllib.error.HTTPError(url, status, "error", message, None)
        return FakeResponse(body, headers)

    def urls(self) -> list[str]:
        return [call[0] for call in self.calls]


def make_net(tmp_path: Path, routes: dict, clock: FakeClock | None = None):
    clock = clock or FakeClock()
    transport = FakeTransport(routes, clock)
    limiter = _net.RateLimiter(clock=clock, sleep=clock.sleep)
    net = _net.Net(tmp_path / "listings", transport=transport, limiter=limiter, log=lambda m: None)
    return net, transport


def fake_source(name: str, candidates: list[Candidate], target: int = 10, cap: int = 20):
    return Source(
        name=name,
        target=target,
        cap=cap,
        probe_url=f"https://{name}.example.org/",
        candidates=lambda net: iter(candidates),
    )


def cand(source: str, key: str, title: str = "", url: str | None = None) -> Candidate:
    return Candidate(
        id=f"{source}-{key}",
        url=url or f"https://{source}.example.org/{key}.pdf",
        source=source,
        licence="test licence",
        doc_type_hint="government.government_report",
        title=title,
        publisher="Test",
    )


def fake_inspect(path: Path) -> fetch.PdfInfo:
    return fetch.PdfInfo(pages=4, text_chars=5000)


def make_fetcher(tmp_path, sources, target, net, exclusions=None, **kwargs):
    return fetch.Fetcher(
        sources,
        target,
        net=net,
        exclusions=exclusions or fetch.Exclusions(),
        cache_dir=tmp_path / "cache",
        manifest_path=tmp_path / "manifest.jsonl",
        inspect=kwargs.pop("inspect", fake_inspect),
        log=kwargs.pop("log", lambda m: None),
        now=lambda: "2026-10-19T00:00:00Z",
        **kwargs,
    )


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --- Review Focus 1: exclusion -------------------------------------------------------------


def test_exclusion_by_hash_url_and_title(tmp_path, monkeypatch):
    dev_bytes, heldout_bytes, clean_bytes = _pdf("dev"), _pdf("heldout"), _pdf("clean")
    dev = tmp_path / "corpus" / "manifest.jsonl"
    dev.parent.mkdir()
    dev.write_text(
        json.dumps(
            {
                "id": "doc_001",
                "file": "pdfs/doc_001.pdf",
                "url": "https://www.example.com/reports/acme-2023.pdf",
                "title": "Acme Corp. 2023 Annual Report",
                "publisher": "Acme Corp.",
                "label": "financial.annual_report",
                "sha256": _sha(dev_bytes),
            }
        )
        + "\n"
    )
    heldout_dir = tmp_path / "heldout"
    (heldout_dir / "pdfs").mkdir(parents=True)
    (heldout_dir / "pdfs" / "h_001.pdf").write_bytes(heldout_bytes)  # must never be opened
    heldout = heldout_dir / "manifest.jsonl"
    heldout.write_text(
        json.dumps(
            {
                "id": "h_001",
                "file": "pdfs/h_001.pdf",
                "url": "https://heldout.example.net/docs/secret.pdf",
                "title": "Secret Held-Out Prospectus",
                "sha256": _sha(heldout_bytes),
            }
        )
        + "\n"
    )

    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(io, "open", spy_open)
    # Python 3.10's pathlib opens through an accessor bound at import time, which the spies
    # above do not reach; spy on Path.open as well so every version records the same opens.
    real_path_open = Path.open

    def spy_path_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy_path_open)

    exclusions = fetch.Exclusions.from_manifests({"dev": dev, "heldout": heldout})
    # Only the three fields are kept.
    assert set(exclusions.sha256) == {_sha(dev_bytes), _sha(heldout_bytes)}
    assert set(exclusions.urls) == {
        "example.com/reports/acme-2023.pdf",
        "heldout.example.net/docs/secret.pdf",
    }
    assert set(exclusions.titles) == {"acme corp 2023 annual report", "secret held out prospectus"}

    candidates = [
        # the dev URL, spelt differently: dropped before download
        cand("src", "dev_url", url="http://example.com/reports/acme-2023.pdf/#page=2"),
        # the dev title, different case and punctuation: dropped before download
        cand("src", "dev_title", title="ACME Corp 2023 annual report!"),
        # held-out bytes re-hosted at a new URL under a new title: dropped by sha256
        cand("src", "rehosted", title="Some Fund Summary"),
        # the held-out URL with a tracking query
        cand("src", "heldout_url", url="https://heldout.example.net/docs/secret.pdf?utm=x"),
        # dev bytes re-hosted
        cand("src", "dev_bytes"),
        cand("src", "clean", title="A Clean Report"),
    ]
    routes = {
        "https://src.example.org/rehosted.pdf": heldout_bytes,
        "https://src.example.org/dev_bytes.pdf": dev_bytes,
        "https://src.example.org/clean.pdf": clean_bytes,
    }
    net, transport = make_net(tmp_path, routes)
    logs: list[str] = []
    fetcher = make_fetcher(
        tmp_path, [fake_source("src", candidates)], 6, net, exclusions, log=logs.append
    )
    report = fetcher.run()

    rows = read_manifest(tmp_path / "manifest.jsonl")
    assert [r["id"] for r in rows] == ["src-clean"]
    # Held-out drops are a bare count: which field matched is not recorded either.
    assert report.drops == Counter({"dev:url": 1, "dev:title": 1, "heldout": 2, "dev:sha256": 1})
    # URL and title matches never reach the network.
    downloaded = transport.urls()
    assert "http://example.com/reports/acme-2023.pdf/#page=2" not in downloaded
    assert "https://src.example.org/dev_title.pdf" not in downloaded
    assert not any("heldout.example.net" in u for u in downloaded)
    # Re-hosted bytes are deleted, and held-out drops do not record the URL or title.
    pdfs = sorted(p.name for p in (tmp_path / "cache" / "pdfs").iterdir())
    assert pdfs == ["src-clean.pdf"]
    drops_text = (tmp_path / "cache" / "drops.jsonl").read_text()
    drops = [json.loads(line) for line in drops_text.splitlines()]
    assert all(drop["permanent"] is True for drop in drops)
    heldout_rows = [d for d in drops if "key" in d]
    assert len(heldout_rows) == 2 and all(set(d) == {"key", "permanent"} for d in heldout_rows)
    for secret in ("src-rehosted", "src-heldout_url", "heldout.example.net", "Some Fund Summary"):
        assert secret not in drops_text
        assert not any(secret in line for line in logs)  # never logged one by one
    # A re-run skips the held-out drops without any request.
    net2, transport2 = make_net(tmp_path, {})
    make_fetcher(tmp_path, [fake_source("src", candidates)], 6, net2, exclusions).run()
    assert transport2.urls() == ["https://src.example.org/robots.txt"]
    # Under the held-out directory only the manifest file was opened.
    under_heldout = {p for p in opened if str(heldout_dir) in p}
    assert under_heldout == {str(heldout)}


def test_default_exclusions_read_the_dev_and_heldout_manifests_only():
    assert fetch.DEV_MANIFEST.name == "manifest.jsonl"
    assert fetch.DEV_MANIFEST.parent.name == "corpus"
    assert fetch.HELDOUT_MANIFEST.name == "manifest.jsonl"
    assert fetch.HELDOUT_MANIFEST.parent.name == "heldout"
    assert fetch.EXCLUSION_FIELDS == ("sha256", "url", "title")


# --- rate limits, robots, user agent -------------------------------------------------------


def test_rate_limiter_respects_per_source_limits(tmp_path):
    clock = FakeClock()
    oai = "https://oaipmh.arxiv.org/oai?verb=ListRecords&resumptionToken="
    routes = {f"https://efts.sec.gov/LATEST/search-index?from={i}": b"{}" for i in range(12)}
    routes.update({f"{oai}{i}": b"<x/>" for i in range(3)})
    routes["https://arxiv.org/robots.txt"] = b"User-agent: *\nCrawl-delay: 15\nAllow: /pdf\n"
    routes.update({f"https://arxiv.org/pdf/2501.0000{i}": _pdf(str(i)) for i in range(2)})
    demo = "https://api.govinfo.gov/published/2004-01-01/2008-12-31?api_key=DEMO_KEY&p="
    routes.update({f"{demo}{i}": b"{}" for i in range(2)})
    routes["https://api.congress.gov/v3/crsreport?api_key=DEMO_KEY"] = b"{}"
    routes["https://www.federalreserve.gov/robots.txt"] = b"User-agent: *\nDisallow: /private/\n"
    routes.update({f"https://www.federalreserve.gov/econres/{i}.htm": b"x" for i in range(3)})
    net, transport = make_net(tmp_path, routes, clock)
    net.cache_dir = None  # every call reaches the transport

    for i in range(12):
        net.listing(f"https://efts.sec.gov/LATEST/search-index?from={i}")
    for i in range(3):
        net.listing(f"{oai}{i}")
    for i in range(2):
        net.download_pdf(f"https://arxiv.org/pdf/2501.0000{i}", tmp_path / f"a{i}.pdf")
    for i in range(2):
        net.listing(f"{demo}{i}")
    net.listing("https://api.congress.gov/v3/crsreport?api_key=DEMO_KEY")
    for i in range(3):
        net.listing(f"https://www.federalreserve.gov/econres/{i}.htm")

    def times(pred):
        return [t for url, t, _, _ in transport.calls if pred(url)]

    def gaps(ts):
        return [b - a for a, b in zip(ts, ts[1:], strict=False)]

    sec = times(lambda u: ".sec.gov" in u)
    assert len(sec) == 12 and min(gaps(sec)) >= 0.2 - 1e-9
    assert all(sum(1 for t in sec if start <= t < start + 1.0) <= 5 for start in sec)
    arxiv_all = times(lambda u: "arxiv.org" in u)
    assert min(gaps(arxiv_all)) >= 3.0 - 1e-9  # one request per 3 s across arXiv hosts
    pdf_times = times(lambda u: u.startswith("https://arxiv.org/pdf/"))
    assert len(pdf_times) == 2 and gaps(pdf_times)[0] >= 15.0 - 1e-9  # robots crawl delay
    demo_times = times(lambda u: "DEMO_KEY" in u)  # both APIs share the demo key's budget
    assert len(demo_times) == 3 and min(gaps(demo_times)) >= _net.DEMO_KEY_INTERVAL - 1e-9
    fed = times(lambda u: "federalreserve.gov/econres" in u)
    assert min(gaps(fed)) >= 1.0 - 1e-9
    # robots.txt once per web host, none for API hosts; one fixed User-Agent everywhere.
    robots = [u for u in transport.urls() if u.endswith("/robots.txt")]
    assert sorted(robots) == [
        "https://arxiv.org/robots.txt",
        "https://www.federalreserve.gov/robots.txt",
    ]
    assert {call[2] for call in transport.calls} == {UA}
    assert _net.USER_AGENT == UA and "@" not in UA


def test_blocked_host_is_not_retried(tmp_path):
    routes = {
        "https://www.sec.gov/robots.txt": (403, b"no", {}),
        "https://ok.example.org/a.pdf": _pdf("a"),
    }
    net, transport = make_net(tmp_path, routes)
    blocked = Source("blocked", 5, 10, "https://www.sec.gov/Archives/", lambda n: iter([]))
    ok = fake_source("ok", [cand("ok", "a")], target=5, cap=10)
    report = make_fetcher(tmp_path, [blocked, ok], 2, net).run()
    assert report.sources["blocked"].status == "unavailable"
    assert "403" in report.sources["blocked"].reason
    assert transport.urls().count("https://www.sec.gov/robots.txt") == 1
    assert not any("sec.gov/Archives" in u for u in transport.urls())

    routes = {
        "https://x.example.org/1.pdf": (429, b"", {}),
        "https://x.example.org/2.pdf": _pdf("2"),
    }
    net, transport = make_net(tmp_path, routes)
    with pytest.raises(_net.HostBlocked):
        net.download_pdf("https://x.example.org/1.pdf", tmp_path / "1.pdf")
    with pytest.raises(_net.HostBlocked):
        net.download_pdf("https://x.example.org/2.pdf", tmp_path / "2.pdf")
    assert transport.urls().count("https://x.example.org/2.pdf") == 0


def test_robots_rules_and_listing_cache_without_key(tmp_path):
    robots = _net.parse_robots(
        "User-agent: Brightbot\nDisallow: /\n\nUser-agent: *\nDisallow: /*?truepdf*\n"
        "Disallow: /*.dl$\nAllow: /private/ok\nDisallow: /private\nCrawl-delay: 2\n"
    )
    assert robots.crawl_delay == 2
    assert robots.allows("https://h/us/literature/fact-sheet/x.pdf")
    assert not robots.allows("https://h/us/x.pdf?truepdf=1")
    assert not robots.allows("https://h/file.dl")
    assert robots.allows("https://h/file.dl.pdf")
    assert not robots.allows("https://h/private/x")
    assert robots.allows("https://h/private/ok/x")
    assert not _net.parse_robots("User-agent: docrooter-research\nDisallow: /\n").allows(
        "https://h/"
    )

    url = "https://api.govinfo.gov/published/a?api_key=SECRET"
    net, transport = make_net(tmp_path, {url: b"listing"})
    assert net.listing(url) == b"listing"
    net2, transport2 = make_net(tmp_path, {})
    assert net2.listing("https://api.govinfo.gov/published/a?api_key=OTHER") == b"listing"
    assert transport2.calls == []
    cached = b"".join(p.read_bytes() + p.name.encode() for p in (tmp_path / "listings").iterdir())
    assert b"SECRET" not in cached
    assert _net.redact(url) == "https://api.govinfo.gov/published/a?api_key=***"


def test_download_checks_magic_and_size_cap(tmp_path):
    routes = {
        "https://h.example.org/html.pdf": b"<html>not a pdf</html>",
        "https://h.example.org/big.pdf": (200, _pdf("x"), {"Content-Length": str(51 << 20)}),
        "https://h.example.org/stream.pdf": b"%PDF-1.4\n" + b"0" * 4096,
        "https://h.example.org/ok.pdf": _pdf("ok"),
    }
    net, _ = make_net(tmp_path, routes)
    for name in ("html", "big"):
        with pytest.raises(_net.Rejected):
            net.download_pdf(f"https://h.example.org/{name}.pdf", tmp_path / f"{name}.pdf")
    with pytest.raises(_net.Rejected):
        net.download_pdf("https://h.example.org/stream.pdf", tmp_path / "s.pdf", max_bytes=1024)
    with pytest.raises(_net.Rejected):
        net.download_pdf("https://h.example.org/missing.pdf", tmp_path / "m.pdf")
    assert net.download_pdf("https://h.example.org/ok.pdf", tmp_path / "ok.pdf") == len(_pdf("ok"))
    assert sorted(p.name for p in tmp_path.glob("*.pdf*")) == ["ok.pdf"]  # no partial files
    assert _net.MAX_PDF_BYTES == 50 * 1024 * 1024


# --- source filters ------------------------------------------------------------------------

_OAI_PAGE_1 = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
%s
<resumptionToken cursor="0" completeListSize="7">tok|1001</resumptionToken>
</ListRecords></OAI-PMH>"""
_OAI_PAGE_2 = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
%s
<resumptionToken cursor="6" completeListSize="7"></resumptionToken>
</ListRecords></OAI-PMH>"""


def _record(arxiv_id: str, licence: str | None) -> bytes:
    lic = f"<license>{licence}</license>" if licence else ""
    return (
        f"""<record><header><identifier>oai:arXiv.org:{arxiv_id}</identifier></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>{arxiv_id}</id>
<authors><author><keyname>Doe</keyname><forenames>Jane</forenames></author></authors>
<title>Paper {arxiv_id}</title><categories>q-fin.ST stat.ML</categories>{lic}
</arXiv></metadata></record>"""
    ).encode()


def test_arxiv_licence_filter_keeps_only_cc_by_and_cc0(tmp_path):
    page1 = _OAI_PAGE_1 % b"".join(
        [
            _record("2501.00001", "http://creativecommons.org/licenses/by/4.0/"),
            _record("2501.00002", "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
            _record("2501.00003", "http://creativecommons.org/licenses/by-sa/4.0/"),
            _record("2501.00004", "http://creativecommons.org/licenses/by-nc-sa/4.0/"),
            _record("2501.00005", "http://creativecommons.org/publicdomain/zero/1.0/"),
            _record("2501.00006", None),
        ]
    )
    page2 = _OAI_PAGE_2 % b"".join(
        [
            _record("2501.00007", "https://creativecommons.org/licenses/by/4.0/"),
            _record("2501.00008", "http://creativecommons.org/licenses/by-nc-nd/4.0/"),
        ]
    )
    first = arxiv.first_url("q-fin", "2025-01-01", "2025-06-30")
    routes = {first: page1, arxiv.next_url("tok|1001"): page2}
    net, transport = make_net(tmp_path, routes)
    kept = list(arxiv.harvest(net, "q-fin", "2025-01-01", "2025-06-30"))
    assert [(c.id, c.licence) for c in kept] == [
        ("arxiv-2501_00001", "CC-BY-4.0"),
        ("arxiv-2501_00005", "CC0-1.0"),
        ("arxiv-2501_00007", "CC-BY-4.0"),
    ]
    assert kept[0].url == "https://arxiv.org/pdf/2501.00001"
    assert kept[0].extra["authors"] == ["Jane Doe"]
    assert kept[0].doc_type_hint == "research.research_paper"
    assert transport.urls() == [first, arxiv.next_url("tok|1001")]
    assert arxiv.licence_name("http://creativecommons.org/licenses/by/4.0") is None


def _hit(adsh: str, filename: str, file_type: str, cik: str = "0000105016") -> dict:
    return {
        "_id": f"{adsh}:{filename}",
        "_source": {
            "ciks": [cik],
            "adsh": adsh,
            "form": "ARS",
            "root_forms": ["ARS"],
            "file_type": file_type,
            "file_date": "2025-04-25",
            "period_ending": "2024-12-31",
            "display_names": ["WATSCO INC  (WSO, WSO-B)  (CIK 0000105016)"],
        },
    }


def test_edgar_ars_keeps_pdf_main_documents_only():
    payload = {
        "hits": {
            "total": {"value": 5},
            "hits": [
                _hit("0001193125-25-095167", "d946364dars.pdf", "ARS"),
                _hit("0001770450-25-000018", "xeroxars2024.htm", "ARS"),  # HTML main document
                _hit("0001193125-25-000001", "ex99.pdf", "EX-99.1"),  # a PDF exhibit
                _hit("0001193125-25-095167", "d946364dars2.pdf", "ARS"),  # same accession again
                _hit("0000927628-25-065538", "D819004DARS.PDF", "ARS", cik="0000927628"),
            ],
        }
    }
    kept = edgar_ars.parse_hits(payload)
    assert [c.url for c in kept] == [
        "https://www.sec.gov/Archives/edgar/data/105016/000119312525095167/d946364dars.pdf",
        "https://www.sec.gov/Archives/edgar/data/927628/000092762825065538/D819004DARS.PDF",
    ]
    assert kept[0].id == "ars-0001193125-25-095167"
    assert kept[0].doc_type_hint == "financial.annual_report"
    assert kept[0].publisher == "WATSCO INC"
    assert kept[0].title == "WATSCO INC annual report 2024"


# --- splits, quotas, make-up ---------------------------------------------------------------


def test_split_assignment_deterministic_and_stratified():
    records = [{"id": f"a-{i:04d}", "source": "a"} for i in range(1000)]
    records += [{"id": f"b-{i:04d}", "source": "b"} for i in range(1000)]
    splits = fetch.assign_splits(records)
    assert splits == fetch.assign_splits(list(reversed(records)))
    assert all(splits[r["id"]] == fetch.split_for(r["id"]) for r in records)
    # Realised shares within 3 points of 85/5/10 per source, on 1,000 synthetic ids each.
    for source in ("a", "b"):
        rows = [{**r, "split": splits[r["id"]]} for r in records if r["source"] == source]
        percent = fetch.split_shares(rows)[source]["percent"]
        assert abs(percent["train"] - 85) <= 3
        assert abs(percent["calib"] - 5) <= 3
        assert abs(percent["gold"] - 10) <= 3
    # Adding or removing documents never changes any other document's split.
    grown = fetch.assign_splits(records + [{"id": f"c-{i}", "source": "a"} for i in range(300)])
    assert all(grown[k] == v for k, v in splits.items())
    shrunk = fetch.assign_splits(records[::3])
    assert all(shrunk[k] == splits[k] for k in shrunk)
    # A split already in the manifest is never re-assigned.
    flipped = {"a-0000": "gold" if splits["a-0000"] != "gold" else "train"}
    assert fetch.assign_splits(records, flipped)["a-0000"] == flipped["a-0000"]


def test_manifest_splits_are_sticky_across_runs(tmp_path):
    candidates = [cand("src", str(i)) for i in range(6)]
    net, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    make_fetcher(tmp_path, [fake_source("src", candidates[:3])], 3, net).run()
    manifest = tmp_path / "manifest.jsonl"
    rows = read_manifest(manifest)
    rows[0]["split"] = "gold" if rows[0]["split"] != "gold" else "calib"  # as if assigned earlier
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    net2, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    make_fetcher(tmp_path, [fake_source("src", candidates)], 6, net2).run()
    after = {r["id"]: r["split"] for r in read_manifest(manifest)}
    assert after[rows[0]["id"]] == rows[0]["split"]
    assert all(after[r["id"]] == r["split"] for r in rows)
    assert all(after[c.id] == fetch.split_for(c.id) for c in candidates[3:])


def test_quotas_scale_to_target_and_sources_match_the_spec():
    assert {s.name: s.target for s in ALL_SOURCES} == {
        "ir_annual_reports": 250,
        "fund_docs": 150,
        "govinfo_gao": 150,
        "crs": 100,
        "fed": 60,
        "arxiv": 150,
        "uscourts": 100,
        "rfc": 40,
        "cuad": 100,
    }
    assert sum(s.target for s in ALL_SOURCES) >= 1100
    assert sorted(MAKEUP_ORDER) == sorted(s.name for s in ALL_SOURCES + OPTIONAL_SOURCES)
    for target in (10, 30, 1100, 1500):
        quotas = fetch.scaled_quotas(ALL_SOURCES, target)
        assert sum(quotas.values()) == target
    assert fetch.scaled_quotas(ALL_SOURCES, 1100)["ir_annual_reports"] == 250
    assert all(s.cap >= s.target for s in ALL_SOURCES)


def test_shortfall_made_up_from_other_sources(tmp_path):
    routes = {"https://www.sec.gov/robots.txt": (403, b"", {})}
    a = [cand("aa", str(i)) for i in range(10)]
    routes.update({c.url: _pdf(c.id) for c in a})
    net, _ = make_net(tmp_path, routes)
    blocked = Source("edgar_ars", 5, 10, "https://www.sec.gov/Archives/", lambda n: iter([]))
    # caps are at the 1,100-document scale: 880 scales to 8 at a target of 10
    fund = Source("fund_docs", 5, 880, "https://aa.example.org/", lambda n: iter(a))
    report = make_fetcher(tmp_path, [blocked, fund], 10, net).run()
    assert report.sources["edgar_ars"].shortfall == 5
    assert report.sources["fund_docs"].have == 8  # its quota 5, made up to its scaled cap 8
    assert report.sources["fund_docs"].made_up == 3
    assert len(read_manifest(tmp_path / "manifest.jsonl")) == 8


# --- resume --------------------------------------------------------------------------------


def test_resume_skips_fetched(tmp_path):
    candidates = [cand("src", str(i), title=f"Report {i}") for i in range(4)]
    routes = {c.url: _pdf(c.id) for c in candidates[:3]}
    routes[candidates[3].url] = b"<html>gone</html>"  # a permanent rejection
    net, transport = make_net(tmp_path, routes)
    source = fake_source("src", candidates, target=4, cap=4)
    make_fetcher(tmp_path, [source], 4, net).run()
    first = read_manifest(tmp_path / "manifest.jsonl")
    assert [r["id"] for r in first] == ["src-0", "src-1", "src-2"]
    assert {r["split"] for r in first} <= {"train", "calib", "gold"}
    assert all(r["sha256"] == _sha(_pdf(r["id"])) and r["pages"] == 4 for r in first)

    # A crash after a PDF reached its final name but before its journal line, plus a torn
    # journal tail and a leftover .part file.
    pdf_dir = tmp_path / "cache" / "pdfs"
    extra = cand("src", "4", title="Report 4")
    (pdf_dir / "src-4.pdf").write_bytes(_pdf("src-4"))
    (pdf_dir / "src-5.pdf.part").write_bytes(b"%PDF-partial")
    with (tmp_path / "cache" / "fetched.jsonl").open("a") as f:
        f.write('{"id": "src-torn", "sou')

    net2, transport2 = make_net(tmp_path, {})  # any download would 404
    source2 = fake_source("src", candidates + [extra], target=5, cap=5)
    inspected: list[str] = []

    def counting_inspect(path):
        inspected.append(path.name)
        return fake_inspect(path)

    make_fetcher(tmp_path, [source2], 5, net2, inspect=counting_inspect).run()
    second = read_manifest(tmp_path / "manifest.jsonl")
    assert [r["id"] for r in second] == ["src-0", "src-1", "src-2", "src-4"]
    # fetched, adopted and permanently dropped documents: no download, only robots.txt
    assert transport2.urls() == ["https://src.example.org/robots.txt"]
    assert inspected == ["src-4.pdf"]


def test_dry_run_downloads_nothing(tmp_path):
    candidates = [cand("src", str(i)) for i in range(5)]
    net, transport = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    report = make_fetcher(tmp_path, [fake_source("src", candidates)], 3, net, dry_run=True).run()
    assert [c.id for c in report.chosen] == ["src-0", "src-1", "src-2"]
    assert not any(u.endswith(".pdf") for u in transport.urls())
    assert not (tmp_path / "manifest.jsonl").exists()
    assert (tmp_path / "cache" / "dry_run.jsonl").read_text().count("\n") == 3


def test_scanned_pdf_is_dropped(tmp_path):
    candidates = [cand("src", "scan"), cand("src", "native")]
    net, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})

    def inspect(path):
        return fetch.PdfInfo(pages=10, text_chars=0 if "scan" in path.name else 9000)

    report = make_fetcher(tmp_path, [fake_source("src", candidates)], 2, net, inspect=inspect).run()
    assert [r["id"] for r in report.records] == ["src-native"]
    assert report.drops["no-text-layer"] == 1


def test_pdf_meta_runs_isolated(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for i in range(3):
        doc.new_page().insert_text((72, 72), f"Native text on page {i + 1}. " * 20)
    path = tmp_path / "native.pdf"
    doc.save(str(path))
    calls = []
    real_popen = fetch.subprocess.Popen

    def spy_popen(cmd, **kwargs):
        calls.append(cmd)
        return real_popen(cmd, **kwargs)

    monkeypatch.setattr(fetch.subprocess, "Popen", spy_popen)
    info = fetch.inspect_pdf(path)
    assert info.pages == 3 and info.text_chars > fetch.MIN_TEXT_CHARS and not info.encrypted
    assert calls[0][1] == "-I"
    out = fetch.subprocess.run(
        [sys.executable, "-I", str(fetch.PDF_META), str(path)], capture_output=True, text=True
    )
    assert isinstance(json.loads(out.stdout)["memory_cap"], bool)  # True where the OS enforces it
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.4 garbage")
    with pytest.raises(fetch.InspectError):
        fetch.inspect_pdf(tmp_path / "bad.pdf")


def test_listing_parsers_for_the_other_sources():
    from benchmarks.training_corpus.sources import crs, fed, fund_docs, govinfo_gao, uscourts

    page = (
        '<div id="2025110"><time>December 2025</time>'
        '<h5><a href="/econres/feds/x.htm">Liquidity &amp; Labor</a></h5></div>'
        '<div id="20251429"><h5><a href="/econres/ifdp/y.htm">Dollarization Waves </a></h5></div>'
    )
    feds = fed.parse_series(page, "feds")
    assert [(c.id, c.url, c.title) for c in feds] == [
        (
            "fed-feds-2025110",
            "https://www.federalreserve.gov/econres/feds/files/2025110pap.pdf",
            "Liquidity & Labor",
        )
    ]
    assert [c.url for c in fed.parse_series(page, "ifdp")] == [
        "https://www.federalreserve.gov/econres/ifdp/files/ifdp1429.pdf"
    ]
    minutes = fed.parse_minutes('<a href="/monetarypolicy/files/fomcminutes20250129.pdf">x</a>')
    assert minutes[0].doc_type_hint == "government.government_report"

    reports = crs.parse_reports(
        {
            "CRSReports": [
                {"id": "R43657", "contentType": "Reports", "status": "Active", "version": 3},
                {"id": "RL33110", "contentType": "Reports", "status": "Archived", "version": 12},
                {"id": "IF13328", "contentType": "Resources", "status": "Active", "version": 1},
                {"id": "IN12747", "contentType": "Posts", "status": "Active", "version": 1},
            ]
        }
    )
    assert [c.url for c in reports] == [
        "https://www.congress.gov/crs_external_products/R/PDF/R43657/R43657.3.pdf",
        "https://www.congress.gov/crs_external_products/RL/PDF/RL33110/RL33110.12.pdf",
    ]

    gao = govinfo_gao.parse_packages(
        {
            "packages": [
                {"packageId": "GAOREPORTS-GAO-08-705", "docClass": "REPORT", "title": "DOD"},
                {"packageId": "GAOREPORTS-B-1", "docClass": "COMPTROLLERDECISION"},
            ]
        }
    )
    assert [(c.id, c.url) for c in gao] == [
        (
            "gao-gao-08-705",
            "https://www.govinfo.gov/content/pkg/GAOREPORTS-GAO-08-705/pdf/GAOREPORTS-GAO-08-705.pdf",
        )
    ]

    results = [
        {"packageId": f"USCOURTS-{court}-{n}", "granuleId": f"USCOURTS-{court}-{n}-0"}
        for court, n in (("ca5", 1), ("ca5", 2), ("ca5", 3), ("ca9", 1), ("caDC", 1))
    ]
    opinions = uscourts.parse_results({"results": results})
    assert [c.id for c in opinions][:3] == [
        "uscourts-ca5-1-0",
        "uscourts-ca9-1-0",
        "uscourts-cadc-1-0",
    ]
    assert opinions[0].url.endswith("/USCOURTS-ca5-1/pdf/USCOURTS-ca5-1-0.pdf")

    funds = list(fund_docs.candidates(None))  # an explicit list: no network
    assert len(funds) == fund_docs.SOURCE.cap >= 150
    assert len({c.id for c in funds}) == len(funds)
    assert {c.url.split("/")[2] for c in funds} == {"www.ssga.com", "www.ishares.com"}
    assert [c.extra["kind"] for c in funds[:2]] == ["fact sheet", "summary prospectus"]
    assert {c.extra["kind"] for c in funds} == {"fact sheet", "summary prospectus"}
    prospectuses = [c for c in funds if c.extra["kind"] == "summary prospectus"]
    assert len(prospectuses) >= 50
    assert all(c.doc_type_hint == "financial.prospectus" for c in prospectuses)
    assert all(family_for_doc_type(c.doc_type_hint) == "fund_document" for c in funds)


def test_default_transport_passes_the_timeout_by_keyword(monkeypatch):
    seen = {}

    def fake_urlopen(request, data=None, timeout=None):
        seen.update(data=data, timeout=timeout)
        return FakeResponse(b"ok")

    net = _net.Net(None, log=lambda m: None)
    monkeypatch.setattr(net.opener, "open", fake_urlopen)
    assert net.listing("https://api.govinfo.gov/x?api_key=K") == b"ok"
    assert seen == {"data": None, "timeout": _net.TIMEOUT_SECONDS}


def test_a_second_host_blocking_does_not_stop_the_source(tmp_path):
    main = [cand("src", str(i)) for i in range(3)]
    other = Candidate(**{**cand("src", "x").__dict__, "url": "https://other.example.org/x.pdf"})
    routes = {c.url: _pdf(c.id) for c in main}
    routes[other.url] = (403, b"", {})
    net, transport = make_net(tmp_path, routes)
    source = fake_source("src", [main[0], other, main[1], main[2]], target=3, cap=3)
    report = make_fetcher(tmp_path, [source], 3, net).run()
    assert report.sources["src"].status == "ok"
    assert [r["id"] for r in report.records] == ["src-0", "src-1", "src-2"]
    assert transport.urls().count(other.url) == 1


# --- fix round 1 ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [503, 429, 403])
def test_unreadable_robots_on_a_second_host_is_transient(tmp_path, status):
    main = [cand("src", str(i)) for i in range(2)]
    other = [
        Candidate(**{**cand("src", f"o{i}").__dict__, "url": f"https://other.example.org/{i}.pdf"})
        for i in range(2)
    ]
    routes = {c.url: _pdf(c.id) for c in main + other}
    routes["https://other.example.org/robots.txt"] = (status, b"", {})
    net, transport = make_net(tmp_path, routes)
    source = fake_source("src", [main[0], other[0], other[1], main[1]], target=4, cap=4)
    report = make_fetcher(tmp_path, [source], 4, net).run()
    assert [r["id"] for r in report.records] == ["src-0", "src-1"]
    assert report.sources["src"].status == "ok"
    drops_path = tmp_path / "cache" / "drops.jsonl"
    drops = (
        [json.loads(x) for x in drops_path.read_text().splitlines()] if drops_path.exists() else []
    )
    assert not any(d["permanent"] for d in drops if d.get("id", "").startswith("src-o"))
    assert not any(
        u.startswith("https://other.example.org/") and u.endswith(".pdf") for u in transport.urls()
    )
    assert transport.urls().count("https://other.example.org/robots.txt") == 1  # once per run

    # The next run asks again and fetches them once robots.txt is readable.
    routes["https://other.example.org/robots.txt"] = b"User-agent: *\nAllow: /\n"
    net2, _ = make_net(tmp_path, routes)
    report2 = make_fetcher(tmp_path, [source], 4, net2).run()
    assert sorted(r["id"] for r in report2.records) == ["src-0", "src-1", "src-o0", "src-o1"]


def test_redirects_are_vetted_like_first_requests(tmp_path):
    clock = FakeClock()
    routes = {
        "https://b.example.org/robots.txt": b"User-agent: *\nDisallow: /private/\nCrawl-delay: 5\n",
    }
    net, transport = make_net(tmp_path, routes, clock)
    # https -> http and non-http schemes are refused.
    for target in ("http://a.example.org/x.pdf", "ftp://a.example.org/x.pdf"):
        with pytest.raises(_net.Rejected):
            net.check_redirect("https://a.example.org/x.pdf", target)
    # robots.txt of the target host applies.
    with pytest.raises(_net.Rejected):
        net.check_redirect("https://a.example.org/x.pdf", "https://b.example.org/private/x.pdf")
    # Each hop is sent through the target's rate limit and crawl delay.
    routes.update(
        {
            "https://a.example.org/x.pdf": (302, b"", {"Location": "https://b.example.org/ok/1"}),
            "https://a.example.org/y.pdf": (302, b"", {"Location": "https://b.example.org/ok/2"}),
            "https://b.example.org/ok/1": _pdf("1"),
            "https://b.example.org/ok/2": _pdf("2"),
        }
    )
    net.download_pdf("https://a.example.org/x.pdf", tmp_path / "1.pdf")
    net.download_pdf("https://a.example.org/y.pdf", tmp_path / "2.pdf")
    hops = [t for u, t, _, _ in transport.calls if u.startswith("https://b.example.org/ok/")]
    assert hops[1] - hops[0] >= 5.0 - 1e-9
    # A blocked host stays blocked behind a redirect.
    net._blocked["c.example.org"] = "blocked"
    with pytest.raises(_net.HostBlocked):
        net.check_redirect("https://a.example.org/x.pdf", "https://c.example.org/x.pdf")
    # urllib never follows redirects itself: the fetcher's bounded loop does.
    real = _net.Net(None, log=lambda m: None)
    assert any(isinstance(h, _net._NoRedirects) for h in real.opener.handlers)


def test_unsafe_ids_are_refused_by_the_fetcher(tmp_path):
    bad = [
        Candidate(**{**cand("src", "x").__dict__, "id": doc_id})
        for doc_id in ("../../etc/passwd", "/abs/x", "Upper-Case", "dots.in.id", "a" * 81, "")
    ]
    good = cand("src", "fine_1")
    net, transport = make_net(tmp_path, {good.url: _pdf(good.id)})
    report = make_fetcher(tmp_path, [fake_source("src", bad + [good])], 7, net).run()
    assert [r["id"] for r in report.records] == ["src-fine_1"]
    assert report.drops["unsafe-id"] == len(bad)
    assert transport.urls().count(good.url) == 1 and len(transport.urls()) == 2  # + robots
    assert [p.name for p in (tmp_path / "cache" / "pdfs").iterdir()] == ["src-fine_1.pdf"]
    from benchmarks.training_corpus.sources.base import safe_id

    for raw in ("../../etc/passwd", "2501.00001", "USCOURTS-caDC-25", "x" * 300):
        assert fetch.SAFE_ID.fullmatch(safe_id("p", raw))
    assert safe_id("p", "x" * 300) != safe_id("p", "x" * 299)  # shortening keeps ids distinct


def test_exclusion_is_rechecked_when_the_manifest_is_written(tmp_path):
    candidates = [cand("src", str(i), title=f"Report {i}") for i in range(3)]
    net, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    make_fetcher(tmp_path, [fake_source("src", candidates)], 3, net).run()
    assert len(read_manifest(tmp_path / "manifest.jsonl")) == 3
    # The dev manifest later gains one of these documents (by bytes) and another (by title).
    grown = fetch.Exclusions(
        sha256={_sha(_pdf("src-0")): "dev"}, titles={fetch.normalise_title("Report 1"): "dev"}
    )
    net2, _ = make_net(tmp_path, {})
    report = make_fetcher(tmp_path, [fake_source("src", candidates)], 3, net2, grown).run()
    assert [r["id"] for r in read_manifest(tmp_path / "manifest.jsonl")] == ["src-2"]
    assert report.drops["manifest-dev:sha256"] == 1 and report.drops["manifest-dev:title"] == 1


def test_inspect_kills_a_child_over_the_memory_cap(tmp_path, monkeypatch):
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(fetch, "PDF_META", slow)
    monkeypatch.setattr(fetch, "_rss_bytes", lambda pid: fetch.INSPECT_MEMORY_CAP + 1)
    with pytest.raises(fetch.InspectError, match="memory cap"):
        fetch.inspect_pdf(tmp_path / "x.pdf")


def test_demo_key_stops_when_the_hourly_budget_is_spent(tmp_path):
    url = "https://api.govinfo.gov/a?api_key=DEMO_KEY"
    routes = {url: (200, b"{}", {"X-RateLimit-Remaining": "0"})}
    net, transport = make_net(tmp_path, routes)
    net.cache_dir = None
    net.listing(url)
    with pytest.raises(_net.HostBlocked):
        net.listing("https://api.congress.gov/v3/crsreport?api_key=DEMO_KEY")
    assert len(transport.calls) == 1


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_cuad_lists_contract_pdfs_only_and_extracts_safely(tmp_path, monkeypatch):
    folder = "CUAD_v1/full_contract_pdf/Part_I/Distributor"
    members = {f"{folder}/ACME_01_01_2001-EX-10-{i} AGREEMENT.pdf": _pdf(str(i)) for i in range(5)}
    members["CUAD_v1/master_clauses.csv"] = b"Filename,Parties\n"
    members["CUAD_v1/CUAD_v1.json"] = b"{}"
    members["__MACOSX/CUAD_v1/full_contract_pdf/._x.pdf"] = b"junk"
    members["CUAD_v1/full_contract_pdf/Part_I/Bad/not-a-pdf.pdf"] = b"<html>"
    archive_dir = tmp_path / "archives"
    archive = _zip(tmp_path / "CUAD_v1.zip", members)
    (archive_dir / "cuad").mkdir(parents=True)
    target = archive_dir / "cuad" / cuad.ARCHIVE_NAME
    target.write_bytes(archive.read_bytes())
    monkeypatch.setattr(cuad, "ARCHIVE_SHA256", _sha(target.read_bytes()))
    net, transport = make_net(tmp_path, {})
    net.archive_dir = archive_dir
    found = list(cuad.candidates(net))
    assert transport.calls == []  # the cached archive is used
    assert len(found) == 6
    assert all(c.archive_member and "full_contract_pdf" in c.archive_member for c in found)
    assert not any("master_clauses" in (c.archive_member or "") for c in found)
    assert {c.licence for c in found} == {"CC-BY-4.0"}
    assert all("Atticus" in c.extra["attribution"] for c in found)
    assert all(c.doc_type_hint == "legal.contract" for c in found)
    assert all(family_for_doc_type(c.doc_type_hint) == "contract" for c in found)
    assert [c.archive_member for c in found] == cuad.sample(cuad.contract_members(target))
    # The fetcher reads members with the PDF checks; a non-PDF member is a permanent drop.
    source = Source("cuad", 6, 6, None, lambda n: iter(found))
    report = make_fetcher(tmp_path, [source], 6, net).run()
    assert len(report.records) == 5 and report.drops["rejected"] == 1
    assert all(r["licence"] == "CC-BY-4.0" and "attribution" in r for r in report.records)
    assert not any("archive" in k for r in report.records for k in r if k != "cuad_file")
    with pytest.raises(_net.Rejected):
        fetch.extract_pdf(target, "CUAD_v1/full_contract_pdf/missing.pdf", tmp_path / "m.pdf")
    first = next(iter(members))
    with pytest.raises(_net.Rejected):
        fetch.extract_pdf(target, first, tmp_path / "c.pdf", max_bytes=10)
    assert not list(tmp_path.glob("*.part"))


def test_ir_annual_reports_list_is_explicit_and_well_formed():
    found = list(ir_annual_reports.candidates(None))  # an explicit list: no network
    assert len(found) == ir_annual_reports.SOURCE.cap >= 200
    assert len({c.id for c in found}) == len(found)
    assert len({c.url for c in found}) == len(found)
    assert all(c.url.startswith("https://") for c in found)
    assert all(fetch.SAFE_ID.fullmatch(c.id) for c in found)
    assert all(c.doc_type_hint == "financial.annual_report" for c in found)
    assert all(family_for_doc_type(c.doc_type_hint) == "annual_report" for c in found)
    assert all(2019 <= c.extra["year"] <= 2025 for c in found)
    assert len({c.publisher for c in found}) >= 30
    assert "sec.gov" not in " ".join(c.url for c in found)
    # Interleaved by company, so a small quota still spans many companies.
    assert len({c.publisher for c in found[:30]}) >= 20


# --- fix round 2: redirects against a real urllib opener -------------------------------------


@pytest.fixture
def local_servers():
    """Two loopback ``http.server`` instances. The first has an allowing robots.txt, a
    ``/loop/robots.txt`` that redirects to itself forever, a 6-hop page redirect chain, a
    2-hop chain that ends in a PDF, and a plain PDF. The second is the same, except that its
    own ``/robots.txt`` redirects to itself forever (the production crash)."""
    import http.server
    import threading

    pdf = _pdf("local")

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output clean
            pass

        def _redirect(self, location):
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            path = self.path
            if path.startswith("/loop/robots.txt"):  # robots.txt redirects to itself forever
                n = int(path.rsplit("=", 1)[1]) + 1 if "=" in path else 1
                return self._redirect(f"/loop/robots.txt?n={n}")
            if path.startswith("/robots.txt") and self.server.robots_loop:  # type: ignore[attr-defined]
                n = int(path.rsplit("=", 1)[1]) + 1 if "=" in path else 1
                return self._redirect(f"/robots.txt?n={n}")
            if path == "/robots.txt":
                body = b"User-agent: *\nAllow: /\n"
            elif path.startswith("/hop/"):
                n = int(path.rsplit("/", 1)[1])
                return self._redirect(f"/hop/{n + 1}") if n < 6 else self._pdf()
            elif path.startswith("/short/"):
                n = int(path.rsplit("/", 1)[1])
                return self._redirect(f"/short/{n + 1}") if n < 2 else self._pdf()
            elif path == "/plain.pdf":
                return self._pdf(_pdf("plain"))
            elif path == "/to-cafe":  # the Location carries UTF-8 bytes, as servers send them
                return self._redirect("/café.pdf".encode().decode("latin-1"))
            elif urllib.parse.unquote(path) == "/café.pdf":
                return self._pdf(_pdf("cafe"))
            elif path == "/to-space":
                return self._redirect("/with space.pdf")
            elif urllib.parse.unquote(path) == "/with space.pdf":
                return self._pdf(_pdf("space"))
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _pdf(self, data=pdf):
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    servers = []
    for robots_loop in (False, True):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.robots_loop = robots_loop  # type: ignore[attr-defined]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
    try:
        yield tuple(f"http://127.0.0.1:{srv.server_address[1]}" for srv in servers)
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def test_redirect_loops_end_in_recorded_failures_with_a_real_opener(tmp_path, local_servers):
    local_server = local_servers[0]
    import sys as _sys

    clock = FakeClock()
    limiter = _net.RateLimiter(clock=clock, sleep=clock.sleep)  # no real waiting
    net = _net.Net(tmp_path / "listings", limiter=limiter, log=lambda m: None)  # real opener

    # A robots.txt redirect loop: robots.txt for the host is unreadable after 5 hops, which
    # is transient, and never recursive.
    loop_net = _net.Net(None, limiter=limiter, log=lambda m: None)
    robots = loop_net._fetch_robots(f"{local_server}/loop/robots.txt")
    assert robots.error is not None and "redirects" in robots.error

    # Pages: a 6-hop chain fails, a 2-hop chain and a plain file succeed, all with the real
    # urllib opener and under a low recursion limit (vetting must not recurse).
    pages = [
        Candidate(**{**cand("loc", "six").__dict__, "url": f"{local_server}/hop/0"}),
        Candidate(**{**cand("loc", "two").__dict__, "url": f"{local_server}/short/0"}),
        Candidate(**{**cand("loc", "plain").__dict__, "url": f"{local_server}/plain.pdf"}),
    ]
    source = Source("loc", 3, 3, None, lambda n: iter(pages))
    limit = _sys.getrecursionlimit()
    _sys.setrecursionlimit(400)
    try:
        report = make_fetcher(tmp_path, [source], 3, net).run()
    finally:
        _sys.setrecursionlimit(limit)
    assert sorted(r["id"] for r in report.records) == ["loc-plain", "loc-two"]
    drops = [json.loads(x) for x in (tmp_path / "cache" / "drops.jsonl").read_text().splitlines()]
    assert [(d["id"], d["permanent"]) for d in drops] == [("loc-six", False)]
    assert "more than 5 redirects" in drops[0]["reason"]


def test_robots_redirect_loop_makes_downloads_transient_failures(tmp_path, local_servers):
    """The production crash: a host whose own robots.txt redirects in a loop. The download is
    a recorded transient failure, a document on another host still arrives, the run ends."""
    good_server, loop_server = local_servers
    clock = FakeClock()
    limiter = _net.RateLimiter(clock=clock, sleep=clock.sleep)
    net = _net.Net(tmp_path / "listings", limiter=limiter, log=lambda m: None)  # real opener
    bad = Candidate(**{**cand("loc", "bad").__dict__, "url": f"{loop_server}/plain.pdf"})
    good = Candidate(**{**cand("loc", "good").__dict__, "url": f"{good_server}/plain.pdf"})
    source = Source("loc", 2, 2, None, lambda n: iter([bad, good]))
    report = make_fetcher(tmp_path, [source], 2, net).run()
    assert [r["id"] for r in report.records] == ["loc-good"]
    drops = [json.loads(x) for x in (tmp_path / "cache" / "drops.jsonl").read_text().splitlines()]
    assert [(d["id"], d["permanent"]) for d in drops] == [("loc-bad", False)]
    assert "robots.txt unavailable" in drops[0]["reason"] and "redirects" in drops[0]["reason"]


def test_recursion_error_in_one_document_is_a_transient_failure(tmp_path):
    candidates = [cand("src", "boom"), cand("src", "ok")]
    net, _ = make_net(tmp_path, {candidates[1].url: _pdf("ok")})
    fetcher = make_fetcher(tmp_path, [fake_source("src", candidates)], 2, net)
    real_take = fetcher._take

    def take(candidate, report):
        if candidate.id == "src-boom":
            raise RecursionError("maximum recursion depth exceeded")
        return real_take(candidate, report)

    fetcher._take = take
    report = fetcher.run()
    assert [r["id"] for r in report.records] == ["src-ok"]
    drops = [json.loads(x) for x in (tmp_path / "cache" / "drops.jsonl").read_text().splitlines()]
    assert [(d["id"], d["permanent"]) for d in drops] == [("src-boom", False)]


# --- fix round 3: quoted redirect targets, per-document isolation ----------------------------


def test_redirect_targets_are_quoted_like_urllib(tmp_path, local_servers):
    server = local_servers[0]
    clock = FakeClock()
    net = _net.Net(
        None, limiter=_net.RateLimiter(clock=clock, sleep=clock.sleep), log=lambda m: None
    )
    assert net.download_pdf(f"{server}/to-cafe", tmp_path / "cafe.pdf") == len(_pdf("cafe"))
    assert (tmp_path / "cafe.pdf").read_bytes() == _pdf("cafe")
    assert net.download_pdf(f"{server}/to-space", tmp_path / "space.pdf") == len(_pdf("space"))
    assert (tmp_path / "space.pdf").read_bytes() == _pdf("space")
    utf8_location = "/café.pdf".encode().decode("latin-1")  # as http.client decodes it
    assert _net.redirect_target("https://h/x", utf8_location) == "https://h/caf%C3%A9.pdf"
    assert _net.redirect_target("https://h/a/b", "c d.pdf") == "https://h/a/c%20d.pdf"
    assert _net.redirect_target("https://h/x", "https://o.example") == "https://o.example/"


def test_any_exception_in_one_document_is_recorded_and_the_run_continues(tmp_path):
    candidates = [cand("src", "boom"), cand("src", "ok")]
    net, _ = make_net(tmp_path, {candidates[1].url: _pdf("ok")})
    fetcher = make_fetcher(tmp_path, [fake_source("src", candidates)], 2, net)
    real_take = fetcher._take

    def take(candidate, report):
        if candidate.id == "src-boom":
            raise UnicodeEncodeError("ascii", "café", 3, 4, "x" * 500)
        return real_take(candidate, report)

    fetcher._take = take
    report = fetcher.run()
    assert [r["id"] for r in report.records] == ["src-ok"]
    drops = [json.loads(x) for x in (tmp_path / "cache" / "drops.jsonl").read_text().splitlines()]
    assert [(d["id"], d["permanent"]) for d in drops] == [("src-boom", False)]
    assert drops[0]["reason"].startswith("failed: UnicodeEncodeError: ")
    assert len(drops[0]["reason"]) < 300  # the message is truncated

    # KeyboardInterrupt and SystemExit still stop the run (and the manifest is still written).
    for stop in (KeyboardInterrupt, SystemExit):
        net2, _ = make_net(tmp_path, {})
        fetcher2 = make_fetcher(tmp_path, [fake_source("src", [cand("src", "new")])], 3, net2)

        def interrupt(candidate, report, stop=stop):
            raise stop()

        fetcher2._take = interrupt
        with pytest.raises(stop):
            fetcher2.run()
        assert [r["id"] for r in read_manifest(tmp_path / "manifest.jsonl")] == ["src-ok"]


# --- corpus families (L5 fix 1 rulings) ------------------------------------------------------


def test_fed_series_map_to_research_or_government_families():
    from benchmarks.training_corpus.sources import fed

    expected = {
        "FEDS": ("research.research_paper", "research_paper"),
        "feds": ("research.research_paper", "research_paper"),
        "IFDP": ("research.research_paper", "research_paper"),
        "FOMC minutes": ("government.government_report", "generic"),
        "Monetary Policy Report": ("government.government_report", "generic"),
        "Financial Stability Report": ("government.government_report", "generic"),
        "testimony": ("government.government_report", "generic"),
    }
    for series, (doc_type, family) in expected.items():
        assert fed.doc_type_for_series(series) == doc_type, series
        assert family_for_doc_type(doc_type) == family, series

    page = (
        '<div id="2025110"><h5><a href="/econres/feds/x.htm">A</a></h5></div>'
        '<div id="20251429"><h5><a href="/econres/ifdp/y.htm">B</a></h5></div>'
    )
    for series in ("feds", "ifdp"):
        (paper,) = fed.parse_series(page, series)
        assert paper.doc_type_hint == "research.research_paper"
    (minutes,) = fed.parse_minutes('<a href="/monetarypolicy/files/fomcminutes20250129.pdf">')
    assert minutes.doc_type_hint == "government.government_report"


def _fund(key: str, doc_type: str, kind: str) -> Candidate:
    return Candidate(
        id=f"fund_docs-{key}",
        url=f"https://fund_docs.example.org/{key}.pdf",
        source="fund_docs",
        licence="test licence",
        doc_type_hint=doc_type,
        title=f"Fund {key}",
        publisher="Test",
        extra={"kind": kind},
    )


def test_fund_shareholder_reports_are_excluded(tmp_path):
    from benchmarks.training_corpus.sources import fund_docs

    assert fund_docs.excluded({}) is None  # not a fund_docs record
    keep = {"source": "fund_docs", "doc_type_hint": "financial.prospectus", "kind": "prospectus"}
    assert fund_docs.excluded(keep) is None
    assert fund_docs.excluded({**keep, "doc_type_hint": "financial.fact_sheet"}) is None
    assert fund_docs.excluded({**keep, "doc_type_hint": "financial.annual_report"})
    assert fund_docs.excluded({**keep, "kind": "semi-annual report"})
    assert (
        fund_docs.excluded({**keep, "source": "ir_annual_reports", "kind": "annual report"}) is None
    )

    # A shareholder report fetched before the ruling (here: by a source that still lists it) is
    # dropped when the manifest is written, recorded once in drops.jsonl, and not counted.
    candidates = [
        _fund("fs", "financial.fact_sheet", "fact sheet"),
        _fund("sp", "financial.prospectus", "summary prospectus"),
        _fund("ar", "financial.annual_report", "annual report"),
        _fund("sar", "financial.annual_report", "semi-annual report"),
    ]
    net, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    source = fake_source("fund_docs", candidates, target=4, cap=4)
    report = make_fetcher(tmp_path, [source], 4, net).run()
    ids = [r["id"] for r in read_manifest(tmp_path / "manifest.jsonl")]
    assert ids == ["fund_docs-fs", "fund_docs-sp"]
    assert report.drops["excluded"] == 2
    drops_path = tmp_path / "cache" / "drops.jsonl"
    drops = [json.loads(line) for line in drops_path.read_text().splitlines()]
    assert sorted(d["id"] for d in drops) == ["fund_docs-ar", "fund_docs-sar"]
    assert all(d["permanent"] and "shareholder report" in d["reason"] for d in drops)

    net2, transport2 = make_net(tmp_path, {})
    report2 = make_fetcher(tmp_path, [source], 4, net2).run()
    assert report2.sources["fund_docs"].have == 2  # the reports no longer count
    assert [r["id"] for r in read_manifest(tmp_path / "manifest.jsonl")] == ids
    assert len(drops_path.read_text().splitlines()) == 2  # recorded once
    assert transport2.urls() == ["https://fund_docs.example.org/robots.txt"]  # none refetched


def test_family_hint_is_derived_from_doc_type(tmp_path):
    candidates = [
        Candidate(
            id="fed-fomcminutes-20250129",
            url="https://fed.example.org/minutes.pdf",
            source="fed",
            licence="test licence",
            doc_type_hint="government.government_report",
            extra={"series": "FOMC minutes"},
        ),
        Candidate(
            id="fed-feds-2025110",
            url="https://fed.example.org/feds.pdf",
            source="fed",
            licence="test licence",
            doc_type_hint="research.research_paper",
            extra={"series": "FEDS"},
        ),
    ]
    net, _ = make_net(tmp_path, {c.url: _pdf(c.id) for c in candidates})
    source = fake_source("fed", candidates, target=2, cap=2)
    make_fetcher(tmp_path, [source], 2, net).run()
    families = {r["id"]: r["family_hint"] for r in read_manifest(tmp_path / "manifest.jsonl")}
    assert families == {"fed-fomcminutes-20250129": "generic", "fed-feds-2025110": "research_paper"}

    # A checkpoint written by older code with a stale family is corrected at manifest-write time.
    journal = tmp_path / "cache" / "fetched.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    journal.write_text(
        "".join(json.dumps({**r, "family_hint": "research_paper"}) + "\n" for r in rows)
    )
    net2, _ = make_net(tmp_path, {})
    make_fetcher(tmp_path, [source], 2, net2).run()
    again = {r["id"]: r["family_hint"] for r in read_manifest(tmp_path / "manifest.jsonl")}
    assert again == families
