"""Training-corpus fetcher (v0.5 L4): exclusion, rate limits, licence and document filters,
splits and resume. Offline: every response comes from a fake transport; no PDF is real except
the one test that runs the isolated page counter on a PDF it makes itself."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import urllib.error
from collections import Counter
from email.message import Message
from pathlib import Path

import pytest

from benchmarks.training_corpus import _net, fetch
from benchmarks.training_corpus.sources import ALL_SOURCES, MAKEUP_ORDER, arxiv, edgar_ars
from benchmarks.training_corpus.sources.base import Candidate, Source

UA = "docrooter-research/0.5 (research; contact via repository)"


def _pdf(tag: str) -> bytes:
    return b"%PDF-1.7\n% " + tag.encode() + b"\n" + b"0" * 64 + b"\n%%EOF\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- fakes ---------------------------------------------------------------------------------


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
        if status >= 400:
            raise urllib.error.HTTPError(url, status, "error", Message(), None)
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
        log=lambda m: None,
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
    fetcher = make_fetcher(tmp_path, [fake_source("src", candidates)], 6, net, exclusions)
    report = fetcher.run()

    rows = read_manifest(tmp_path / "manifest.jsonl")
    assert [r["id"] for r in rows] == ["src-clean"]
    assert report.drops == Counter(
        {"dev:url": 1, "dev:title": 1, "heldout:sha256": 1, "heldout:url": 1, "dev:sha256": 1}
    )
    # URL and title matches never reach the network.
    downloaded = transport.urls()
    assert "http://example.com/reports/acme-2023.pdf/#page=2" not in downloaded
    assert "https://src.example.org/dev_title.pdf" not in downloaded
    assert not any("heldout.example.net" in u for u in downloaded)
    # Re-hosted bytes are deleted, and held-out drops do not record the URL or title.
    pdfs = sorted(p.name for p in (tmp_path / "cache" / "pdfs").iterdir())
    assert pdfs == ["src-clean.pdf"]
    drops = [
        json.loads(line) for line in (tmp_path / "cache" / "drops.jsonl").read_text().splitlines()
    ]
    for drop in drops:
        assert drop["permanent"] is True
        if drop["reason"].startswith("heldout"):
            assert "url" not in drop and "title" not in drop
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
        ("arxiv-2501.00001", "CC-BY-4.0"),
        ("arxiv-2501.00005", "CC0-1.0"),
        ("arxiv-2501.00007", "CC-BY-4.0"),
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
    records = [{"id": f"a-{i:03d}", "source": "a"} for i in range(100)]
    records += [{"id": f"b-{i:03d}", "source": "b"} for i in range(40)]
    records += [{"id": "c-only", "source": "c"}]
    splits = fetch.assign_splits(records)
    again = fetch.assign_splits(list(reversed(records)))
    assert splits == again
    by_source = {
        s: Counter(v for k, v in splits.items() if k.startswith(f"{s}-")) for s in ("a", "b", "c")
    }
    assert by_source["a"] == Counter({"train": 85, "gold": 10, "calib": 5})
    assert by_source["b"] == Counter({"train": 34, "gold": 4, "calib": 2})
    assert by_source["c"] == Counter({"train": 1})
    assert fetch.assign_splits(records, seed=1) != splits  # the seed matters
    # Adding a document to one source leaves the other sources' splits alone.
    more = fetch.assign_splits(records + [{"id": "b-new", "source": "b"}])
    assert {k: v for k, v in more.items() if k.startswith("a-")} == {
        k: v for k, v in splits.items() if k.startswith("a-")
    }


def test_quotas_scale_to_target_and_sources_match_the_spec():
    assert {s.name: s.target for s in ALL_SOURCES} == {
        "edgar_ars": 300,
        "fund_docs": 150,
        "govinfo_gao": 150,
        "crs": 100,
        "fed": 60,
        "arxiv": 150,
        "uscourts": 100,
        "rfc": 40,
        "edgar_ex10": 50,
    }
    assert sorted(MAKEUP_ORDER) == sorted(s.name for s in ALL_SOURCES)
    for target in (10, 30, 1100, 1500):
        quotas = fetch.scaled_quotas(ALL_SOURCES, target)
        assert sum(quotas.values()) == target
    assert fetch.scaled_quotas(ALL_SOURCES, 1100)["edgar_ars"] == 300
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
    real_run = fetch.subprocess.run

    def spy_run(cmd, **kwargs):
        calls.append(cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(fetch.subprocess, "run", spy_run)
    info = fetch.inspect_pdf(path)
    assert info.pages == 3 and info.text_chars > fetch.MIN_TEXT_CHARS and not info.encrypted
    assert calls[0][1] == "-I"
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
            "gao-GAO-08-705",
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
        "uscourts-caDC-1-0",
    ]
    assert opinions[0].url.endswith("/USCOURTS-ca5-1/pdf/USCOURTS-ca5-1-0.pdf")

    funds = list(fund_docs.candidates(None))  # an explicit list: no network
    assert len(funds) == fund_docs.SOURCE.cap >= 150
    assert len({c.id for c in funds}) == len(funds)
    assert {c.url.split("/")[2] for c in funds} == {"www.ssga.com", "www.ishares.com"}
    assert [c.extra["sponsor"] for c in funds[:3]] == ["SSGA", "iShares", "iShares"]
    assert all(c.family_hint == "fund_document" for c in funds)


def test_default_transport_passes_the_timeout_by_keyword(monkeypatch):
    seen = {}

    def fake_urlopen(request, data=None, timeout=None):
        seen.update(data=data, timeout=timeout)
        return FakeResponse(b"ok")

    monkeypatch.setattr(_net.urllib.request, "urlopen", fake_urlopen)
    net = _net.Net(None, log=lambda m: None)
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
