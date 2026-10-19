"""The playground server: SSE event order, result schema, upload limits and checks, temp-file
cleanup, history, backend switching and the static page. Torch-free: keyword backend and
generated PDFs."""

import asyncio
import importlib
import json
import logging
import os
import socket
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pytest.importorskip("httpx")  # Starlette's TestClient needs it; it is not in the `ui` extra
pymupdf = pytest.importorskip("pymupdf")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from docrooter import cli  # noqa: E402
from docrooter.backends import KeywordBackend, registry  # noqa: E402
from docrooter.layout import registry as layout_registry  # noqa: E402
from docrooter.ocr import registry as ocr_registry  # noqa: E402
from docrooter.rooter import PROFILES, RouterConfig  # noqa: E402
from docrooter.ui import server  # noqa: E402

rooter_process = importlib.import_module("docrooter.rooter.process")  # not the function

BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."
MARKDOWN = (
    b"# Abstract\n\nWe study things.\n\n# Methods\n\nWe did experiments.\n\n"
    b"# Results\n\nIt works.\n"
)


def pdf_bytes(pages: int = 3) -> bytes:
    doc = pymupdf.open()
    for n in range(pages):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 90), f"Chapter {n + 1}", fontsize=22, fontname="hebo")
        y = 130
        for k in range(4):
            page.insert_text((72, y), f"{BODY} Line {k}.", fontsize=11, fontname="helv")
            y += 15
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    """``ocr="auto"`` must never pick an installed engine here (ocrmac is forbidden in unit
    tests); tests that want one patch this again."""
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


def make_client(backend="keyword", host="127.0.0.1", port=8765):
    """A TestClient that addresses the app the way a browser would (Host 127.0.0.1:8765)."""
    app = server.create_app(backend, host=host, port=port)
    return TestClient(app, base_url=f"http://{host}:{port}")


@pytest.fixture
def client():
    with make_client() as test_client:
        yield test_client


@pytest.fixture
def temp_paths(monkeypatch):
    """Every temp file the server creates."""
    made: list[str] = []
    real = tempfile.mkstemp

    def spy(*args, **kwargs):
        fd, path = real(*args, **kwargs)
        made.append(path)
        return fd, path

    monkeypatch.setattr(server.tempfile, "mkstemp", spy)
    return made


def post(client, name="doc.pdf", data=None, **options):
    if data is None:
        data = pdf_bytes()
    fields = {key: str(value) for key, value in options.items()}
    return client.post("/api/runs", files={"file": (name, data)}, data=fields)


def stream(client, run_id, headers=None):
    """The run's SSE events, read until the server closes the stream."""
    out = []
    with client.stream("GET", f"/api/runs/{run_id}/events", headers=headers or {}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        ids = []
        for line in response.iter_lines():
            if line.startswith("id: "):
                ids.append(int(line[4:]))
            elif line.startswith("data: "):
                out.append(json.loads(line[6:]))
    assert len(ids) == len(out)
    return out, ids


def run(client, name="doc.pdf", data=None, **options):
    response = post(client, name, data, **options)
    assert response.status_code == 202, response.text
    run_id = response.json()["run_id"]
    events, _ = stream(client, run_id)
    return run_id, events


# --- events -------------------------------------------------------------------------------


def test_events_stream_in_order_and_end_with_the_job_sentinel(client):
    _, events = run(client, normalize="true", mode="view", profile="fast")
    starts = [e["stage"] for e in events if e["status"] == "start"]
    assert starts == [
        "upload",
        "extract",
        "signals",
        "headings",
        "assemble",
        "view",
        "classify",
        "normalize",
    ]
    for stage in starts:  # every stage that started also ended, after its start
        statuses = [e["status"] for e in events if e["stage"] == stage]
        assert statuses[0] == "start" and statuses[-1] == "end"
    assert events[-2]["stage"] == "done"
    assert events[-1] == {
        "stage": "job",
        "status": "end",
        "t": events[-1]["t"],
        "data": {"status": "done", "seconds": events[-1]["t"]},
    }
    times = [e["t"] for e in events[:-1]]
    assert times == sorted(times)
    upload_end = next(e for e in events if e["stage"] == "upload" and e["status"] == "end")
    assert upload_end["data"]["bytes"] > 0 and upload_end["data"]["kind"] == "pdf"
    extract_end = next(e for e in events if e["stage"] == "extract" and e["status"] == "end")
    assert extract_end["data"]["pages"] == 3


def test_stream_replays_finished_runs_and_resumes_after_last_event_id(client):
    run_id, events = run(client)
    again, ids = stream(client, run_id)
    assert again == events and ids == list(range(len(events)))
    rest, rest_ids = stream(client, run_id, headers={"Last-Event-ID": "2"})
    assert rest == events[3:] and rest_ids[0] == 3


def test_failed_run_reports_error_then_job_end(client, temp_paths):
    run_id, events = run(client, data=b"%PDF-1.7 this is not really a PDF")
    assert [e["stage"] for e in events[-2:]] == ["error", "job"]
    assert events[-1]["data"]["status"] == "error"
    assert all(temp not in events[-1]["data"]["error"] for temp in temp_paths)
    # the pipeline's own error event names the upload, never the server's temp file
    stored = client.get(f"/api/runs/{run_id}").json()["events"][-2]
    for message in (events[-2]["data"]["message"], stored["data"]["message"]):
        assert "doc.pdf" in message
        for temp in temp_paths:
            assert temp not in message and os.path.realpath(temp) not in message
            assert os.path.basename(temp) not in message
    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)


def test_markdown_upload_runs_without_pdf_stages(client):
    _, events = run(client, "paper.md", MARKDOWN, profile="fast")
    starts = [e["stage"] for e in events if e["status"] == "start"]
    assert starts == ["upload", "assemble", "view", "classify"]
    run_id, events = run(client, "paper.md", MARKDOWN)  # balanced: summaries, still no PDF stages
    starts = [e["stage"] for e in events if e["status"] == "start"]
    assert starts == ["upload", "assemble", "summarize", "view", "classify"]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert [e["title"] for e in detail["toc"]] == ["Abstract", "Methods", "Results"]
    assert detail["sections"] is None


# --- results ------------------------------------------------------------------------------


def test_result_schema(client):
    run_id, _ = run(client, normalize="true", engine="pymupdf", mode="hybrid")
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "done" and detail["error"] is None
    assert detail["filename"] == "doc.pdf" and detail["kind"] == "pdf"
    assert detail["options"] == {
        "backend": "keyword",
        "engine": "pymupdf",
        "mode": "hybrid",
        "normalize": True,
        "profile": "balanced",
    }
    assert detail["document"]["source"] == "doc.pdf"  # never the temp path
    assert detail["meta"]["n_pages"] == 3 and detail["meta"]["engine"] == "pymupdf"

    toc = detail["toc"]
    assert [e["title"] for e in toc] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert set(toc[0]) == {"id", "title", "level", "depth", "page", "heading_score", "flags"}
    assert [e["page"] for e in toc] == [1, 2, 3]

    doc_type = detail["doc_type"]
    assert set(doc_type) == {
        "id",
        "label",
        "path",
        "node_ids",
        "confidence",
        "stopped_reason",
        "mode",
        "leaf_probs",
    }
    assert doc_type["mode"] == "hybrid"
    assert 1 <= len(doc_type["leaf_probs"]) <= 5
    assert set(doc_type["leaf_probs"][0]) == {"id", "label", "prob"}
    probs = [p["prob"] for p in doc_type["leaf_probs"]]
    assert probs == sorted(probs, reverse=True)

    sections = detail["sections"]
    assert sections["taxonomy"].startswith("section_types/")
    assert [e["title"] for e in sections["entries"]] == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert set(sections["entries"][0]) == {
        "id",
        "title",
        "level",
        "depth",
        "canonical",
        "label",
        "score",
    }

    timings = detail["timings"]
    for stage in ("upload", "extract", "signals", "headings", "view", "classify", "normalize"):
        assert timings[stage] >= 0
    assert detail["seconds"] >= detail["events"][-2]["t"]
    assert detail["events"][-1]["stage"] == "job"

    (summary,) = client.get("/api/runs").json()
    assert summary["id"] == run_id and summary["status"] == "done"
    assert summary["doc_type"] == doc_type["label"]


def test_unknown_run_is_404(client):
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/events").status_code == 404


def test_history_keeps_the_last_20_runs(client):
    ids = [run(client, f"doc{i}.md", MARKDOWN)[0] for i in range(server.HISTORY + 2)]
    listed = [r["id"] for r in client.get("/api/runs").json()]
    assert listed == list(reversed(ids[2:]))  # newest first
    assert client.get(f"/api/runs/{ids[0]}").status_code == 404
    assert client.get(f"/api/runs/{ids[2]}").status_code == 200


# --- uploads ------------------------------------------------------------------------------


def test_oversize_upload_is_413(client, monkeypatch, temp_paths):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    response = post(client, data=pdf_bytes())  # a few KB: caught while copying
    assert response.status_code == 413
    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)
    big = b"%PDF" + b"0" * 200_000  # Content-Length alone gives it away
    assert post(client, data=big).status_code == 413
    assert client.get("/api/runs").json() == []


def test_non_pdf_bytes_named_pdf_are_415(client, temp_paths):
    response = post(client, "report.pdf", b"PK\x03\x04 a zip, not a PDF")
    assert response.status_code == 415
    assert "PDF" in response.json()["detail"]
    assert not any(os.path.exists(p) for p in temp_paths)


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("report.docx", b"PK\x03\x04", 415),
        ("notes.txt", b"\xff\xfe\x00bad", 415),
        ("empty.md", b"", 400),
    ],
)
def test_other_rejected_uploads(client, name, data, status):
    assert post(client, name, data).status_code == status


@pytest.mark.parametrize(
    "options", [{"backend": "trained:/tmp/x"}, {"engine": "ocr"}, {"mode": "vibes"}]
)
def test_invalid_options_are_400(client, options):
    response = post(client, **options)
    assert response.status_code == 400
    assert "Unknown" in response.json()["detail"]


def test_cross_site_upload_is_refused(client):
    response = client.post(
        "/api/runs",
        files={"file": ("doc.md", MARKDOWN)},
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_temp_file_is_deleted_after_the_run(client, temp_paths):
    run(client)
    assert len(temp_paths) == 1
    assert temp_paths[0].endswith(".pdf")
    assert not os.path.exists(temp_paths[0])


@pytest.mark.parametrize("profile", ["fast", "balanced"])
def test_pdfs_are_read_with_a_page_cap(client, monkeypatch, profile):
    seen = {}
    real = server.load

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real(path, **kwargs)

    monkeypatch.setattr(server, "load", spy)
    monkeypatch.setattr(rooter_process, "load", spy)  # the other profiles load through the rooter
    run(client, engine="pdfplumber", profile=profile)
    assert seen["max_pages"] == server.MAX_PAGES == 2000
    assert seen["engine"] == "pdfplumber"


# --- backends -----------------------------------------------------------------------------


class ClosingBackend(KeywordBackend):
    made = 0
    closed = 0

    def __init__(self, *args, **kwargs):
        super().__init__()
        type(self).made += 1

    def close(self):
        type(self).closed += 1


def test_backend_is_reused_and_switching_closes_the_previous_classifier(monkeypatch):
    monkeypatch.setitem(registry._KINDS, "closing", ClosingBackend)
    monkeypatch.setattr(ClosingBackend, "made", 0)
    monkeypatch.setattr(ClosingBackend, "closed", 0)
    with make_client("closing") as client:
        playground = client.app.state.playground
        for _ in range(2):
            _, events = run(client, "a.md", MARKDOWN)
            assert events[-1]["data"]["status"] == "done"
        assert (ClosingBackend.made, ClosingBackend.closed) == (1, 0)
        assert playground.backends.spec == "closing"
        assert client.get("/api/backends").json()["loaded"] == "closing"

        _, events = run(client, "a.md", MARKDOWN, backend="keyword")
        assert events[-1]["data"]["status"] == "done"
        assert ClosingBackend.closed == 1
        assert playground.backends.spec == "keyword"
    assert ClosingBackend.closed == 1  # nothing else of ours to close on shutdown


def test_shutdown_closes_the_loaded_backend(monkeypatch):
    monkeypatch.setitem(registry._KINDS, "closing", ClosingBackend)
    monkeypatch.setattr(ClosingBackend, "closed", 0)
    with make_client("closing") as client:
        run(client, "a.md", MARKDOWN)
    assert ClosingBackend.closed == 1


def test_backends_listing(client, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    info = client.get("/api/backends").json()
    specs = [b["spec"] for b in info["backends"]]
    assert specs == [
        "keyword",
        "embedding:dmis-lab/UltraBERT-Embed-150M",
        "embedding:BAAI/bge-small-en-v1.5",
        "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    ]
    assert info["default"] == "keyword" and info["loaded"] is None
    assert info["engines"][0] == "auto" and info["modes"] == ["view", "sections", "hybrid"]
    for entry in info["backends"]:
        assert entry["description"] and isinstance(entry["installed"], bool)
    assert info["backends"][0]["installed"] is True

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    specs = [b["spec"] for b in client.get("/api/backends").json()["backends"]]
    assert specs[-1] == "jev:jev-1.13.0"


def test_blank_jev_key_is_no_key_in_the_playground_and_the_profiles(client, monkeypatch):
    # final review minor: one check (strip, then non-empty) for the server and the profiles
    from docrooter.rooter.profiles import profile_defaults

    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    specs = [b["spec"] for b in client.get("/api/backends").json()["backends"]]
    assert "jev:jev-1.13.0" not in specs
    assert profile_defaults("accurate")["escalate"] == "none"
    assert post(client, backend="jev:jev-1.13.0").status_code == 400


def test_summary_keyphrases_are_cut():
    from docrooter.summarize import Summary
    from docrooter.ui.server import MAX_KEYPHRASE_CHARS, MAX_KEYPHRASES, _summary_out

    summary = Summary(key_sentences=["s"], keyphrases=["x" * 5000] * 50, method="lead")
    out = _summary_out(summary)
    assert len(out["keyphrases"]) == MAX_KEYPHRASES
    assert all(len(p) == MAX_KEYPHRASE_CHARS and p.endswith("…") for p in out["keyphrases"])
    assert _summary_out(Summary(key_sentences=[], keyphrases=["revenue growth"], method="lead"))[
        "keyphrases"
    ] == ["revenue growth"]


def test_jev_needs_the_key_to_be_selectable(client, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert post(client, backend="jev:jev-1.13.0").status_code == 400


def test_listing_backends_imports_no_model_library(client):
    import sys

    client.get("/api/backends")
    assert "sentence_transformers" not in sys.modules and "transformers" not in sys.modules


# --- static page --------------------------------------------------------------------------


def test_static_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "default-src 'self'" in response.headers["content-security-policy"]
    html = response.text
    assert '<script type="module" src="/static/app.js">' in html
    assert 'aria-live="polite"' in html
    for name, kind in (("app.js", "text/javascript"), ("style.css", "text/css")):
        asset = client.get(f"/static/{name}")
        assert asset.status_code == 200 and asset.headers["content-type"].startswith(kind)
    icon = client.get("/static/favicon.svg")
    assert icon.status_code == 200 and icon.headers["content-type"].startswith("image/svg+xml")
    assert 'rel="icon" type="image/svg+xml" href="/static/favicon.svg"' in html
    assert "default-src 'self'" in icon.headers["content-security-policy"]
    assert client.get("/static/server.py").status_code == 404
    assert client.get("/static/..%2Fserver.py").status_code == 404


def test_page_uses_no_external_assets(client):
    for path in ("/", "/static/app.js", "/static/style.css", "/static/favicon.svg"):
        text = client.get(path).text
        assert "http://" not in text.replace("http://www.w3.org", "")
        assert "https://" not in text


# --- env file and CLI ---------------------------------------------------------------------


def test_load_env_file_sets_only_missing_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nexport A=\"1\"\nB=2\nnot a pair\nC='3'\n", encoding="utf-8")
    environ = {"B": "kept"}
    server.load_env_file(env, environ)
    assert environ == {"A": "1", "B": "kept", "C": "3"}


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("192.168.1.4", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host, loopback):
    assert cli._is_loopback(host) is loopback


def test_browser_url():
    assert cli._browser_url("127.0.0.1", 8765) == "http://127.0.0.1:8765/"
    assert cli._browser_url("0.0.0.0", 9000) == "http://127.0.0.1:9000/"
    assert cli._browser_url("::1", 8765) == "http://[::1]:8765/"


def test_serve_warns_on_a_non_local_host(monkeypatch):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: started.append(self.config.host))
    result = CliRunner().invoke(cli.app, ["serve", "--host", "192.168.1.4", "--port", "8799"])
    assert result.exit_code == 0, result.output
    assert started == ["192.168.1.4"]
    assert "warning: binding to 192.168.1.4" in result.output

    result = CliRunner().invoke(cli.app, ["serve"])
    assert result.exit_code == 0 and "warning" not in result.output
    assert "http://127.0.0.1:8765/" in result.output


def test_open_waits_for_the_server_then_opens_the_browser(monkeypatch):
    import webbrowser

    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)

    class FakeServer:
        started = False
        should_exit = False

    fake = FakeServer()
    thread = threading.Thread(target=cli._open_when_started, args=(fake, "http://x/"))
    thread.start()
    assert opened == []
    fake.started = True
    thread.join(5)
    assert opened == ["http://x/"]


@pytest.mark.parametrize(
    ("spec", "message"),
    [("nope:x", "unknown backend kind 'nope'"), ("openai", "openai needs a model")],
)
def test_serve_refuses_an_unknown_backend_at_startup(monkeypatch, spec, message):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: started.append(True))
    result = CliRunner().invoke(cli.app, ["serve", "--backend", spec])
    assert result.exit_code == 1
    assert message in result.output and started == []


# --- review round 1: host guard, body limits, queue bound, cleanup ------------------------


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        "evil.example:8765",
        "127.0.0.1:9999",
        "127.0.0.1",  # port 80
        "127.0.0.1.evil.example:8765",
        "user@127.0.0.1:8765",
    ],
)
def test_foreign_host_headers_are_refused_on_every_route(client, host):
    for path in ("/", "/static/app.js", "/api/runs", "/api/backends", "/api/runs/x/events"):
        response = client.get(path, headers={"Host": host})
        assert response.status_code == 400, path
    response = client.post(
        "/api/runs", files={"file": ("doc.md", MARKDOWN)}, headers={"Host": host}
    )
    assert response.status_code == 400
    assert client.get("/api/runs").json() == []


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8765", "localhost:8765", "LOCALHOST:8765", "[::1]:8765"]
)
def test_loopback_host_headers_are_accepted(client, host):
    assert client.get("/api/backends", headers={"Host": host}).status_code == 200


def test_a_non_loopback_bind_accepts_its_own_host_only():
    with make_client(host="192.168.1.4", port=9000) as lan:
        assert lan.get("/api/backends").status_code == 200
        assert lan.get("/", headers={"Host": "evil.example:9000"}).status_code == 400
        assert lan.get("/", headers={"Host": "192.168.1.4:8765"}).status_code == 400


def test_any_address_bind_accepts_this_machine(monkeypatch):
    monkeypatch.setattr(server.socket, "gethostname", lambda: "Bench")
    monkeypatch.setattr(
        server.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.7", 0))]
    )
    names = server.allowed_host_names("0.0.0.0")
    assert {"127.0.0.1", "localhost", "::1", "bench", "bench.local", "10.0.0.7"} <= names
    assert "evil.example" not in names
    assert server.allowed_host_names("127.0.0.1") == {"127.0.0.1", "localhost", "::1"}


def test_null_origin_is_refused(client):
    response = client.post(
        "/api/runs", files={"file": ("doc.md", MARKDOWN)}, headers={"Origin": "null"}
    )
    assert response.status_code == 403


def test_same_origin_upload_is_accepted(client):
    response = post(client, "doc.md", MARKDOWN)
    assert response.status_code == 202
    response = client.post(
        "/api/runs",
        files={"file": ("doc.md", MARKDOWN)},
        headers={"Origin": "http://127.0.0.1:8765"},
    )
    assert response.status_code == 202


def multipart(name: str, data: bytes, boundary: str = "x0x0x0") -> tuple[bytes, bytes]:
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}".encode()


def test_chunked_upload_without_content_length_is_411(client, temp_paths):
    body, content_type = multipart("doc.md", MARKDOWN)

    def chunks():
        yield body[:50]
        yield body[50:]

    response = client.post(
        "/api/runs", content=chunks(), headers={"Content-Type": content_type.decode()}
    )
    assert response.status_code == 411
    assert temp_paths == [] and client.get("/api/runs").json() == []


@pytest.mark.parametrize(
    "content_length",
    # "\xb2" is latin-1 for "2" superscript: isdigit(). Past 18 digits it is no byte count
    # (and past 4,300 digits int() itself refuses it).
    [b"12abc", b"-1", b"\xb2", b"1\xb9", b"9" * 19, b"1" * 4301],
    ids=["letters", "negative", "superscript", "mixed", "19-digits", "4301-digits"],
)
def test_malformed_content_length_is_400(content_length):
    status = asyncio.run(drive_post([b"x"], content_length=content_length))
    assert status == 400


def test_an_18_digit_content_length_is_too_large_not_malformed():
    assert asyncio.run(drive_post([b"x"], content_length=b"9" * 18)) == 413


@pytest.mark.parametrize(
    "last", [b"\xb2", b"1" * 19, b"1" * 4301], ids=["superscript", "19-digits", "4301-digits"]
)
def test_non_ascii_or_overlong_last_event_id_is_ignored(client, last):
    run_id, events = run(client)
    again, ids = stream(client, run_id, headers={"Last-Event-ID": last})
    assert again == events and ids == list(range(len(events)))


async def drive_post(chunks: list[bytes], content_length: bytes, content_type: bytes = b"") -> int:
    """POST ``chunks`` to the app over raw ASGI with any declared Content-Length (TestClient
    always sends the true one); returns the response status."""
    app = server.create_app("keyword")
    messages = [
        {"type": "http.request", "body": chunk, "more_body": i < len(chunks) - 1}
        for i, chunk in enumerate(chunks)
    ]
    sent = []

    async def receive():
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"127.0.0.1:8765"), (b"content-length", content_length)]
    if content_type:
        headers.append((b"content-type", content_type))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/runs",
        "raw_path": b"/api/runs",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }
    await app(scope, receive, send)
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_body_larger_than_its_declared_content_length_is_413(monkeypatch, temp_paths):
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 1000)
    body, content_type = multipart("doc.pdf", b"%PDF" + b"0" * 300_000)
    chunks = [body[i : i + 16_384] for i in range(0, len(body), 16_384)]
    status = asyncio.run(drive_post(chunks, content_length=b"100", content_type=content_type))
    assert status == 413
    assert temp_paths == []


def test_text_is_decoded_incrementally(client, monkeypatch):
    monkeypatch.setattr(server, "_CHUNK", 3)  # split multi-byte characters across reads
    text = "# Résumé\n\nCafé, naïve, ✓ and 日本語.\n".encode()
    assert post(client, "notes.md", text).status_code == 202
    assert post(client, "bom.txt", b"\xef\xbb\xbf" + text).status_code == 202
    assert post(client, "bad.md", text + b"\xff more").status_code == 415
    assert post(client, "cut.txt", text + "é".encode()[:1]).status_code == 415


def blocking_load(monkeypatch):
    """Make the worker's ``load`` wait for ``release``; ``started`` is set once it is inside."""
    started, release = threading.Event(), threading.Event()
    real = server.load

    def slow(path, **kwargs):
        started.set()
        release.wait(10)
        return real(path, **kwargs)

    monkeypatch.setattr(server, "load", slow)
    monkeypatch.setattr(rooter_process, "load", slow)
    return started, release


def test_queue_is_bounded(monkeypatch, temp_paths):
    started, release = blocking_load(monkeypatch)
    with make_client() as client:
        try:
            first = post(client, "a.md", MARKDOWN)
            assert first.status_code == 202
            assert started.wait(5)  # the worker took it: the queue is empty again
            waiting = [post(client, f"w{i}.md", MARKDOWN) for i in range(server.MAX_PENDING)]
            assert [r.status_code for r in waiting] == [202] * server.MAX_PENDING
            refused = post(client, "late.md", MARKDOWN)
            assert refused.status_code == 503
            assert len(temp_paths) == server.MAX_PENDING + 1  # none made for the refused one
        finally:
            release.set()
        for response in [first, *waiting]:
            events, _ = stream(client, response.json()["run_id"])
            assert events[-1]["data"]["status"] == "done"
    assert not any(os.path.exists(p) for p in temp_paths)


def test_stop_deletes_the_running_jobs_file_when_the_join_times_out(
    monkeypatch, temp_paths, caplog
):
    started, release = blocking_load(monkeypatch)
    monkeypatch.setattr(server, "STOP_TIMEOUT_SECONDS", 0.2)
    with make_client() as client:
        playground = client.app.state.playground
        assert post(client, "a.md", MARKDOWN).status_code == 202
        assert started.wait(5)
        assert os.path.exists(temp_paths[0])
    assert not os.path.exists(temp_paths[0])  # deleted by stop(), the job still running
    with caplog.at_level(logging.CRITICAL):  # the late job fails quietly on its own
        release.set()
        playground._worker.join(5)
    assert not playground._worker.is_alive()


def test_startup_sweeps_stale_temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(server.tempfile, "gettempdir", lambda: str(tmp_path))
    hour_ago = time.time() - 3700
    stale = tmp_path / "docrooter-stale.pdf"
    fresh = tmp_path / "docrooter-fresh.pdf"
    other = tmp_path / "someone-else.pdf"
    for path in (stale, fresh, other):
        path.write_bytes(b"%PDF")
    for path in (stale, other):
        os.utime(path, (hour_ago, hour_ago))
    with make_client():
        pass
    assert not stale.exists() and fresh.exists() and other.exists()


def test_load_env_file_strips_inline_comments_and_is_silent(tmp_path, capsys, caplog):
    env = tmp_path / ".env"
    env.write_text(
        "KEY=value # the key\n"
        "URL=http://host/#fragment\n"
        'QUOTED="a # b" # trailing\n'
        "SINGLE='x' \n"
        "TABBED=v\t# tab comment\n"
        "EMPTY= # only a comment\n"
        "HASHED=#x1\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}
    with caplog.at_level(logging.DEBUG):
        server.load_env_file(env, environ)
    assert environ == {
        "KEY": "value",
        "URL": "http://host/#fragment",
        "QUOTED": "a # b",
        "SINGLE": "x",
        "TABBED": "v",
        "EMPTY": "",
        "HASHED": "#x1",  # no whitespace before "#": part of the value
    }
    assert capsys.readouterr() == ("", "")
    assert caplog.records == []


# --- final review: shutdown, text uploads, TOC cap ----------------------------------------


def free_port() -> int:
    """A free local port in the range kept for test servers."""
    for port in range(18900, 19000):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    pytest.skip("no free port in 18900-18999")


def test_serve_stops_while_a_page_follows_a_run(monkeypatch, temp_paths, capfd, caplog):
    """A page following a live run keeps an SSE response open until the run ends; stopping
    the server must not wait for that (Ctrl+C used to hang until every followed run ended).
    The stream is ended, not cancelled, so uvicorn logs no ASGI exception."""
    import httpx

    started, release = blocking_load(monkeypatch)
    monkeypatch.setattr(server, "STOP_TIMEOUT_SECONDS", 0.2)
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    app = server.create_app("keyword", port=port)
    srv = cli._playground_server(app, "127.0.0.1", port)
    serving = threading.Thread(target=srv.run, daemon=True)
    serving.start()
    try:
        deadline = time.monotonic() + 10
        while not srv.started:
            assert time.monotonic() < deadline and serving.is_alive()
            time.sleep(0.02)
        with httpx.Client(base_url=base, timeout=5) as http:
            run_id = http.post("/api/runs", files={"file": ("a.md", MARKDOWN)}).json()["run_id"]
        assert started.wait(5)
        following = threading.Event()

        def follow():
            try:
                with httpx.stream("GET", f"{base}/api/runs/{run_id}/events", timeout=None) as r:
                    following.set()
                    for _ in r.iter_lines():
                        pass
            except httpx.HTTPError:
                pass  # the server closed the stream

        follower = threading.Thread(target=follow, daemon=True)
        follower.start()
        assert following.wait(5)
        stopping = time.monotonic()
        srv.should_exit = True
        serving.join(5)
        assert not serving.is_alive(), "the server waited for the followed run"
        assert time.monotonic() - stopping < 3
        follower.join(5)
        assert not follower.is_alive()
    finally:
        with caplog.at_level(logging.CRITICAL):  # the abandoned job fails on its own
            release.set()
            serving.join(10)
            app.state.playground._worker.join(5)
    assert not any(os.path.exists(p) for p in temp_paths)
    assert "Exception in ASGI application" not in capfd.readouterr().err


def test_serve_configures_a_graceful_shutdown_timeout(monkeypatch):
    import uvicorn

    configs = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: configs.append(self.config))
    result = CliRunner().invoke(cli.app, ["serve", "--port", "18999"])
    assert result.exit_code == 0, result.output
    (config,) = configs
    assert config.timeout_graceful_shutdown == cli.SHUTDOWN_GRACE_SECONDS > 0


def test_text_uploads_have_a_lower_size_cap(client, monkeypatch, temp_paths):
    assert server.MAX_TEXT_UPLOAD_BYTES == 1 * 1024 * 1024
    big = b"# Big\n\n" + b"x" * server.MAX_TEXT_UPLOAD_BYTES
    response = post(client, "big.md", big)
    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "1 MB" in detail and ".md" in detail and ".txt" in detail
    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)
    assert client.get("/api/runs").json() == []

    monkeypatch.setattr(server, "MAX_TEXT_UPLOAD_BYTES", len(MARKDOWN))
    assert client.get("/api/backends").json()["max_text_upload_bytes"] == len(MARKDOWN)
    assert post(client, "fits.txt", MARKDOWN).status_code == 202
    assert post(client, "over.txt", MARKDOWN + b"!").status_code == 413
    assert len(pdf_bytes()) > len(MARKDOWN)
    assert post(client, "doc.pdf").status_code == 202  # PDFs keep MAX_UPLOAD_BYTES


def test_toc_entries_are_capped_with_a_truncated_flag(client, monkeypatch):
    assert server.MAX_TOC_ENTRIES == 5000
    monkeypatch.setattr(server, "MAX_TOC_ENTRIES", 2)
    run_id, _ = run(client, "paper.md", MARKDOWN, normalize="true")
    detail = client.get(f"/api/runs/{run_id}").json()
    assert [e["title"] for e in detail["toc"]] == ["Abstract", "Methods"]
    assert [e["title"] for e in detail["sections"]["entries"]] == ["Abstract", "Methods"]
    assert detail["truncated"] is True and detail["toc_total"] == 3
    assert detail["document"]["sections"] > 2  # the document itself is whole

    monkeypatch.setattr(server, "MAX_TOC_ENTRIES", 3)
    run_id, _ = run(client, "paper.md", MARKDOWN)
    detail = client.get(f"/api/runs/{run_id}").json()
    assert len(detail["toc"]) == 3 and detail["truncated"] is False
    assert detail["toc_total"] == 3


def test_page_shows_the_truncation_and_the_text_cap(client):
    script = client.get("/static/app.js").text
    assert "run.truncated" in script and "toc_total" in script
    assert "max_text_upload_bytes" in script


# --- profiles, elements, summaries and routes -----------------------------------------------


def test_server_profile_field_and_elements_payload(client):
    run_id, events = run(client, profile="balanced", normalize="true")
    starts = [e["stage"] for e in events if e["status"] == "start"]
    for stage in ("route", "elements", "summarize"):
        assert stage in starts
    assert starts.index("route") < starts.index("summarize") < starts.index("classify")
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["options"]["profile"] == "balanced" and detail["status"] == "done"

    elements = detail["elements"]
    assert detail["elements_total"] == len(elements) > 0
    assert detail["elements_truncated"] is False
    assert set(elements[0]) == {
        "id",
        "kind",
        "order",
        "page",
        "section_id",
        "text",
        "text_chars",
        "source",
        "confidence",
        "flags",
        "level",
        "summary",
    }
    assert [e["order"] for e in elements] == sorted(e["order"] for e in elements)
    assert {e["kind"] for e in elements} >= {"heading", "paragraph"}
    assert {e["page"] for e in elements} == {1, 2, 3}
    assert all(e["source"] == "text" for e in elements)
    assert sum(detail["element_counts"].values()) == detail["elements_total"]
    counts = detail["element_counts"]
    assert counts["heading"] + counts.get("title", 0) == 3  # the first chapter may be the title

    toc_ids = {e["id"] for e in detail["toc"]}
    assert detail["summaries"] and set(detail["summaries"]) <= toc_ids
    summary = next(iter(detail["summaries"].values()))
    assert set(summary) == {"key_sentences", "keyphrases"} and summary["key_sentences"]

    routes = detail["routes"]
    assert routes["pages"] == 3 and routes["tasks"] == {"text": 3} and routes["regions"] == 0
    assert {d["task"] for d in routes["decisions"]} >= {"text", "ocr", "layout"}
    assert all(set(d) == {"task", "engine", "reason"} for d in routes["decisions"])
    assert "routes" not in detail["meta"]  # the per-page log is summarised, not repeated
    route_end = [e for e in events if e["stage"] == "route" and e["status"] == "end"]
    assert [e["data"].get("pages") for e in route_end] == [None, 3]


def test_server_fast_profile_has_no_elements_or_routes(client):
    run_id, events = run(client, profile="fast")
    assert not {"route", "elements", "summarize"} & {e["stage"] for e in events}
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["options"]["profile"] == "fast"
    assert detail["elements"] == [] and detail["elements_total"] == 0
    assert detail["summaries"] == {} and detail["routes"] is None
    assert detail["toc"]  # the headings are those of v0.3


def test_server_markdown_gets_elements_without_routes(client):
    run_id, _ = run(client, "paper.md", MARKDOWN)
    detail = client.get(f"/api/runs/{run_id}").json()
    assert [e["kind"] for e in detail["elements"]][:2] == ["heading", "paragraph"]
    assert {e["page"] for e in detail["elements"]} == {None}
    assert detail["routes"]["pages"] == 0


def test_server_elements_capped(client, monkeypatch):
    assert server.MAX_TOC_ENTRIES == 5000
    run_id, _ = run(client)
    total = client.get(f"/api/runs/{run_id}").json()["elements_total"]
    assert total > 3

    monkeypatch.setattr(server, "MAX_TOC_ENTRIES", 3)
    run_id, _ = run(client)
    detail = client.get(f"/api/runs/{run_id}").json()
    assert len(detail["elements"]) == 3 and detail["elements_total"] == total
    assert detail["elements_truncated"] is True
    assert [e["order"] for e in detail["elements"]] == [0, 1, 2]
    assert detail["element_counts"]  # counts cover the whole document

    monkeypatch.setattr(server, "MAX_TOC_ENTRIES", total)
    run_id, _ = run(client)
    detail = client.get(f"/api/runs/{run_id}").json()
    assert len(detail["elements"]) == total and detail["elements_truncated"] is False


def test_server_element_text_is_cut_to_200_characters(client):
    long_line = "word " * 120
    body = f"# Heading\n\n{long_line.strip()}. Second sentence here.\n"
    run_id, _ = run(client, "long.md", body.encode())
    paragraph = next(
        e for e in client.get(f"/api/runs/{run_id}").json()["elements"] if e["kind"] == "paragraph"
    )
    assert len(paragraph["text"]) <= 200 and paragraph["text_chars"] > 200


def test_server_rejects_unknown_profile_400(client):
    response = post(client, profile="turbo")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unknown profile" in detail and "balanced" in detail
    assert client.get("/api/runs").json() == []


def test_server_engines_without_element_mode_need_the_fast_profile(client):
    for engine in ("both", "pymupdf4llm"):
        response = post(client, engine=engine, profile="balanced")
        assert response.status_code == 400 and "fast" in response.json()["detail"]
    assert post(client, engine="both", profile="fast").status_code == 202
    assert post(client, "paper.md", MARKDOWN, engine="both", profile="balanced").status_code == 202


def test_backends_listing_has_profiles(client):
    info = client.get("/api/backends").json()
    assert info["default_profile"] == "balanced"
    assert [p["name"] for p in info["profiles"]] == list(PROFILES)
    for entry in info["profiles"]:
        assert entry["label"] and entry["description"] and isinstance(entry["installed"], bool)
    installed = {p["name"]: p["installed"] for p in info["profiles"]}
    assert installed["fast"] and installed["balanced"]


def test_accurate_profile_is_unavailable_without_the_layout_extra(client, monkeypatch):
    real = server.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        return None if name == "onnxruntime" else real(name, *args, **kwargs)

    monkeypatch.setattr(server.importlib.util, "find_spec", find_spec)
    profiles = {p["name"]: p for p in client.get("/api/backends").json()["profiles"]}
    assert profiles["accurate"]["installed"] is False
    assert profiles["balanced"]["installed"] is True
    response = post(client, profile="accurate")
    assert response.status_code == 400
    assert "docrooter[layout]" in response.json()["detail"]
    assert client.get("/api/runs").json() == []


class FakeLayout:
    name = "fakelayout"
    made = 0
    closed = 0

    def __init__(self):
        type(self).made += 1

    def detect(self, path, pages, *, dpi=None):
        return []

    def close(self):
        type(self).closed += 1


@pytest.fixture
def fake_layout(monkeypatch):
    monkeypatch.setattr(layout_registry, "_FACTORIES", dict(layout_registry._FACTORIES))
    layout_registry.register_layout_engine("fakelayout", FakeLayout)
    monkeypatch.setattr(FakeLayout, "made", 0)
    monkeypatch.setattr(FakeLayout, "closed", 0)


class FakeOcr:
    name = "fakeocr"
    made = 0
    closed = 0

    def __init__(self, *, lang="en"):
        type(self).made += 1

    def recognize(self, image, *, width_px, height_px, page):
        return []

    def close(self):
        type(self).closed += 1


@pytest.fixture
def fake_ocr(monkeypatch):
    monkeypatch.setattr(ocr_registry, "_FACTORIES", dict(ocr_registry._FACTORIES))
    ocr_registry.register_ocr_engine("fakeocr", FakeOcr)
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: "fakeocr")
    monkeypatch.setattr(FakeOcr, "made", 0)
    monkeypatch.setattr(FakeOcr, "closed", 0)


def test_engine_pool_keeps_one_set_per_profile(fake_layout, fake_ocr):
    cfg = RouterConfig(profile="accurate", layout="fakelayout")
    pool = server.EnginePool()
    first = pool.get("accurate", cfg)
    assert isinstance(first["layout"], FakeLayout) and first["ocr"].name == "fakeocr"
    assert pool.get("accurate", cfg) == first
    assert (FakeLayout.made, FakeLayout.closed) == (1, 0)
    assert FakeOcr.made == 0  # created on the first page that needs it

    page = SimpleNamespace(page=1, x0=0, y0=0, x1=1, y1=1)
    first["ocr"].recognize(b"", width_px=1, height_px=1, page=page)
    assert FakeOcr.made == 1

    other = pool.get("balanced", RouterConfig(profile="balanced"))
    assert (FakeLayout.made, FakeLayout.closed) == (1, 1)  # the profile changed
    assert (FakeOcr.made, FakeOcr.closed) == (1, 1)
    assert "layout" not in other and other["ocr"] is not first["ocr"]

    pool.get("accurate", cfg)
    assert (FakeLayout.made, FakeLayout.closed) == (2, 1)
    pool.close()
    pool.close()  # safe twice
    assert FakeLayout.closed == 2 and pool.key is None


def test_engine_pool_makes_no_ocr_engine_for_none_or_without_one_installed():
    pool = server.EnginePool()
    assert pool.get("fast", RouterConfig(profile="fast")) == {}
    assert pool.get("balanced", RouterConfig(profile="balanced")) == {}  # none installed


@pytest.fixture
def accurate_with_fake_layout(monkeypatch):
    """The ``accurate`` profile with the counting fake standing in for heron (no extra needed)."""
    real = RouterConfig.resolve

    def resolve(profile=None, file=None, **overrides):
        if profile == "accurate":
            overrides = {**overrides, "layout": "fakelayout"}
        return real(profile=profile, file=file, **overrides)

    monkeypatch.setattr(server.RouterConfig, "resolve", staticmethod(resolve))
    others = tuple(i for i in server.PROFILE_INFO if i.name != "accurate")
    monkeypatch.setattr(
        server, "PROFILE_INFO", others + (server.ProfileInfo("accurate", "Accurate", "test", ()),)
    )


def test_playground_engine_lifetimes(fake_layout, fake_ocr, accurate_with_fake_layout, monkeypatch):
    monkeypatch.setitem(registry._KINDS, "closing", ClosingBackend)
    monkeypatch.setattr(ClosingBackend, "made", 0)
    monkeypatch.setattr(ClosingBackend, "closed", 0)
    counts = lambda: (FakeLayout.made, FakeLayout.closed)  # noqa: E731
    with make_client("closing") as client:

        def done(**options):
            _, events = run(client, **options)
            assert events[-1]["data"]["status"] == "done", events[-1]
            return events

        done(profile="accurate")
        done(profile="accurate")
        assert counts() == (1, 0)  # reused by the next run of the profile
        done(profile="accurate", backend="keyword")  # only the classification backend changed
        assert ClosingBackend.closed == 1
        assert counts() == (1, 0)  # heron was not reloaded
        done(profile="fast")  # a fast run uses no engine: heron is released
        assert counts() == (1, 1)
        done(profile="accurate")
        assert counts() == (2, 1)
        done(profile="balanced")
        assert counts() == (2, 2)  # a profile without the layout model releases it
    assert counts() == (2, 2)
    assert FakeOcr.made == FakeOcr.closed  # none left open


def test_app_js_uses_textcontent_only(client):
    script = client.get("/static/app.js").text
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert sink not in script, sink
    page = client.get("/").text
    assert "<script>" not in page and " onclick=" not in page


def test_page_has_the_profile_selector_and_the_elements_tab(client):
    page = client.get("/").text
    assert 'id="profile"' in page and 'id="tab-elements"' in page
    assert 'id="panel-elements"' in page
    script = client.get("/static/app.js").text
    for name in ("elements_total", "elements_truncated", "summaries", "routes", "profile"):
        assert name in script


def pdf_with_image_regions() -> bytes:
    """Three pages; page 2 also carries two large pictures with no text in them."""
    doc = pymupdf.open(stream=pdf_bytes(), filetype="pdf")
    picture = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40))
    picture.set_rect(picture.irect, (200, 60, 60))
    page = doc[1]
    page.insert_image(pymupdf.Rect(72, 300, 272, 500), pixmap=picture)
    page.insert_image(pymupdf.Rect(300, 300, 500, 500), pixmap=picture)
    data = doc.tobytes()
    doc.close()
    return data


def test_routes_count_pages_with_image_regions(client):
    run_id, events = run(client, data=pdf_with_image_regions())
    routes = client.get(f"/api/runs/{run_id}").json()["routes"]
    # every page is read once: page 2 has a region decision instead of a page decision
    assert routes["pages"] == 3 and routes["tasks"] == {"text": 3}
    assert routes["regions"] == 2 and routes["region_pages"] == 1
    assert routes["needs_ocr"] == {"pages": 0, "merge_pages": 0, "regions": 2}
    end = next(e for e in events if e["stage"] == "route" and e["data"].get("pages") is not None)
    assert (end["data"]["pages"], end["data"]["regions"]) == (routes["pages"], routes["regions"])


def test_route_rows_are_named_the_same_in_the_timeline_and_the_status(client):
    script = client.get("/static/app.js").text
    assert script.count("Choose engines") == 1  # one place names the rooter's engine choices
    assert 'event.stage === "route" && event.data?.scope === "document"' in script
    assert "setStatus(`${stageMeta(event).name}…`)" in script  # the status reuses the row's name
    assert "const meta = stageMeta(event)" in script
    assert "STAGES[event.stage]?.name" not in script


def test_cut_element_text_gets_one_ellipsis_from_the_server_only(client):
    body = ("# Heading\n\n" + "word " * 120 + "end. Second sentence.\n").encode()
    run_id, _ = run(client, "long.md", body)
    text = next(
        e for e in client.get(f"/api/runs/{run_id}").json()["elements"] if e["kind"] == "paragraph"
    )["text"]
    assert text.endswith("…") and not text.endswith("……") and len(text) == 200
    script = client.get("/static/app.js").text
    assert "more ?" not in script and "text_chars" not in script  # the page adds none of its own


# --- element tags (v0.5 L10) --------------------------------------------------------------


class FakeTagger:
    """Tags every element with its family's ``other`` leaf and two functions; counts loads
    (``made``), calls and closes."""

    made = 0
    closed = 0

    def __init__(self):
        type(self).made += 1
        self.calls = []

    def tag(self, elements, family):
        from docrooter.elements import TagResult

        self.calls.append((len(elements), family))
        leaf = f"{family}.other"
        probs = {f"function_{i}": 0.1 for i in range(8)}
        probs.update({"<b>risk_disclosure</b>": 0.91, "definition": 0.6})
        return [
            {
                "topic": TagResult(
                    labels=[leaf],
                    probs={leaf: 0.7, f"{family}.x": 0.3},
                    confidence=0.7,
                    source="bundle:fake",
                ),
                "function": TagResult(
                    labels=["<b>risk_disclosure</b>", "definition"],
                    probs=probs,
                    confidence=0.6,
                    source="bundle:fake",
                ),
            }
            for _ in elements
        ]

    def close(self):
        type(self).closed += 1


@pytest.fixture
def tag_bundle(tmp_path, monkeypatch):
    from docrooter.tagger import element_tags

    path = tmp_path / "bundle"
    path.mkdir()
    (path / "tagger.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(FakeTagger, "made", 0)
    monkeypatch.setattr(FakeTagger, "closed", 0)
    monkeypatch.setattr(element_tags, "load_bundle", lambda spec: FakeTagger())
    return path


def test_server_tag_chips_textcontent(tag_bundle):
    with TestClient(server.create_app("keyword", element_tags=str(tag_bundle)),
                    base_url="http://127.0.0.1:8765") as client:  # fmt: skip
        assert client.get("/api/backends").json()["element_tags"] == "bundle"
        for _ in range(2):
            run_id, events = run(client, "paper.md", MARKDOWN, profile="balanced")
            assert events[-1]["data"]["status"] == "done"
            tag_end = [e for e in events if e["stage"] == "tag" and e["status"] == "end"]
            assert len(tag_end) == 1 and tag_end[0]["data"]["elements"] == 3
            tag_start = [e for e in events if e["stage"] == "tag" and e["status"] == "start"]
            assert tag_start[0]["data"]["engine"] == "bundle"  # the display name only
            assert str(tag_bundle) not in json.dumps(events)  # never the filesystem path
        assert FakeTagger.made == 1 and FakeTagger.closed == 0  # loaded once, kept for runs

        detail = client.get(f"/api/runs/{run_id}").json()
        tagged = [e for e in detail["elements"] if "tags" in e]
        assert [e["kind"] for e in tagged] == ["paragraph"] * 3  # headings get none
        assert all("tags" not in e for e in detail["elements"] if e["kind"] == "heading")
        family = detail["element_tags"]["family"]
        topic, function = tagged[0]["tags"]["topic"], tagged[0]["tags"]["function"]
        assert topic == {
            "labels": [f"{family}.other"],
            "confidence": 0.7,
            "probs": [{"id": f"{family}.other", "prob": 0.7}, {"id": f"{family}.x", "prob": 0.3}],
        }
        assert function["labels"] == ["<b>risk_disclosure</b>", "definition"]
        assert [p["id"] for p in function["probs"]][:2] == function["labels"]
        assert len(function["probs"]) == 5  # the top five, labels first
        assert detail["element_tags"] == {"source": "bundle", "family": family}

        fast_id, _ = run(client, profile="fast")
        fast = client.get(f"/api/runs/{fast_id}").json()
        assert fast["elements"] == [] and fast["element_tags"] is None

        script = client.get("/static/app.js").text
        assert "function tagChips(" in script
        assert 'h("span", { class: `chip tag-chip' in script  # text nodes, via h()
        assert "innerHTML" not in script and "insertAdjacentHTML" not in script
        assert 'tag: { name: "Tag elements"' in script
    assert FakeTagger.closed == 1  # closed at shutdown


def test_server_without_element_tags_has_no_tags(client):
    assert client.get("/api/backends").json()["element_tags"] is None
    run_id, events = run(client, "paper.md", MARKDOWN, profile="balanced")
    assert not [e for e in events if e["stage"] == "tag"]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["element_tags"] is None
    assert all("tags" not in e for e in detail["elements"])


def test_serve_element_tags_option(monkeypatch, tag_bundle, tmp_path):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: started.append(self.config.app))
    result = CliRunner().invoke(cli.app, ["serve", "--element-tags", str(tag_bundle)])
    assert result.exit_code == 0, result.output
    assert started[0].state.playground.element_tags == str(tag_bundle)

    missing = CliRunner().invoke(cli.app, ["serve", "--element-tags", str(tmp_path / "nope")])
    assert missing.exit_code == 1 and "no element-tagger bundle" in missing.output
    jev = CliRunner().invoke(cli.app, ["serve", "--element-tags", "teacher:jev:jev-1.13.0"])
    assert jev.exit_code == 1 and "2.3(b)" in jev.output
    assert len(started) == 1


# --- the completeness dial ------------------------------------------------------------------

RISK_ID = "annual_report.risk_factors"


def _combiner(feature_names, weights, intercept=-5.0):
    from docrooter.select.relevance import Combiner

    names = list(feature_names)
    return Combiner.from_dict(
        {
            "format": "docrooter-relevance-combiner",
            "version": 1,
            "feature_names": names,
            "l2": 1.0,
            "max_iter": 100,
            "mean": [0.0] * len(names),
            "scale": [1.0] * len(names),
            "coef": [weights.get(name, 0.0) for name in names],
            "intercept": intercept,
            "n_train": 10,
            "n_docs": 2,
            "converged": True,
        }
    )


def fake_selector(embedder="fakeenc:m"):
    """Hand-set combiners (the field one listens to p_topic, the others to sim) and one dial for
    the annual-report family: λ̂(0.9) = 0.5, capped for a target no calibration set reaches."""
    from docrooter.select import builtin_fields, calibration_of
    from docrooter.select.crc import DialTable
    from docrooter.select.relevance import FEATURE_NAMES
    from docrooter.select.selection import FALLBACK_FEATURES, Selector

    field = _combiner(FEATURE_NAMES, {"p_topic": 10.0})
    dial = DialTable(
        family="annual_report",
        n=100,
        n_docs=100,
        n_excluded=0,
        grid=(0.5, 0.9),
        curve_lambdas=[0.0, 0.5, 1.0],
        curve_risk=[0.0, 0.0, 1.0],
        combiner_fingerprint=field.fingerprint(),
    )
    fallback = _combiner(FALLBACK_FEATURES, {"sim": 10.0})
    return Selector(
        field=field,
        field_fallback=fallback,
        query=_combiner(FEATURE_NAMES, {"sim": 10.0}),
        query_fallback=fallback,
        dials={"annual_report": dial},
        calibrated_fields=calibration_of(builtin_fields("annual_report")),
        embedder=embedder,
    )


class FakeEncoder:
    """A model-free embedder: texts that mention risk, and everything else."""

    made = 0
    closed = 0
    encodes = 0

    def __init__(self, *args, **kwargs):
        type(self).made += 1

    def encode(self, texts, prompt=""):
        import numpy as np

        type(self).encodes += 1

        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])

    def close(self):
        type(self).closed += 1


@pytest.fixture
def selector_file(tmp_path, monkeypatch):
    monkeypatch.setitem(registry._KINDS, "fakeenc", FakeEncoder)
    monkeypatch.setattr(FakeEncoder, "made", 0)
    monkeypatch.setattr(FakeEncoder, "closed", 0)
    monkeypatch.setattr(FakeEncoder, "encodes", 0)
    path = tmp_path / "selector.json"
    fake_selector().save(path)
    return path


def make_selector_client(selector_file, backend="keyword"):
    app = server.create_app(backend, selector=str(selector_file))
    return TestClient(app, base_url="http://127.0.0.1:8765")


def tagged_analysis(n_filler=20):
    """An annual report of tagged paragraphs: three about risk, the rest filler."""
    from docrooter import Document
    from docrooter.classify.results import DocTypeResult, PathResult
    from docrooter.elements.models import Element, TagResult
    from docrooter.rooter.analysis import Analysis

    doc = Document.from_sections([{"title": "Factors", "level": 1}])
    items = [("heading", "Factors")]
    for k in range(n_filler):
        items.append(("paragraph", f"Filler sentence number {k} about the offices."))
        if k % 7 == 3:
            items.append(("paragraph", f"Interest rate risk could harm our results {k}."))
    elements = []
    for i, (kind, text) in enumerate(items):
        risky = "risk" in text.lower()
        leaf = f"annual_report.{'risk_factors' if risky else 'other'}"
        tags = None
        if kind == "paragraph":
            tags = {
                "topic": TagResult(labels=[leaf], probs={leaf: 1.0}, confidence=1.0, source="t")
            }
        elements.append(
            Element(
                id=f"e{i}",
                kind=kind,
                text=text,
                boxes=[],
                order=i,
                section_id="s1",
                tags=tags,
                **({"level": 1} if kind == "heading" else {}),
            )
        )
    doc.elements = elements
    doc_type = DocTypeResult(
        source="test", best=PathResult(node_ids=["financial", "financial.annual_report"])
    )
    return Analysis(document=doc, doc_type=doc_type, config=RouterConfig(profile="balanced"))


def inject(client, analysis=None):
    """A finished run with a hand-built analysis, as if the worker had kept it."""
    playground = client.app.state.playground
    run_obj = server.Run("injected", "report.pdf", "pdf", 1, {"profile": "balanced"})
    run_obj.status = "done"
    playground.runs[run_obj.id] = run_obj
    playground.keep(run_obj.id, analysis or tagged_analysis())
    return run_obj.id


def select_post(client, run_id, need=RISK_ID, completeness=0.9, **body):
    return client.post(
        f"/api/runs/{run_id}/select", json={"need": need, "completeness": completeness, **body}
    )


def test_select_endpoint_returns_selection(selector_file):
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        response = select_post(client, run_id)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["need"] == RISK_ID and body["completeness"] == 0.9
        assert body["guarantee"] is True and body["expected_recall"] == 0.9
        assert body["capped"] is False and body["doc_type_mismatch"] is False
        assert body["readout"].startswith("≈ 90% of relevant content kept · ")
        assert body["readout"].endswith("guaranteed on calibration-like documents")
        assert 0 < body["token_share"] < 1 and body["tokens"] > 0
        assert body["truncated"] is False and body["selected_total"] == len(body["element_ids"])
        analysis = client.app.state.playground.analyses[run_id]
        risky = [e.id for e in analysis.document.elements if "risk" in e.text.lower()]
        assert set(risky) <= set(body["element_ids"])
        assert len(body["element_ids"]) < len(analysis.document.elements)
        assert "chunks" not in body

        # a free-text query has no guarantee: its readout says it is an estimate
        query = select_post(client, run_id, "interest rate risk", 0.9).json()
        assert query["guarantee"] is False and "(estimate)" in query["readout"]
        assert query["need"] == "interest rate risk"

        # several fields of one family are one need
        both = select_post(client, run_id, [RISK_ID, "annual_report.legal_proceedings"]).json()
        assert both["need"] == f"{RISK_ID},annual_report.legal_proceedings"

        # c = 1.0 is the whole document, with no selector or model involved
        whole = select_post(client, run_id, RISK_ID, 1.0).json()
        assert whole["guarantee"] is True and whole["expected_recall"] == 1.0
        assert whole["selected_total"] == len(analysis.document.elements)

        # the embedder was loaded through the pool, once, and the pool holds it
        assert client.app.state.playground.backends.spec == "fakeenc:m"
        assert FakeEncoder.made == 1

        fields = client.get("/api/fields").json()
        assert set(fields) >= {"annual_report", "fund_document", "contract"}
        assert {"id": RISK_ID, "label": "Risk factors"}.items() <= next(
            f for f in fields["annual_report"] if f["id"] == RISK_ID
        ).items()
        assert set(fields["annual_report"][0]) == {"id", "label", "description"}
        assert client.get("/api/backends").json()["selector"] == {
            "loaded": True,
            "name": "selector.json",
            "embedder": "fakeenc:m",
        }
    assert FakeEncoder.closed == 1  # closed with the pool at shutdown


def test_select_runs_through_a_real_run_and_keeps_three_analyses(selector_file):
    with make_selector_client(selector_file) as client:
        ids = [run(client, f"{n}.md", MARKDOWN, profile="balanced")[0] for n in range(4)]
        evicted = select_post(client, ids[0], RISK_ID, 1.0)
        assert evicted.status_code == 404 and "run it again" in evicted.json()["detail"]
        assert select_post(client, "nope", RISK_ID, 1.0).status_code == 404
        kept = select_post(client, ids[3], RISK_ID, 1.0)
        assert kept.status_code == 200 and kept.json()["guarantee"] is True
        assert list(client.app.state.playground.analyses) == ids[1:]
        # a markdown document has no annual-report type: the field is a mismatch, no guarantee
        mismatch = select_post(client, ids[3], RISK_ID, 0.9).json()
        assert mismatch["doc_type_mismatch"] is True and mismatch["guarantee"] is False
        # the detail names the family whose fields belong to the document
        assert client.get(f"/api/runs/{ids[3]}").json()["field_family"] == "research_paper"
        fast_id, _ = run(client, profile="fast")
        fast = select_post(client, fast_id, RISK_ID, 1.0)
        assert fast.status_code == 409 and "Balanced" in fast.json()["detail"]


def test_select_without_a_selector_serves_the_whole_document_only(client):
    run_id = inject(client)
    refused = select_post(client, run_id, RISK_ID, 0.9)
    assert refused.status_code == 409
    assert refused.json()["detail"] == "no selector: fit one with docrooter fit-selector"
    assert client.get("/api/backends").json()["selector"] == {
        "loaded": False,
        "name": None,
        "embedder": None,
    }
    whole = select_post(client, run_id, RISK_ID, 1.0)
    assert whole.status_code == 200 and whole.json()["guarantee"] is True


@pytest.mark.parametrize(
    "body",
    [
        {"need": RISK_ID},  # no completeness
        {"need": RISK_ID, "completeness": 1.5},
        {"need": RISK_ID, "completeness": -0.1},
        {"need": RISK_ID, "completeness": "0.9"},
        {"need": RISK_ID, "completeness": True},
        '{"need": "annual_report.risk_factors", "completeness": NaN}',  # raw JSON text
        "not json",
        {"need": "x" * 501, "completeness": 0.9},
        {"need": "  ", "completeness": 0.9},
        {"need": [], "completeness": 0.9},
        {"need": ["annual_report.risk_factors", "a query"], "completeness": 0.9},
        {"need": [RISK_ID, "fund_document.fees"], "completeness": 0.9},
        {"need": 5, "completeness": 0.9},
        {"need": RISK_ID, "completeness": 0.9, "budget": 0},
        {"need": RISK_ID, "completeness": 1.0, "budget": 100},
        [1, 2],
    ],
)
def test_select_bad_requests_are_400(selector_file, body):
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        url = f"/api/runs/{run_id}/select"
        response = (
            client.post(url, content=body) if isinstance(body, str) else client.post(url, json=body)
        )
        assert response.status_code == 400, response.text
        assert isinstance(response.json()["detail"], str)
    assert FakeEncoder.made == 0  # refused before any model was loaded


def test_select_query_may_be_500_characters(selector_file):
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        assert select_post(client, run_id, "x" * 500, 1.0).status_code == 200


def test_select_payload_bounded(selector_file, monkeypatch):
    with make_selector_client(selector_file) as client:
        run_id = inject(client, tagged_analysis(n_filler=80))
        total = len(client.app.state.playground.analyses[run_id].document.elements)
        monkeypatch.setattr(server, "MAX_TOC_ENTRIES", 5)
        for completeness in (1.0, 0.9):
            response = select_post(client, run_id, RISK_ID, completeness)
            body = response.json()
            assert len(body["element_ids"]) <= 5
            assert body["truncated"] is (body["selected_total"] > 5)
        whole = select_post(client, run_id, RISK_ID, 1.0)
        body = whole.json()
        assert body["selected_total"] == total and len(body["element_ids"]) == 5
        assert body["truncated"] is True
        assert body["element_ids"] == [f"e{i}" for i in range(5)]  # the first, in reading order
        assert "chunks" not in body
        assert "Filler sentence" not in whole.text  # no element text, only ids
        assert set(body) == {
            "readout",
            "need",
            "completeness",
            "threshold",
            "tokens",
            "token_share",
            "expected_recall",
            "guarantee",
            "capped",
            "doc_type_mismatch",
            "relevant_expected",
            "over_budget",
            "uncalibrated_field",
            "element_ids",
            "selected_total",
            "truncated",
        }


def test_dial_presets(selector_file, client):
    page = client.get("/").text
    for element in ('id="select-slider"', 'min="0.5"', 'max="1"', 'step="0.01"'):
        assert element in page
    for element in ('id="select-field"', 'id="select-query"', 'id="select-readout"'):
        assert element in page
    script = client.get("/static/app.js").text
    for name, value in (
        ("Lean", 0.8),
        ("Balanced", 0.9),
        ("Thorough", 0.98),
        ("Whole document", 1.0),
    ):
        assert f'{{ name: "{name}", value: {value} }}' in script
    assert "SELECT_DEBOUNCE_MS = 250;" in script  # the readout is debounced

    # the presets are monotone in tokens: a higher completeness never reads less
    with make_selector_client(selector_file) as selecting:
        run_id = inject(selecting, tagged_analysis(n_filler=60))
        for need in (RISK_ID, "interest rate risk"):
            tokens = [
                select_post(selecting, run_id, need, value).json()["tokens"]
                for value in (0.8, 0.9, 0.98, 1.0)
            ]
            assert tokens == sorted(tokens), (need, tokens)
        whole = select_post(selecting, run_id, RISK_ID, 1.0).json()
        assert whole["tokens"] == tokens[-1] and whole["token_share"] == 1.0


def test_textcontent_only(client):
    script = client.get("/static/app.js").text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script, sink
    # the readout and badges are set as text; selected elements get a class, never markup
    assert '$("select-readout").textContent' in script
    assert "function selectBadges(" in script and 'h("span", { class: "badge' in script
    assert 'classList.toggle("is-selected"' in script
    assert "is-selected" in client.get("/static/style.css").text
    page = client.get("/").text
    assert (
        'id="select-panel"' in page and 'id="el-selected-only"' not in page
    )  # built by the script
    assert "<script>" not in page and " onclick=" not in page


def test_select_embedder_follows_the_single_model_rule(selector_file, monkeypatch):
    monkeypatch.setitem(registry._KINDS, "closing", ClosingBackend)
    monkeypatch.setattr(ClosingBackend, "made", 0)
    monkeypatch.setattr(ClosingBackend, "closed", 0)
    with make_selector_client(selector_file, backend="closing") as client:
        run_id, events = run(client, "a.md", MARKDOWN)
        assert events[-1]["data"]["status"] == "done"
        assert (ClosingBackend.made, ClosingBackend.closed) == (1, 0)
        for _ in range(2):  # the encoder replaces the classifier in the pool, then is reused
            assert select_post(client, run_id, RISK_ID, 0.9).status_code == 200
        assert (ClosingBackend.closed, FakeEncoder.made, FakeEncoder.closed) == (1, 1, 0)
        assert client.app.state.playground.backends.spec == "fakeenc:m"
        _, events = run(client, "b.md", MARKDOWN)  # the next run takes its backend back
        assert events[-1]["data"]["status"] == "done"
        assert (FakeEncoder.closed, ClosingBackend.made) == (1, 2)


def test_selector_is_found_in_the_element_tags_bundle(tag_bundle, selector_file, monkeypatch):
    import shutil

    shutil.copy(selector_file, tag_bundle / "selector.json")
    app = server.create_app("keyword", element_tags=str(tag_bundle))
    assert app.state.playground.selector is not None
    app = server.create_app("keyword", element_tags=str(tag_bundle), selector=str(selector_file))
    assert app.state.playground.selector_name == "selector.json"
    with pytest.raises(Exception, match="no selector"):
        server.create_app("keyword", selector=str(selector_file.parent / "nope.json"))


def test_serve_selector_option(monkeypatch, selector_file, tmp_path):
    import uvicorn

    started = []
    monkeypatch.setattr(uvicorn.Server, "run", lambda self: started.append(self.config.app))
    result = CliRunner().invoke(cli.app, ["serve", "--selector", str(selector_file)])
    assert result.exit_code == 0, result.output
    assert started[0].state.playground.selector is not None
    missing = CliRunner().invoke(cli.app, ["serve", "--selector", str(tmp_path / "nope.json")])
    assert missing.exit_code == 1 and "no selector" in missing.output
    assert len(started) == 1


def test_select_budget_is_used_below_one(selector_file):
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        roomy = select_post(client, run_id, RISK_ID, 0.9, budget=200).json()
        assert roomy["guarantee"] is False and roomy["tokens"] <= 200
        assert roomy["over_budget"] is False and "guaranteed" not in roomy["readout"]
        tiny = select_post(client, run_id, RISK_ID, 0.9, budget=1).json()
        assert tiny["over_budget"] is True and tiny["guarantee"] is False
        assert "exceeds the budget" in tiny["readout"]
        assert tiny["selected_total"] == 1  # the most relevant element, alone


def test_query_needs_an_embedder_in_the_selector(tmp_path, monkeypatch):
    monkeypatch.setitem(registry._KINDS, "fakeenc", FakeEncoder)
    bare = tmp_path / "bare.json"
    fake_selector(embedder=None).save(bare)
    with make_selector_client(bare) as client:
        assert client.get("/api/backends").json()["selector"] == {
            "loaded": True,
            "name": "bare.json",
            "embedder": None,
        }
        run_id = inject(client)
        refused = select_post(client, run_id, "interest rate risk", 0.9)
        assert refused.status_code == 409
        assert refused.json()["detail"] == (
            "free-text queries need an embedder; this selector was fitted without one"
        )
        # fields still work (on the fallback combiner), and the whole document needs nothing
        assert select_post(client, run_id, RISK_ID, 0.9).status_code == 200
        assert select_post(client, run_id, "interest rate risk", 1.0).status_code == 200
        page = client.get("/static/app.js").text
        assert "query.disabled = hasSelector && !hasEmbedder" in page


def test_select_features_are_cached_per_run(selector_file):
    with make_selector_client(selector_file) as client:
        playground = client.app.state.playground
        run_id = inject(client)
        first = select_post(client, run_id, RISK_ID, 0.9)
        assert first.status_code == 200
        encodes = FakeEncoder.encodes
        assert encodes >= 2  # the elements, and the need

        async def refuse(fn):  # a cache hit must not go through the worker queue
            raise AssertionError("re-thresholding went to the worker")

        monkey = playground.on_worker
        playground.on_worker = refuse
        try:
            for completeness in (0.5, 0.8, 0.98):
                again = select_post(client, run_id, RISK_ID, completeness)
                assert again.status_code == 200, again.text
        finally:
            playground.on_worker = monkey
        assert FakeEncoder.encodes == encodes  # nothing was embedded again
        assert select_post(client, run_id, RISK_ID, 0.9).json() == first.json()

        other = select_post(client, run_id, "interest rate risk", 0.9)  # a new need: one encode
        assert other.status_code == 200 and FakeEncoder.encodes == encodes + 1
        assert FakeEncoder.made == 1

        # the caches go with the analysis they belong to
        for n in range(3):
            playground.keep(f"other{n}", tagged_analysis())
        assert run_id not in playground.analyses and run_id not in playground.scored


def test_select_loads_the_tagger_only_for_untagged_elements(tag_bundle, selector_file):
    app = server.create_app("keyword", element_tags=str(tag_bundle), selector=str(selector_file))
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        tagged_id = inject(client)  # every content element has its tags already
        assert select_post(client, tagged_id, RISK_ID, 0.9).status_code == 200
        assert FakeTagger.made == 0

        bare = tagged_analysis()
        for element in bare.document.elements:
            element.tags = None
        bare_id = "bare"
        playground = client.app.state.playground
        playground.runs[bare_id] = server.Run(bare_id, "b.pdf", "pdf", 1, {"profile": "balanced"})
        playground.runs[bare_id].status = "done"
        playground.keep(bare_id, bare)
        assert select_post(client, bare_id, RISK_ID, 0.9).status_code == 200
        assert FakeTagger.made == 1


def test_a_settled_task_is_skipped_by_the_worker():
    loop = asyncio.new_event_loop()
    try:
        called = []
        cancelled = loop.create_future()
        cancelled.cancel()
        done = loop.create_future()
        done.set_result(1)
        for future in (cancelled, done):
            server._Task(lambda: called.append(1), loop, future).run()
        assert called == []
        live = loop.create_future()
        server._Task(lambda: called.append(1), loop, live).run()
        assert called == [1]
    finally:
        loop.close()


def test_select_body_is_capped(selector_file):
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        big = {"need": ["annual_report.risk_factors"] * 5000, "completeness": 0.9}
        response = client.post(f"/api/runs/{run_id}/select", json=big)
        assert response.status_code == 413


def test_select_on_a_running_run_says_so(client):
    run_obj = server.Run("busy", "a.pdf", "pdf", 1, {"profile": "balanced"})
    run_obj.status = "running"
    client.app.state.playground.runs["busy"] = run_obj
    response = select_post(client, "busy", RISK_ID, 1.0)
    assert response.status_code == 409 and "still running" in response.json()["detail"]


def test_no_estimate_badge_without_a_relevance_estimate(client):
    script = client.get("/static/app.js").text
    assert "!selection.guarantee && selection.expected_recall !== null" in script


def test_uncalibrated_field_has_a_badge_and_reaches_the_page(client):
    script = client.get("/static/app.js").text
    assert "selection.uncalibrated_field" in script
    assert 'uncalibrated: "field not calibrated"' in script


def test_select_endpoint_reports_a_field_the_dial_was_not_calibrated_on(selector_file):
    data = json.loads(selector_file.read_text(encoding="utf-8"))
    del data["calibrated_fields"]  # an old selector.json records none
    selector_file.write_text(json.dumps(data), encoding="utf-8")
    with make_selector_client(selector_file) as client:
        run_id = inject(client)
        body = select_post(client, run_id).json()
        assert body["guarantee"] is False and body["uncalibrated_field"] is True
        assert body["readout"].endswith("field not calibrated (estimate)")
        whole = select_post(client, run_id, RISK_ID, 1.0).json()
        assert whole["guarantee"] is True and whole["uncalibrated_field"] is False
