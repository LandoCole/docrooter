"""The playground server: SSE event order, result schema, upload limits and checks, temp-file
cleanup, history, backend switching and the static page. Torch-free: keyword backend and
generated PDFs."""

import asyncio
import json
import logging
import os
import tempfile
import threading
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("python_multipart")
pymupdf = pytest.importorskip("pymupdf")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from docrooter import cli  # noqa: E402
from docrooter.backends import KeywordBackend, registry  # noqa: E402
from docrooter.ui import server  # noqa: E402

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
    _, events = run(client, normalize="true", mode="view")
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
    _, events = run(client, data=b"%PDF-1.7 this is not really a PDF")
    assert [e["stage"] for e in events[-2:]] == ["error", "job"]
    assert events[-1]["data"]["status"] == "error"
    assert all(temp not in events[-1]["data"]["error"] for temp in temp_paths)
    assert temp_paths and not any(os.path.exists(p) for p in temp_paths)


def test_markdown_upload_runs_without_pdf_stages(client):
    run_id, events = run(client, "paper.md", MARKDOWN)
    starts = [e["stage"] for e in events if e["status"] == "start"]
    assert starts == ["upload", "assemble", "view", "classify"]
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


def test_pdfs_are_read_with_a_page_cap(client, monkeypatch):
    seen = {}
    real = server.load

    def spy(path, **kwargs):
        seen.update(kwargs)
        return real(path, **kwargs)

    monkeypatch.setattr(server, "load", spy)
    run(client, engine="pdfplumber")
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
    [b"12abc", b"-1", b"\xb2", b"1\xb9"],  # "\xb2" is latin-1 for "2" superscript: isdigit()
)
def test_malformed_content_length_is_400(content_length):
    status = asyncio.run(drive_post([b"x"], content_length=content_length))
    assert status == 400


def test_non_ascii_last_event_id_is_ignored(client):
    run_id, events = run(client)
    again, ids = stream(client, run_id, headers={"Last-Event-ID": b"\xb2"})
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
