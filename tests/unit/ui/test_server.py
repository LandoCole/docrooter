"""The playground server: SSE event order, result schema, upload limits and checks, temp-file
cleanup, history, backend switching and the static page. Torch-free: keyword backend and
generated PDFs."""

import json
import os
import tempfile
import threading

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


@pytest.fixture
def client():
    with TestClient(server.create_app("keyword")) as test_client:
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
    with TestClient(server.create_app("closing")) as client:
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
    with TestClient(server.create_app("closing")) as client:
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
    assert client.get("/static/server.py").status_code == 404
    assert client.get("/static/..%2Fserver.py").status_code == 404


def test_page_uses_no_external_assets(client):
    for path in ("/", "/static/app.js", "/static/style.css"):
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
    result = CliRunner().invoke(cli.app, ["serve", "--host", "0.0.0.0", "--port", "8799"])
    assert result.exit_code == 0, result.output
    assert started == ["0.0.0.0"]
    assert "warning: binding to 0.0.0.0" in result.output

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
