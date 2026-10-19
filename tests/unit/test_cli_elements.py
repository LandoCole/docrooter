"""``docrooter elements``, ``process`` and ``config show``: the table and JSON output, the
analysis file, chunks, the profile and config options, and the errors. Keyword backend and a
generated PDF; no model is loaded and no OCR engine is used."""

import json

import pytest
from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.ocr import registry as ocr_registry

pymupdf = pytest.importorskip("pymupdf")

runner = CliRunner()

BODY = "This sentence is ordinary body text that stays small and keeps the body size at eleven."
CONFIG = 'ocr = "none"\ndoc_type = "keyword"\nsections = "keyword"\n'


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch):
    monkeypatch.setattr(ocr_registry, "resolve_auto_ocr", lambda: None)


@pytest.fixture
def pdf(tmp_path):
    doc = pymupdf.open()
    for n in range(2):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 90), f"Chapter {n + 1}", fontsize=22, fontname="hebo")
        y = 130
        for k in range(4):
            page.insert_text((72, y), f"{BODY} Line {k}.", fontsize=11, fontname="helv")
            y += 15
    path = tmp_path / "doc.pdf"
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "rooter.toml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_elements_cli_table_and_json(pdf, config):
    table = runner.invoke(app, ["elements", str(pdf), "--config", str(config)])
    assert table.exit_code == 0, table.output
    for header in ("Page", "Kind", "Text", "Section"):
        assert header in table.output
    assert "heading" in table.output and "paragraph" in table.output
    assert "Chapter 1" in table.output

    result = runner.invoke(app, ["elements", str(pdf), "--config", str(config), "--json"])
    assert result.exit_code == 0, result.output
    found = json.loads(result.output)
    assert [el["order"] for el in found] == list(range(len(found)))
    assert {el["kind"] for el in found} >= {"heading", "paragraph"}
    assert found[0]["boxes"] and set(found[0]) >= {"id", "kind", "text", "section_id"}

    only = runner.invoke(
        app, ["elements", str(pdf), "--config", str(config), "--json", "--kinds", "heading"]
    )
    kinds = {el["kind"] for el in json.loads(only.output)}
    assert kinds == {"heading"}


def test_elements_cli_rejects_bad_options(pdf, config):
    bad_kind = runner.invoke(app, ["elements", str(pdf), "--kinds", "heading,banana"])
    assert bad_kind.exit_code == 2 and "banana" in bad_kind.output
    bad_profile = runner.invoke(app, ["elements", str(pdf), "--profile", "turbo"])
    assert bad_profile.exit_code == 2 and "turbo" in bad_profile.output
    fast = runner.invoke(app, ["elements", str(pdf), "--profile", "fast", "--config", str(config)])
    assert fast.exit_code == 1 and "builds no elements" in fast.output


def test_process_cli_writes_analysis_json(pdf, config, tmp_path):
    out = tmp_path / "analysis.json"
    result = runner.invoke(app, ["process", str(pdf), "--config", str(config), "--out", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) >= {"document", "doc_type", "sections", "summaries", "config", "timings"}
    assert data["config"]["profile"] == "balanced" and data["config"]["doc_type"] == "keyword"
    assert data["document"]["elements"] and "chunks" not in data
    assert "embeddings" not in data

    with_chunks = tmp_path / "chunks.json"
    result = runner.invoke(
        app, ["process", str(pdf), "--config", str(config), "-o", str(with_chunks), "--chunks"]
    )
    assert result.exit_code == 0, result.output
    chunks = json.loads(with_chunks.read_text(encoding="utf-8"))["chunks"]
    assert chunks and set(chunks[0]) >= {"id", "text", "heading_path", "element_ids"}


def test_process_cli_prints_json_without_out(pdf, config):
    result = runner.invoke(app, ["process", str(pdf), "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["config"]["profile"] == "balanced"


def test_process_cli_chunks_need_elements(pdf, config):
    result = runner.invoke(
        app, ["process", str(pdf), "--profile", "fast", "--config", str(config), "--chunks"]
    )
    assert result.exit_code == 1
    assert result.output.startswith("error:") and "no elements" in result.output


def test_config_show_resolves_profile(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    default = runner.invoke(app, ["config", "show"])
    assert default.exit_code == 0, default.output
    assert 'profile = "balanced"' in default.output and 'layout = "heuristic"' in default.output
    assert "max_pages" not in default.output  # unset: TOML has no null

    accurate = runner.invoke(app, ["config", "show", "--profile", "accurate"])
    assert 'layout = "heron"' in accurate.output and 'escalate = "none"' in accurate.output
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    keyed = runner.invoke(app, ["config", "show", "--profile", "accurate"])
    assert 'escalate = "jev:jev-1.13.0"' in keyed.output
    fast = runner.invoke(app, ["config", "show", "--profile", "fast"])
    assert 'layout = "none"' in fast.output and 'summaries = "none"' in fast.output
    assert runner.invoke(app, ["config", "show", "--profile", "turbo"]).exit_code == 2


def test_config_show_output_loads_back_as_a_config(tmp_path, monkeypatch):
    from docrooter.rooter import RouterConfig

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    shown = runner.invoke(app, ["config", "show", "--profile", "accurate"]).output
    path = tmp_path / "saved.toml"
    path.write_text(shown, encoding="utf-8")
    assert RouterConfig.from_file(path) == RouterConfig(profile="accurate")
