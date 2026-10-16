"""``docrooter doctype`` and ``docrooter normalize`` on Markdown inputs with the keyword backend."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docrooter.cli import app

runner = CliRunner()
WIDE = {"COLUMNS": "400"}  # keep rich from wrapping paths

TAXONOMY = {
    "nodes": [
        {
            "id": "financial",
            "label": "Financial",
            "description": "Money matters",
            "children": [
                {
                    "id": "financial.tax",
                    "label": "Tax",
                    "description": "income tax deferred tax revenue earnings",
                },
                {
                    "id": "financial.audit",
                    "label": "Audit",
                    "description": "audit auditor findings controls opinion",
                },
            ],
        },
        {
            "id": "legal",
            "label": "Legal",
            "description": "Legal matters",
            "children": [
                {
                    "id": "legal.contract",
                    "label": "Contract",
                    "description": "agreement parties obligations termination clause",
                },
                {
                    "id": "legal.statute",
                    "label": "Statute",
                    "description": "legislation act section enacted lawmakers",
                },
            ],
        },
    ]
}

TAX_DOC = "# Income tax\n\nIncome tax and deferred tax on revenue earnings.\n"
AUDIT_DOC = "# Audit findings\n\nThe auditor reports audit findings on controls.\n"
CONTRACT_DOC = "# Agreement\n\nThe parties agree to obligations; termination clause applies.\n"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "docs"
    (root / "sub").mkdir(parents=True)
    (root / "tax.md").write_text(TAX_DOC, encoding="utf-8")
    (root / "audit.txt").write_text(AUDIT_DOC, encoding="utf-8")
    (root / "sub" / "contract.md").write_text(CONTRACT_DOC, encoding="utf-8")
    (root / "ignored.csv").write_text("a,b\n", encoding="utf-8")
    return root


@pytest.fixture
def taxonomy(tmp_path: Path) -> str:
    path = tmp_path / "tax.json"
    path.write_text(json.dumps(TAXONOMY), encoding="utf-8")
    return str(path)


def _doctype(*args: str):
    return runner.invoke(app, ["doctype", *args, "-b", "keyword", "--workers", "1"], env=WIDE)


def _records(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {Path(r["source"]).name: r for r in map(json.loads, lines)}


def test_directory_is_expanded_and_every_file_listed(corpus, taxonomy):
    result = _doctype(str(corpus), "-t", taxonomy)
    assert result.exit_code == 0, result.output
    for name in ("tax.md", "audit.txt", "contract.md"):
        assert name in result.output
    assert "ignored.csv" not in result.output
    for header in ("File", "Type", "Confidence", "Parse s", "Classify s"):
        assert header in result.output
    assert "Financial > Tax" in result.output
    assert "Legal > Contract" in result.output


def test_jsonl_records(corpus, taxonomy, tmp_path):
    out = tmp_path / "out.jsonl"
    result = _doctype(str(corpus), "-t", taxonomy, "--jsonl", str(out))
    assert result.exit_code == 0, result.output
    records = _records(out)
    assert sorted(records) == ["audit.txt", "contract.md", "tax.md"]
    tax = records["tax.md"]
    assert set(tax) == {
        "source",
        "type",
        "label",
        "path",
        "confidence",
        "leaf_probs",
        "timings",
        "error",
    }
    assert tax["type"] == "financial.tax"
    assert tax["label"] == "Tax"
    assert tax["path"] == ["financial", "financial.tax"]
    assert 0 < tax["confidence"] <= 1
    assert tax["leaf_probs"]["financial.tax"] == pytest.approx(tax["confidence"])
    assert set(tax["timings"]) == {"parse", "classify"}
    assert tax["error"] is None
    assert records["contract.md"]["type"] == "legal.contract"
    assert records["audit.txt"]["type"] == "financial.audit"


def test_keep_ancestor_filters_table_and_jsonl(corpus, taxonomy, tmp_path):
    out = tmp_path / "out.jsonl"
    result = _doctype(str(corpus), "-t", taxonomy, "--keep", "financial", "--jsonl", str(out))
    assert result.exit_code == 0, result.output
    assert "tax.md" in result.output and "audit.txt" in result.output
    assert "contract.md" not in result.output
    assert sorted(_records(out)) == ["audit.txt", "tax.md"]


def test_keep_leaf_id(corpus, taxonomy):
    result = _doctype(str(corpus), "-t", taxonomy, "--keep", "legal.contract")
    assert result.exit_code == 0, result.output
    assert "contract.md" in result.output
    assert "tax.md" not in result.output


def test_keep_unknown_id_lists_valid_ids(corpus, taxonomy):
    result = _doctype(str(corpus), "-t", taxonomy, "--keep", "nope")
    assert result.exit_code != 0
    flat = " ".join(result.output.split())
    assert "nope" in flat and "financial.tax" in flat and "legal" in flat


def test_paths_only_prints_just_kept_paths(corpus, taxonomy):
    result = runner.invoke(
        app,
        ["doctype", str(corpus), "-t", taxonomy, "-b", "keyword", "--workers", "1"]
        + ["--keep", "financial", "--paths-only"],
    )
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert sorted(Path(line).name for line in lines) == ["audit.txt", "tax.md"]
    assert all(Path(line).is_file() for line in lines)


def test_unreadable_file_gives_error_row_and_exit_0(corpus, taxonomy, tmp_path):
    (corpus / "broken.md").write_bytes(b"\xff\xfe\x00bad \x80\x81")
    out = tmp_path / "out.jsonl"
    result = _doctype(str(corpus), "-t", taxonomy, "--jsonl", str(out))
    assert result.exit_code == 0, result.output
    assert "broken.md" in result.output and "error:" in result.output
    records = _records(out)
    broken = records["broken.md"]
    assert broken["error"]
    assert broken["type"] is None and broken["confidence"] is None
    assert records["tax.md"]["error"] is None and records["tax.md"]["type"] == "financial.tax"


def test_error_files_are_not_kept_but_reported_on_stderr(corpus, taxonomy):
    (corpus / "broken.md").write_bytes(b"\xff\xfe\x00bad \x80\x81")
    result = _doctype(str(corpus), "-t", taxonomy, "--keep", "financial", "--paths-only")
    assert result.exit_code == 0, result.output
    assert "warning:" in result.output  # CliRunner mixes stderr into output
    assert "broken.md" in result.output


def test_builtin_taxonomy_plumbing(corpus):
    result = _doctype(str(corpus / "tax.md"), "--keep", "financial")
    assert result.exit_code == 0, result.output
    assert "tax.md" in result.output


def test_unknown_builtin_taxonomy(corpus):
    result = _doctype(str(corpus), "-t", "builtin:nope")
    assert result.exit_code == 1
    assert "error:" in result.output and "available" in " ".join(result.output.split())


def test_empty_directory_is_an_error(tmp_path, taxonomy):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _doctype(str(empty), "-t", taxonomy)
    assert result.exit_code == 1
    assert "no .pdf, .md or .txt files" in " ".join(result.output.split())


def test_invalid_mode_and_engine(corpus, taxonomy):
    assert _doctype(str(corpus), "-t", taxonomy, "--mode", "bogus").exit_code == 2
    assert _doctype(str(corpus), "-t", taxonomy, "--engine", "bogus").exit_code == 2


def test_threshold_gates_the_label(corpus, taxonomy):
    result = _doctype(str(corpus), "-t", taxonomy, "--threshold", "0.999")
    assert result.exit_code == 0, result.output
    assert "low_confidence" in result.output


def test_sections_mode(corpus, taxonomy, tmp_path):
    out = tmp_path / "out.jsonl"
    result = _doctype(str(corpus), "-t", taxonomy, "--mode", "sections", "--jsonl", str(out))
    assert result.exit_code == 0, result.output
    assert _records(out)["tax.md"]["type"] == "financial.tax"


def test_two_workers_match_serial(corpus, taxonomy, tmp_path):
    serial, parallel = tmp_path / "serial.jsonl", tmp_path / "parallel.jsonl"
    assert _doctype(str(corpus), "-t", taxonomy, "--jsonl", str(serial)).exit_code == 0
    result = runner.invoke(
        app,
        ["doctype", str(corpus), "-t", taxonomy, "-b", "keyword", "--workers", "2"]
        + ["--jsonl", str(parallel)],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    a, b = _records(serial), _records(parallel)
    assert list(a) == list(b)
    for name in a:
        assert a[name]["type"] == b[name]["type"]
        assert a[name]["confidence"] == pytest.approx(b[name]["confidence"])


# --- normalize ----------------------------------------------------------------------------

PAPER = (
    "# Abstract\n\nWe summarise the paper.\n\n"
    "# Introduction\n\nBackground and motivation.\n\n"
    "## Related Work\n\nPrior studies.\n\n"
    "# Conclusion\n\nWe conclude.\n"
)


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    path = tmp_path / "paper.md"
    path.write_text(PAPER, encoding="utf-8")
    return path


def test_normalize_json(paper):
    result = runner.invoke(
        app,
        [
            "normalize",
            str(paper),
            "--doc-type",
            "research.research_paper",
            "-b",
            "keyword",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["doc_type"] == "research.research_paper"
    assert data["taxonomy"] == "section_types/research_paper"
    titles = [s["title"] for s in data["sections"].values()]
    assert titles == ["Abstract", "Introduction", "Related Work", "Conclusion"]
    assert all(s["canonical"] for s in data["sections"].values())


def test_normalize_levels_restricts_sections(paper):
    result = runner.invoke(
        app,
        ["normalize", str(paper), "--doc-type", "research.research_paper", "-b", "keyword"]
        + ["--levels", "1", "--json"],
    )
    assert result.exit_code == 0, result.output
    titles = [s["title"] for s in json.loads(result.output)["sections"].values()]
    assert titles == ["Abstract", "Introduction", "Conclusion"]


def test_normalize_table(paper):
    result = runner.invoke(
        app,
        ["normalize", str(paper), "--doc-type", "research.research_paper", "-b", "keyword"],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    for header in ("Section", "Canonical", "Score"):
        assert header in result.output
    assert "Related Work" in result.output
    assert "section_types/research_paper" in result.output


def test_normalize_classifies_doc_type_when_absent(paper):
    result = runner.invoke(app, ["normalize", str(paper), "-b", "keyword", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["doc_type_confidence"] is not None


def test_normalize_rejects_unknown_doc_type_and_bad_levels(paper):
    bad_type = runner.invoke(app, ["normalize", str(paper), "--doc-type", "nope", "-b", "keyword"])
    assert bad_type.exit_code == 2
    assert "financial.annual_report" in " ".join(bad_type.output.split())
    bad_levels = runner.invoke(app, ["normalize", str(paper), "--levels", "a,b", "-b", "keyword"])
    assert bad_levels.exit_code == 2


def _write_pdf(path: Path, title: str, body: str) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_text((72, 90), title, fontsize=22, fontname="hebo")
    for k in range(4):
        page.insert_text((72, 140 + 15 * k), f"{body} Line {k}.", fontsize=11, fontname="helv")
    pdf.save(str(path))
    pdf.close()
    return path


@pytest.fixture
def pdf_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "pdfs"
    root.mkdir()
    _write_pdf(root / "tax.pdf", "Income tax", "Income tax and deferred tax on revenue earnings.")
    _write_pdf(root / "audit.pdf", "Audit findings", "The auditor reports audit findings.")
    _write_pdf(root / "contract.pdf", "Agreement", "The parties agree to a termination clause.")
    (root / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    return root


def test_real_pdfs_through_two_worker_processes(pdf_corpus, taxonomy, tmp_path):
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app,
        ["doctype", str(pdf_corpus), "-t", taxonomy, "-b", "keyword", "--workers", "2"]
        + ["--engine", "pymupdf", "--jsonl", str(out)],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    records = _records(out)
    assert records["tax.pdf"]["type"] == "financial.tax"
    assert records["audit.pdf"]["type"] == "financial.audit"
    assert records["contract.pdf"]["type"] == "legal.contract"
    assert records["broken.pdf"]["error"] and records["broken.pdf"]["type"] is None


class _BrokenFuture:
    def result(self):
        from concurrent.futures.process import BrokenProcessPool

        raise BrokenProcessPool("a worker died")


class _BrokenPool:
    """A process pool whose worker dies: every future raises ``BrokenProcessPool``."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def submit(self, fn, *args):
        return _BrokenFuture()


def test_broken_process_pool_falls_back_to_parsing_in_process(
    corpus, taxonomy, tmp_path, monkeypatch
):
    import concurrent.futures

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", _BrokenPool)
    out = tmp_path / "out.jsonl"
    result = runner.invoke(
        app,
        ["doctype", str(corpus), "-t", taxonomy, "-b", "keyword", "--workers", "2"]
        + ["--jsonl", str(out)],
        env=WIDE,
    )
    assert result.exit_code == 0, result.output
    assert "parsing the remaining 3 file(s) in this process" in " ".join(result.output.split())
    records = _records(out)
    assert records["tax.md"]["type"] == "financial.tax" and records["tax.md"]["error"] is None
    assert records["contract.md"]["type"] == "legal.contract"


def test_file_that_crashes_again_in_process_is_an_error_row(corpus, taxonomy, tmp_path):
    from docrooter import cli

    def crash(path, engine):
        if path.endswith("tax.md"):
            raise MemoryError("parser crashed")
        return real(path, engine)

    real = cli._parse_one
    parsed = cli._parse_remaining(
        [str(corpus / "tax.md"), str(corpus / "audit.txt")], "auto", crash
    )
    assert parsed[0][0] is None and parsed[0][2] == "parser crashed"
    assert parsed[1][0] is not None and parsed[1][2] is None


def test_directory_expansion_skips_hidden_and_tool_directories(corpus, taxonomy):
    for skipped in (".git", ".venv", "node_modules", "__pycache__", "sub/.hidden"):
        (corpus / skipped).mkdir(parents=True, exist_ok=True)
        (corpus / skipped / "noise.md").write_text(TAX_DOC, encoding="utf-8")
    from docrooter.cli import _expand_paths

    names = [p.relative_to(corpus).as_posix() for p in _expand_paths([corpus])]
    assert names == ["audit.txt", "sub/contract.md", "tax.md"]


def test_error_rows_name_the_path_once(tmp_path, taxonomy):
    pytest.importorskip("pymupdf")
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 not really a pdf")
    result = _doctype(str(broken), "-t", taxonomy, "--keep", "financial", "--paths-only")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "warning:" in text and text.count("broken.pdf") == 1
