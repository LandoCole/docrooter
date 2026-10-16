"""The README's runnable code is the code under ``examples/``; keep them honest."""

import json
import re
import runpy
from pathlib import Path

import pytest

from docrooter import Taxonomy
from docrooter.taxonomy.io import from_csv

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
README = ROOT / "README.md"
DOCS = ROOT / "docs"
GENERATED = {"docs/benchmarks.md"}  # written by `python -m benchmarks.report`, not committed yet


def _taxonomy(kind: str, body: str) -> Taxonomy:
    return Taxonomy.from_dict(json.loads(body)) if kind == "json" else from_csv(body)


def test_quickstart_example_runs(capsys):
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["Income tax: Financial > Tax", "Auditor's report: Financial > Audit"]


def test_readme_shows_the_quickstart_verbatim():
    source = (EXAMPLES / "quickstart.py").read_text(encoding="utf-8")
    body = source.split('"""\n', 1)[1].lstrip("\n")  # the example minus its module docstring
    assert body.rstrip() in README.read_text(encoding="utf-8")


def test_every_example_is_valid_python():
    for path in sorted(EXAMPLES.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_readme_relative_links_exist():
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!https?:|#|mailto:)([^)#\s]+)", text):
        assert target in GENERATED or (ROOT / target).exists(), target


def test_readme_taxonomy_examples_describe_one_tree():
    """The five README taxonomy blocks all load, and all have the same ids and parents."""
    text = README.read_text(encoding="utf-8")
    section = text.split("## Taxonomies", 1)[1].split("## Training", 1)[0]
    blocks = re.findall(r"```(json|csv)\n(.*?)```", section, re.DOTALL)
    trees = [{n.id: n.parent_id for n in _taxonomy(k, b).nodes.values()} for k, b in blocks]
    assert len(trees) == 5
    assert all(tree == trees[0] for tree in trees)


def test_taxonomy_format_docs_load():
    """Every tagged block in docs/taxonomy-formats.md loads; the five main ones agree."""
    text = (DOCS / "taxonomy-formats.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```(json|csv) (taxonomy|taxonomy-any)\n(.*?)```", text, re.DOTALL)
    hashes = [
        _taxonomy(kind, body).content_hash() for kind, tag, body in blocks if tag == "taxonomy"
    ]
    assert len(blocks) == 6
    assert len(hashes) == 5
    assert len(set(hashes)) == 1


def _code(line: str) -> str:
    """A source line without its trailing comment (README comments differ from the examples')."""
    return line.split("  #")[0].strip()


def test_every_readme_python_snippet_line_is_in_an_example():
    import docrooter

    example_lines = {
        _code(line)
        for path in EXAMPLES.glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    text = README.read_text(encoding="utf-8")
    snippets = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert len(snippets) >= 6
    for snippet in snippets:
        for line in snippet.splitlines():
            code = _code(line)
            if not code:
                continue
            if code.startswith("from docrooter import "):
                for name in code.removeprefix("from docrooter import ").split(","):
                    assert hasattr(docrooter, name.strip()), code
            else:
                assert code in example_lines, f"README line not in any example: {code!r}"


def test_custom_backend_example_runs(capsys):
    """The README's custom backend: no ``temperature`` and no ``multi_label`` argument."""
    runpy.run_path(str(EXAMPLES / "custom_backend.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == ["Tax", "Audit"]


def test_readme_documents_the_backend_protocol():
    section = README.read_text(encoding="utf-8").split("## Backends", 1)[1].split("\n## ", 1)[0]
    for member in ("name", "max_candidates", "temperature", "score(", "supports(", "tokenizer("):
        assert f"`{member}" in section, member
    assert "examples/custom_backend.py" in section


def test_doc_types_example_runs(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["doc_types.py"])
    runpy.run_path(str(EXAMPLES / "doc_types.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("doc_types, section_types/annual_report")
    assert lines[1:] == [
        "Research > Research paper 0.11",
        "Abstract: abstract",
        "Methods: method",
        "Results: results",
        "References: references",
    ]
    readme = README.read_text(encoding="utf-8")
    assert "\n".join(lines[1:]) in readme  # the README shows this output


def test_doc_types_example_loads_a_pdf_with_pdfplumber(capsys, monkeypatch, tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("pdfplumber")
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_text((72, 90), "Abstract", fontsize=22, fontname="hebo")
    for k in range(4):
        page.insert_text((72, 140 + 15 * k), f"We report experiments {k}.", fontsize=11)
    path = tmp_path / "paper.pdf"
    pdf.save(str(path))
    pdf.close()
    monkeypatch.setattr("sys.argv", ["doc_types.py", str(path)])
    runpy.run_path(str(EXAMPLES / "doc_types.py"), run_name="__main__")
    assert len(capsys.readouterr().out.splitlines()) >= 2
