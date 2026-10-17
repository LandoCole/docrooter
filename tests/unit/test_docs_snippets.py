"""The documentation site (``docs/site``) stays runnable and accurate.

- Every Python line on a docs page is a line of an example under ``examples/``, and the
  examples behind the docs run (with the keyword backend) and print what the pages show.
- Every ``docrooter`` command and option in a docs ``bash`` block exists.
- The API page covers ``docrooter.__all__``; the CLI reference generator finds every command.
- The build hook rewrites repository links for the site.
"""

from __future__ import annotations

import importlib.util
import re
import runpy
import shlex
from pathlib import Path

import pytest

import docrooter

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
SITE = ROOT / "docs" / "site"
PAGES = sorted(SITE.rglob("*.md"))
# Repository documents the site includes as pages (see docs/hooks.py).
INCLUDED = sorted((ROOT / "docs").glob("*.md"))


def _page(relative: str) -> str:
    return (SITE / relative).read_text(encoding="utf-8")


def _code(line: str) -> str:
    """A source line without its trailing comment (docs comments may differ from the examples')."""
    return line.split("  #")[0].strip()


def _blocks(text: str, lang: str) -> list[str]:
    return re.findall(rf"^```{lang}\n(.*?)^```", text, re.DOTALL | re.MULTILINE)


def _run(name: str, capsys, monkeypatch, *argv: str) -> list[str]:
    monkeypatch.setattr("sys.argv", [name, *argv])
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_the_site_has_pages():
    assert len(PAGES) >= 20  # hand-written; the build adds the included and generated pages


def test_every_docs_python_line_is_in_an_example():
    example_lines = {
        _code(line)
        for path in EXAMPLES.glob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    }
    snippets = 0
    for page in [*PAGES, *INCLUDED]:
        for snippet in _blocks(page.read_text(encoding="utf-8"), "python"):
            snippets += 1
            for line in snippet.splitlines():
                code = _code(line)
                if not code:
                    continue
                if code.startswith("from docrooter import "):
                    for name in code.removeprefix("from docrooter import ").split(","):
                        assert hasattr(docrooter, name.strip()), f"{page.name}: {code}"
                else:
                    where = page.relative_to(ROOT)
                    assert code in example_lines, f"{where}: line not in any example: {code!r}"
    assert snippets >= 25


def _cli_options() -> dict[tuple[str, ...], set[str]]:
    """Every command path of the CLI with the option strings it accepts."""
    import typer

    from docrooter.cli import app

    found: dict[tuple[str, ...], set[str]] = {}
    stack = [((), typer.main.get_command(app))]
    while stack:
        path, command = stack.pop()
        options = {"--help"}
        for param in command.params:
            options.update(getattr(param, "opts", []))
            options.update(getattr(param, "secondary_opts", []))
        found[path] = options
        for name, child in getattr(command, "commands", {}).items():
            stack.append(((*path, name), child))
    return found


def test_every_docs_cli_command_and_option_exists():
    commands = _cli_options()
    checked = 0
    for page in PAGES:
        for block in _blocks(page.read_text(encoding="utf-8"), "bash"):
            for line in block.splitlines():
                words = shlex.split(line.split(" #")[0], comments=True) if line.strip() else []
                while words and re.match(r"^[A-Z_]+=", words[0]):
                    words = words[1:]  # VAR=value prefixes
                if not words or words[0] != "docrooter":
                    continue
                path: tuple[str, ...] = ()
                rest = words[1:]
                while rest and (*path, rest[0]) in commands:
                    path, rest = (*path, rest[0]), rest[1:]
                for word in rest:
                    if word.startswith("-") and word != "-":
                        option = word.split("=", 1)[0]
                        assert option in commands[path], f"{page.name}: {line.strip()!r}"
                checked += 1
    assert checked >= 30


def test_api_page_documents_every_public_name():
    text = _page("reference/api.md")
    documented = set(re.findall(r"^::: docrooter\.(\w+)$", text, re.MULTILINE))
    missing = set(docrooter.__all__) - documented - {"__version__"}
    assert not missing, missing


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"docs_{name}", ROOT / "docs" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_reference_generator_finds_every_command():
    gen_cli = _load("gen_cli")
    paths = [" ".join(path) for path in gen_cli.commands()]
    assert paths[0] == ""
    for name in ("toc", "classify", "doctype", "normalize", "serve", "taxonomy validate"):
        assert name in paths
    text = gen_cli.help_text(["doctype"])
    assert text.startswith("Usage: docrooter doctype")
    assert "--paths-only" in text and "╭" not in text  # plain help, not rich boxes


def test_hook_rewrites_repository_links():
    pytest.importorskip("mkdocs")
    hooks = _load("hooks")
    hooks._repo_web = "https://example.org/repo"
    page = "benchmarks/pdf.md"
    source = hooks.INCLUDES[page]
    text = "[a](benchmarks-heldout.md#x) [b](../benchmarks/README.md) [c](https://x.y)\n"
    text += "```\n[not](a-link.md)\n```\n"
    out = hooks.rewrite_links(text, source, page)
    assert "[a](heldout.md#x)" in out
    assert "[b](https://example.org/repo/blob/main/benchmarks/README.md)" in out
    assert "[c](https://x.y)" in out
    assert "[not](a-link.md)" in out  # code blocks are left alone
    own = hooks.rewrite_links(
        "[e](../../../examples/toc.py)", "docs/site/guides/x.md", "guides/x.md"
    )
    assert own == "[e](https://example.org/repo/blob/main/examples/toc.py)"


# --- the examples behind the pages run and print what the pages show ---


def test_strategies_example(capsys, monkeypatch):
    lines = _run("strategies.py", capsys, monkeypatch)
    assert lines[:3] == [
        "Greedy: ['Financial > Tax', 'Financial > Audit']",
        "Beam: ['Financial > Tax', 'Legal > Contracts']",
        "Flat: ['Financial > Tax', 'Legal > Contracts']",
    ]
    assert "\n".join(lines[:3]) in _page("concepts/strategies.md")
    assert "\n".join(lines[3:7]) in _page("concepts/results.md")
    assert lines[-1] == "low_confidence"


def test_views_example(capsys, monkeypatch):
    lines = _run("views.py", capsys, monkeypatch)
    page = _page("concepts/views.md")
    assert "\n".join(lines[:4]) in page  # the first section view
    assert "\n".join(lines[8:12]) in page  # the document view
    assert lines[-2:] == ["Income taxes: Tax", "Risk factors: Risk"]


def test_events_example(capsys, monkeypatch):
    lines = _run("events.py", capsys, monkeypatch)
    stages = [line.split()[1:3] for line in lines]
    assert stages[:3] == [["assemble", "start"], ["assemble", "end"], ["done", "end"]]
    assert [s[0] for s in stages[3:]] == [
        "view",
        "view",
        "classify",
        "classify",
        "normalize",
        "normalize",
        "done",
    ]


def test_doctype_gate_example(capsys, monkeypatch):
    lines = _run("doctype_gate.py", capsys, monkeypatch)
    assert lines == [
        "skip  lease.md: ['Legal', 'Legislation']",
        "skip  paper.md: ['Research', 'Research paper']",
        "keep  results.md: ['Financial', 'Earnings release']",
    ]
    assert "\n".join(lines) in _page("guides/doctype-gating.md")


def test_custom_taxonomy_example(capsys, monkeypatch):
    lines = _run("custom_taxonomy.py", capsys, monkeypatch)
    assert "\n".join(lines[:5]) in _page("guides/custom-taxonomies.md")
    assert lines[-1].endswith("'Corporate', 'Internal']")
    assert lines[-1] in _page("guides/custom-taxonomies.md")


def test_normalize_example(capsys, monkeypatch):
    lines = _run("normalize.py", capsys, monkeypatch)
    assert "\n".join(lines[:6]) in _page("guides/section-normalisation.md")
    assert lines[6] == "['definitions', 'payment', 'term', 'termination']"


def test_toc_example(capsys, monkeypatch):
    lines = _run("toc.py", capsys, monkeypatch)
    assert lines[0] == "[('Introduction', 0), ('Background', 1), ('Method', 0)]"
    assert lines[2] == "1. Business [level 1, p. None, score 0.95]  ['level_repaired']"
    assert lines[-1] == "Our business faces risks."


def _tiny_pdf(tmp_path: Path) -> Path:
    pymupdf = pytest.importorskip("pymupdf")
    pdf = pymupdf.open()
    for title in ("Abstract", "Results"):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 90), title, fontsize=22, fontname="hebo")
        for k in range(4):
            page.insert_text((72, 140 + 15 * k), f"We report experiments {k}.", fontsize=11)
    path = tmp_path / "paper.pdf"
    pdf.save(str(path))
    pdf.close()
    return path


def test_pdf_quickstart_example(capsys, monkeypatch, tmp_path):
    path = _tiny_pdf(tmp_path)
    lines = _run("pdf_quickstart.py", capsys, monkeypatch, str(path))
    assert lines[0] == "pymupdf: 2 pages, 2 headings"
    assert lines[1:3] == ["Abstract  (p. 1)", "Results  (p. 2)"]
    assert lines[3].startswith("type: ")


def test_toc_example_on_a_pdf(capsys, monkeypatch, tmp_path):
    path = _tiny_pdf(tmp_path)
    pytest.importorskip("pdfplumber")
    lines = _run("toc.py", capsys, monkeypatch, str(path))
    assert lines[1].startswith("Abstract [level 1, p. 1")
    assert any("headings without bookmarks" in line for line in lines)
