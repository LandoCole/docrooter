"""The README's runnable code is the code under ``examples/``; keep them honest."""

import json
import re
import runpy
from pathlib import Path

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
