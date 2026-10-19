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
GENERATED = {"docs/benchmarks.md"}  # written by `python -m benchmarks.report` (committed)


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
    assert lines[0].startswith(
        "doc_types, element_functions, element_topics, section_types/annual_report"
    )
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


def test_elements_and_chunks_example_runs(capsys, monkeypatch):
    """Without a path the example writes a two-page report (two columns, a ruled table with a
    caption) and reads it with the rooter, no OCR engine and no model; the README shows it."""
    pytest.importorskip("pymupdf")
    monkeypatch.setattr("sys.argv", ["elements_and_chunks.py"])
    runpy.run_path(str(EXAMPLES / "elements_and_chunks.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "balanced: 11 elements, 4 sections"
    assert lines[1] == "type: Financial > Annual report"
    kinds = [line.split()[2] for line in lines[2:13]]
    assert kinds == [
        *("title", "heading", "paragraph", "paragraph", "heading", "table"),
        *("caption", "page_number", "heading", "paragraph", "page_number"),
    ]
    # two columns are read left, then right; the page number sits at the end of its page
    assert lines[4].split(maxsplit=3)[3].startswith("Our sensors")
    assert lines[5].split(maxsplit=3)[3].startswith("Competition")
    assert lines[9].endswith("Page 1") and lines[10].split()[1] == "p2"
    chunk_lines = lines.index("c0: 4 tokens, Annual Report 2025 ['heading_only']")
    assert lines[chunk_lines : chunk_lines + 4] == [
        "c0: 4 tokens, Annual Report 2025 ['heading_only']",
        "c1: 94 tokens, 1. Business []",
        "c2: 65 tokens, 2. Financial results []",
        "c3: 25 tokens, 3. Outlook []",
    ]
    assert lines[chunk_lines + 4 :][:3] == [
        "2. Financial results",
        "",
        "Table: Table 1. Revenue by segment (millions); 4×3",
    ]
    readme = README.read_text(encoding="utf-8")
    assert "\n".join(lines[2:13]) in readme  # the README shows the elements
    assert "\n".join(lines[chunk_lines + 4 :]) in readme  # and the table's chunk


def test_readme_documents_profiles_with_the_measured_numbers():
    section = README.read_text(encoding="utf-8").split("## Elements and profiles", 1)[1]
    section = section.split("\n## ", 1)[0]
    for fact in ("0.257", "0.736", "0.927", "0.99 s", "4.64 s", "64.6 s", 'tables="none"'):
        assert fact in section, fact
    assert "by design" in section
    assert "examples/elements_and_chunks.py" in section


def test_train_element_tagger_example_runs_without_a_model(capsys):
    """The example labels a tiny fact-sheet corpus with a keyword teacher and embedder (no
    model, no network) and builds the confident set; the guide shows its output and lines."""
    runpy.run_path(str(EXAMPLES / "train_element_tagger.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "keyword:overlap@1: 64 topics, 32 functions"
    assert lines[1:6] == [
        "fund_document.investment_objective: 16 labelled, 12 kept, trained",
        "fund_document.fees_and_expenses: 16 labelled, 12 kept, trained",
        "fund_document.principal_risks: 12 labelled, 9 kept, trained",
        "fund_document.performance: 20 labelled, 12 kept, trained",
        "fund_document.other: 0 labelled, 0 kept, left untrained",
    ]
    assert lines[-1] == "training itself needs the models extra: docrooter train-elements"
    guide = (DOCS / "site" / "guides" / "train-your-tagger.md").read_text(encoding="utf-8")
    assert "\n".join(lines) in guide


def test_train_element_tagger_example_uses_the_teacher_protocol():
    """The stand-ins implement what the real engines do, so the example cannot drift from it."""
    from docrooter.teach import TeacherEngine

    namespace = runpy.run_path(str(EXAMPLES / "train_element_tagger.py"))
    assert isinstance(namespace["KeywordTeacher"](), TeacherEngine)


def test_element_tag_docs_state_the_committed_facts():
    """The corpus and gold-set facts on the pages are those of the committed files, the pages
    carry the licensing statements and the disclosures, and none gives a tagger result."""
    manifest = [
        json.loads(line)
        for line in (ROOT / "benchmarks/training_corpus/manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    splits = {s: sum(r["split"] == s for r in manifest) for s in ("train", "calib", "gold")}
    pages = sum(r["pages"] or 0 for r in manifest)
    agreement = json.loads(
        (ROOT / "benchmarks/training_corpus/gold_agreement.json").read_text(encoding="utf-8")
    )
    guide = (DOCS / "site" / "guides" / "train-your-tagger.md").read_text(encoding="utf-8")
    concept = (DOCS / "site" / "concepts" / "element-tags.md").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    facts = (
        f"{len(manifest):,} public native-PDF documents with {pages:,} pages",
        f"{splits['train']} in `train`, {splits['calib']} in `calib` and "
        f"{splits['gold']} in `gold`",
        f"{agreement['elements']} elements",
        f"{agreement['double_annotated']} of them twice",
        f"{agreement['topic']['kappa']:.3f}",
    )
    for fact in facts:
        assert fact in guide, fact
    assert f"{len(manifest):,} public native-PDF documents with {pages:,} pages" in readme
    assert (
        f"{agreement['topic']['kappa']:.3f}" in readme
        and f"{agreement['topic']['kappa']:.3f}" in concept
    )
    for text in (guide, readme):
        assert "Jev cannot be a teacher" in text
        assert "2.3(b)" in text
        assert "CC BY 4.0" in text
        assert "Claude-annotated" in text
        assert "SEC" in text
    assert "Apache-2.0" in guide and "bound by" in guide
    assert "consistency, not human-verified correctness" in guide
    assert "depends on corpus size" in guide


def test_current_docs_say_nineteen_document_types():
    """``financial.fact_sheet`` made it 19; only the benchmark corpora (labelled before) say 18."""
    from docrooter import builtin_taxonomy

    assert len(builtin_taxonomy("doc_types").leaves()) == 19
    pages = [README, *(DOCS / "site").rglob("*.md")]
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for stale in ("18 document types", "18 types in 6"):
            assert stale not in text, f"{page.relative_to(ROOT)}: {stale!r}"


def test_select_for_extraction_example_runs_without_a_model(capsys, monkeypatch):
    """With no arguments the example selects with keyword stand-ins and a hand-set selector (no
    model, no bundle); the guide shows its output."""
    monkeypatch.setattr("sys.argv", ["select_for_extraction.py"])
    runpy.run_path(str(EXAMPLES / "select_for_extraction.py"), run_name="__main__")
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0] == (
        "≈ 90% of relevant content kept · 41 tokens (43%) · "
        "guaranteed on calibration-like documents"
    )
    assert lines[1:3] == ["guarantee=True capped=False", "elements: h1, p2, p3, p4"]
    assert "Interest rate risk could raise our borrowing costs." in lines
    assert "We own one factory and lease three warehouses." not in out  # outside the selection
    assert lines[-1].startswith("≈ 100% of relevant content kept · 95 tokens (100%)")  # c = 1.0
    guide = (DOCS / "site" / "guides" / "select-for-extraction.md").read_text(encoding="utf-8")
    assert out.rstrip("\n") in guide


def test_select_for_extraction_example_stand_ins_follow_the_protocols():
    """The stand-in tagger tags the way the real one does (a dict of TagResult per element)."""
    from docrooter.elements.models import Element

    namespace = runpy.run_path(str(EXAMPLES / "select_for_extraction.py"))
    element = Element(id="e0", kind="paragraph", text="A risk.", boxes=[], order=0, section_id="s1")
    tags = namespace["KeywordTagger"]().tag([element], "annual_report")
    assert tags[0]["topic"].labels == ["annual_report.risk_factors"]
    assert namespace["hand_set_selector"]().dials["annual_report"].n == 100


def test_selection_docs_state_what_the_guarantee_is_and_is_not():
    """The pages say what the number means (an average, against teacher relevance, capped by
    the calibration size, correlated units) and when it does not apply, and give no result."""
    concept = (DOCS / "site" / "concepts" / "completeness-dial.md").read_text(encoding="utf-8")
    guide = (DOCS / "site" / "guides" / "select-for-extraction.md").read_text(encoding="utf-8")
    readme = README.read_text(encoding="utf-8")
    for fact in (
        "λ̂(c) = max { λ : (n/(n+1)) · mean L_d(λ) + 1/(n+1) ≤ α }",
        "teacher relevance, not human labels",
        "average over documents",
        "capped=True",
        "not exchangeable",
        "free-text query",
        "no tagger",
        "doc_type_mismatch",
        "budget",
        "estimate only",
        "`c = 1.0` is always the whole document",
        "average over documents and fields",
        "Claude-annotated",
        "Tags are not used",
        "Below `c = 1.0`",
        "uncalibrated_field=True",
        "`c > n/(n+1)`",
        "pending",
        "multi_field=True",
        "once per field",
        "fit_mismatch",
        'profile="accurate", element_tags="bundles/tagger-a"',
        "need text",
    ):
        assert fact in concept, fact
    for fact in (
        "docrooter selects",
        "validate_fields",
        "fit-selector",
        "teach label",
        "pending",
        "once per field",
        "fit_mismatch",
        "--profile accurate",
    ):
        assert fact in guide, fact
    for fact in ("teacher", "average", "capped=True", "pending", "fit_mismatch", "once per field"):
        assert fact in readme.split("## Select what to read", 1)[1].split("\n## ", 1)[0], fact
    # no field count drifts from the shipped sets
    from docrooter.select import builtin_fields

    counts = {f: len(builtin_fields(f)) for f in ("annual_report", "fund_document", "contract")}
    assert counts == {"annual_report": 15, "fund_document": 12, "contract": 15}
    assert "(15 fields), `fund_document` (12) and `contract` (15)" in concept
