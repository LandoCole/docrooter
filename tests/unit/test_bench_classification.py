"""``benchmarks.classification_bench``: the classification benchmark (spec G5, §6.1-6.2).

Everything here runs on fake ``classify`` results and tiny fixtures: no corpus, no held-out
file, no model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks import _guard
from benchmarks import classification_bench as bench
from benchmarks.classification.section_sample import title_sha1
from docrooter.backends.base import TokenUsage

ROOT = "/Users/me/Doc Rooter/.venv/bin/python"


# --- fakes -----------------------------------------------------------------------------------


def _doc(**titles: str) -> SimpleNamespace:
    """A Document stand-in: ``sections`` maps a section id to an object with a title."""
    return SimpleNamespace(
        sections={sid.replace("_", "."): SimpleNamespace(title=t) for sid, t in titles.items()}
    )


def _tag(section_id: str, label: str, title: str = "x") -> SimpleNamespace:
    return SimpleNamespace(id=section_id, title=title, level=1, label=label, confidence=0.9)


def _classification(
    doc_type: str | None,
    *,
    sections: list[SimpleNamespace] = (),  # type: ignore[assignment]
    decided_by: str = "model",
    flags: tuple[str, ...] = (),
    sections_usage: TokenUsage | None = None,
    document_usage: TokenUsage | None = None,
) -> SimpleNamespace:
    sections_usage = sections_usage or TokenUsage(100, 4, 2)
    document_usage = document_usage or TokenUsage(400, 2, 1)
    return SimpleNamespace(
        doc_type=doc_type,
        doc_type_confidence=0.8,
        decided_by=decided_by,
        candidates=["financial.annual_report"],
        flags=list(flags),
        sections=list(sections),
        tokens=sections_usage + document_usage,
        stage_tokens={"sections": sections_usage, "document": document_usage},
        levels={"section": "terms", "document": "terms"},
        backend="fake",
    )


# --- doc type --------------------------------------------------------------------------------


def test_doc_type_metrics():
    gold = {"a": "X", "b": "X", "c": "Y", "d": "Z"}
    pred = {"a": "X", "b": "Y", "c": "Y", "d": None}
    m = bench.doc_type_metrics(gold, pred)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.5)
    # Per gold type: X P=1 R=.5 F=2/3; Y P=.5 R=1 F=2/3; Z F=0 (macro over the gold types).
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    assert m["per_type"]["X"]["recall"] == pytest.approx(0.5)
    assert m["per_type"]["Y"]["precision"] == pytest.approx(0.5)
    assert set(m["per_type"]) == {"X", "Y", "Z"}  # a type only predicted adds no class
    assert bench.doc_type_metrics({}, {})["accuracy"] is None


def test_doc_type_gold_from_manifest_and_corrections(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "b", "split": "gold", "doc_type_hint": "legal.contract", "title": "T"},
                {"id": "a", "split": "gold", "doc_type_hint": "legal.court_opinion"},
                {"id": "c", "split": "train", "doc_type_hint": "legal.contract"},
            ]
        )
        + "\n"
    )
    gold = bench.doc_type_gold_from_manifest(manifest)
    assert gold["documents"] == {"a": "legal.court_opinion", "b": "legal.contract"}
    assert gold["corrections"] == []
    assert "T" not in json.dumps(gold)  # ids and types only
    path = tmp_path / "doc_type_gold.json"
    gold["corrections"] = [{"id": "a", "from": "legal.court_opinion", "to": "legal.contract"}]
    path.write_text(json.dumps(gold))
    assert bench.load_doc_type_gold(path) == {"a": "legal.contract", "b": "legal.contract"}
    gold["corrections"] = [{"id": "a", "from": "legal.legislation", "to": "legal.contract"}]
    path.write_text(json.dumps(gold))
    with pytest.raises(ValueError, match="correction"):
        bench.load_doc_type_gold(path)


# --- sections --------------------------------------------------------------------------------


def _gold_row(doc_id: str, section_id: str, title: str, label: str) -> dict:
    return {
        "doc_id": doc_id,
        "section_id": section_id,
        "title_sha1": title_sha1(title),
        "level": 1,
        "label": label,
    }


def test_section_f1_leaf_and_group():
    doc = _doc(s1="Balance sheet", s2="Income statement", s3="Abstract", s4="Renamed")
    gold = [
        _gold_row("d", "s1", "Balance sheet", "financial.balance_sheet"),
        _gold_row("d", "s2", "Income statement", "financial.income_statement"),
        _gold_row("d", "s3", "Abstract", "research.abstract"),
        _gold_row("d", "s4", "Original title", "general.summary"),  # reading drifted
        _gold_row("d", "s9", "Gone", "general.summary"),  # not in this reading
    ]
    tags = [
        _tag("s1", "financial.income_statement"),  # wrong leaf, right group
        _tag("s2", "financial.income_statement"),
        _tag("s4", "general.summary"),
        # s3 untagged (e.g. past max_sections): scored as no prediction
    ]
    records = bench.score_sections(doc, gold, tags, section_taxonomy="section_catalogue")
    assert [r["status"] for r in records] == ["ok", "ok", "ok", "title_mismatch", "absent"]
    assert records[2]["pred"] is None
    m = bench.section_metrics(records)
    assert m["n"] == 3 and m["excluded"] == 2 and m["missing"] == 1
    # Leaf: 1 correct of 2 predicted, 3 gold. P=.5 R=1/3 micro-F1=.4.
    assert m["leaf"]["micro_f1"] == pytest.approx(0.4)
    # Leaf macro over gold leaves: balance_sheet 0; income_statement P=.5 R=1 F=2/3; abstract 0.
    assert m["leaf"]["macro_f1"] == pytest.approx((0 + 2 / 3 + 0) / 3)
    # Group: financial 2/2 correct; research missed. P=1 R=2/3 micro-F1=.8.
    assert m["group"]["micro_f1"] == pytest.approx(0.8)
    # Group macro: financial F=1, research F=0.
    assert m["group"]["macro_f1"] == pytest.approx(0.5)
    assert bench.group_of("financial.balance_sheet") == "financial"


def test_doc_first_section_labels_map_to_catalogue():
    """doc_first labels come from the doc type's own section taxonomy: they are scored through
    the catalogue's ``from`` mapping, and a label with no catalogue leaf counts as unmapped."""
    doc = _doc(s1="Abstract", s2="Odd")
    gold = [
        _gold_row("d", "s1", "Abstract", "research.abstract"),
        _gold_row("d", "s2", "Odd", "general.summary"),
    ]
    tags = [_tag("s1", "abstract"), _tag("s2", "no_such_leaf")]
    records = bench.score_sections(doc, gold, tags, section_taxonomy="section_types/research_paper")
    assert records[0]["pred"] == "research.abstract" and records[0]["raw"] == "abstract"
    assert records[1]["pred"] is None and records[1]["unmapped"] is True
    m = bench.section_metrics(records)
    assert m["unmapped"] == 1 and m["leaf"]["micro_f1"] == pytest.approx(2 / 3)
    # Without a recorded section_taxonomy: the catalogue in sections_first, the predicted doc
    # type's own taxonomy in doc_first; a recorded one wins.
    plain = SimpleNamespace(section_taxonomy=None)
    assert bench.section_taxonomy_of(plain, flow="sections_first", doc_type=None) == (
        "section_catalogue"
    )
    assert (
        bench.section_taxonomy_of(plain, flow="doc_first", doc_type="research.research_paper")
        == "section_types/research_paper"
    )
    recorded = SimpleNamespace(section_taxonomy="section_types/generic")
    assert bench.section_taxonomy_of(recorded, flow="sections_first", doc_type=None) == (
        "section_types/generic"
    )


# --- tokens and the pick ---------------------------------------------------------------------


def _doc_record(doc_id: str, prompt: int, *, exact: bool = True) -> dict:
    return {"id": doc_id, "tokens": {"prompt": prompt, "completion": 1, "calls": 1, "exact": exact}}


def test_token_share_vs_full():
    full = {"documents": [_doc_record("a", 1000), _doc_record("b", 3000), _doc_record("c", 50)]}
    run = {"documents": [_doc_record("a", 100), _doc_record("b", 300), _doc_record("z", 7)]}
    # Over the documents in both runs: (100 + 300) / (1000 + 3000).
    assert bench.token_share(run, full) == pytest.approx(0.1)
    assert bench.token_share(full, full) == pytest.approx(1.0)
    assert bench.token_share(run, None) is None
    assert bench.token_share(run, {"documents": [_doc_record("a", 0)]}) is None


def test_baseline_is_the_same_backend_flow_full_full():
    def result(label, flow, sl, dl):
        return {
            "backend": {"label": label},
            "config": {"flow": flow, "section_level": sl, "doc_level": dl},
            "docs": "gold",
            "documents": [],
        }

    runs = [
        result("kw", "doc_first", "full", "full"),
        result("kw", "sections_first", "full", "full"),
        result("emb", "sections_first", "full", "full"),
        result("kw", "sections_first", "terms", "terms"),
    ]
    assert bench.baseline_for(runs[3], runs) is runs[1]
    assert bench.baseline_for(result("x", "doc_first", "terms", "terms"), runs) is None


def test_cheapest_within_two_points():
    rows = [
        {"level": "headings", "doc_accuracy": 0.70, "tokens": 300},
        {"level": "terms", "doc_accuracy": 0.885, "tokens": 450},
        {"level": "lead", "doc_accuracy": 0.95, "tokens": 700},
        {"level": "full", "doc_accuracy": 0.90, "tokens": 12000},
    ]
    pick = bench.cheapest_within(rows, "doc_accuracy")
    assert pick is not None and pick["level"] == "terms"  # 88.5 >= 90 - 2; headings is not
    exact = [dict(r) for r in rows]
    exact[1]["doc_accuracy"] = 0.88  # exactly 2 points below full still counts
    assert bench.cheapest_within(exact, "doc_accuracy")["level"] == "terms"
    worse = [dict(r) for r in rows]
    worse[1]["doc_accuracy"] = 0.879
    assert bench.cheapest_within(worse, "doc_accuracy")["level"] == "lead"
    assert bench.cheapest_within(rows[:3], "doc_accuracy") is None  # no full row
    no_metric = [dict(r, doc_accuracy=None) if r["level"] == "headings" else r for r in rows]
    assert bench.cheapest_within(no_metric, "doc_accuracy")["level"] == "terms"


# --- a run -----------------------------------------------------------------------------------

SECRET_TITLE = "Confidential merger plan of Acme"
SECRET_TEXT = "the body text of the document that must never be stored"


def _run(tmp_path, *, classify=None, docs_split="gold", extra=None):
    doc = _doc(s1=SECRET_TITLE, s2="Notes to the accounts")
    doc.text = SECRET_TEXT
    gold_rows = {
        "d1": [
            _gold_row("d1", "s1", SECRET_TITLE, "financial.balance_sheet"),
            _gold_row(
                "d1", "s2", "Notes to the accounts", "financial.notes_to_financial_statements"
            ),
        ]
    }

    def fake_classify(document, *, flow, section_level, doc_level, backend):
        assert document is doc and backend == "BACKEND"
        return _classification(
            "financial.annual_report",
            sections=[
                _tag("s1", "financial.balance_sheet", SECRET_TITLE),
                _tag("s2", "financial.notes_to_financial_statements", "Notes to the accounts"),
            ],
            flags=("truncated",) if doc_level == "full" else (),
            document_usage=TokenUsage(17000 if doc_level == "full" else 400, 2, 1),
        )

    config = bench.Config("sections_first", "full", "full")
    inputs = [bench.DocInput("d1", "financial.annual_report", lambda: doc)]
    return bench.run_configuration(
        inputs + ([extra] if extra else []),
        config,
        backend="BACKEND",
        spec="keyword",
        label="kw",
        docs=docs_split,
        section_gold=gold_rows if docs_split == "gold" else None,
        classify=classify or fake_classify,
        settings={"spec": "keyword"},
    )


def test_run_configuration_metrics(tmp_path):
    result = _run(tmp_path)
    m = result["metrics"]
    assert m["doc_type"]["accuracy"] == 1.0
    assert m["sections"]["leaf"]["micro_f1"] == 1.0
    assert m["prompt_tokens_mean"] == 17100
    assert m["tokens_exact"] is True
    assert m["truncated_share"] == 1.0
    assert m["over_cap_share"] == 1.0 and m["over_cap_n"] == 1  # 17,000 > 16,000, exact
    doc = result["documents"][0]
    assert doc["stage_tokens"]["document"]["prompt"] == 17000
    heldout = _run(tmp_path, docs_split="heldout")
    assert heldout["metrics"]["sections"] is None and "sections" not in heldout["documents"][0]


def test_results_hold_no_text(tmp_path):
    result = _run(tmp_path)
    dumped = json.dumps(result)
    assert SECRET_TITLE not in dumped and SECRET_TEXT not in dumped
    assert "Notes to the accounts" not in dumped
    bench.check_ids_and_numbers(result)  # passes
    with pytest.raises(ValueError, match="text"):
        bench.check_ids_and_numbers({"documents": [{"title": SECRET_TITLE}]})
    # A classify error is recorded by its class name only, never its message.

    other = _doc()

    def failing(document, **kwargs):
        if document is not other:
            raise RuntimeError(SECRET_TEXT)
        return _classification("legal.contract")

    extra = bench.DocInput("d2", "legal.contract", lambda: other)
    failed = _run(tmp_path, classify=failing, extra=extra)
    assert failed["documents"][0]["error"] == "RuntimeError"
    assert failed["metrics"]["errors"] == 1 and SECRET_TEXT not in json.dumps(failed)
    with pytest.raises(bench.RunAborted):  # every document failed: nothing to write
        _run(tmp_path, classify=failing)


def test_resume_skips_matching_settings(tmp_path, monkeypatch):
    calls = []

    def fake_classify(document, *, flow, section_level, doc_level, backend):
        calls.append((flow, section_level, doc_level))
        return _classification("legal.contract")

    docs = [bench.DocInput("d1", "legal.contract", lambda: _doc())]
    configs = [
        bench.Config("doc_first", "terms", "terms"),
        bench.Config("doc_first", "lead", "lead"),
    ]
    kwargs = dict(
        backend="B",
        spec="keyword",
        label="kw",
        docs="gold",
        section_gold={},
        classify=fake_classify,
        out=tmp_path,
        settings={"spec": "keyword"},
    )
    written = bench.run_grid(docs, configs, **kwargs)
    assert len(calls) == 2 and [p.name for p in written] == [
        "kw__gold__doc_first__terms__terms.json",
        "kw__gold__doc_first__lead__lead.json",
    ]
    bench.run_grid(docs, configs, **kwargs)
    assert len(calls) == 2  # both files exist with the same settings: skipped
    bench.run_grid(docs, configs, **{**kwargs, "settings": {"spec": "keyword", "x": 1}})
    assert len(calls) == 4  # different settings: re-run


def test_grid_configs():
    paired = bench.grid_configs(["doc_first", "sections_first"], levels=list(bench.LEVELS))
    assert len(paired) == 8 and all(c.section_level == c.doc_level for c in paired)
    cross = bench.grid_configs(
        ["sections_first"], section_levels=list(bench.LEVELS), doc_levels=list(bench.LEVELS)
    )
    assert len(cross) == 16
    assert bench.parse_list("all", bench.LEVELS) == list(bench.LEVELS)
    with pytest.raises(ValueError):
        bench.parse_list("headings,wide", bench.LEVELS)


# --- reading, guard, CLI and report ----------------------------------------------------------


def test_reading_must_match_the_gold(tmp_path):
    agreement = tmp_path / "agreement.json"
    reading = bench.current_reading()
    agreement.write_text(json.dumps({"sample": {"reading": reading}}))
    assert bench.check_reading(agreement)["profile"] == "fast"
    drifted = {**reading, "settings": {**reading["settings"], "layout": "auto"}}
    agreement.write_text(json.dumps({"sample": {"reading": drifted}}))
    with pytest.raises(ValueError, match="reading"):
        bench.check_reading(agreement)


def test_guard_matches_classification_bench_and_llm_server():
    table = [
        (10, 1, "python3.12", f"{ROOT} -m benchmarks.classification_bench run --backend x"),
        (11, 1, "Python", "/x/bin/python -m mlx_lm.server --model /snap --port 18080"),
        (12, 1, "python3.12", "/x/bin/python /x/bin/mlx_lm server --model /snap"),
        (13, 1, "zsh", "zsh -c python -m mlx_lm.server"),  # a wrapping shell
        (14, 1, "python3.12", f"{ROOT} -m pytest tests/unit/test_bench_classification.py"),
    ]
    found = _guard.other_model_processes(table, pid=99, ppid=98)
    assert [pid for pid, _ in found] == [10, 11, 12]


def test_llm_run_ignores_the_server_it_uses(monkeypatch, capsys):
    monkeypatch.setattr(_guard, "apply_env_caps", lambda: {})
    monkeypatch.setattr(
        _guard,
        "other_model_processes",
        lambda: [(11, "python -m mlx_lm.server --port 18080")],
    )
    _guard.single_model_process("classification_bench", ignore=_guard.LLM_SERVER)  # no exit
    with pytest.raises(SystemExit):
        _guard.single_model_process("classification_bench")


def test_jev_needs_explicit_flag(tmp_path):
    with pytest.raises(SystemExit):
        bench.main(["run", "--backend", "jev", "--flow", "doc_first", "--level", "terms"])


def test_report_pending_and_tables(tmp_path):
    pending = bench.render_report([])
    assert "Results pending" in pending
    assert "κ" in pending and "8 of the 19" in pending

    def result(label, flow, level, acc, prompt, exact=True):
        return {
            "backend": {"label": label, "spec": "keyword"},
            "config": {"flow": flow, "section_level": level, "doc_level": level},
            "docs": "gold",
            "settings": {"limit": None},
            "documents": [_doc_record("a", prompt, exact=exact)],
            "metrics": {
                "documents": 1,
                "errors": 0,
                "doc_type": {"accuracy": acc, "macro_f1": acc, "n": 1},
                "sections": {
                    "n": 1,
                    "excluded": 0,
                    "missing": 0,
                    "unmapped": 0,
                    "leaf": {"macro_f1": acc, "micro_f1": acc},
                    "group": {"macro_f1": acc, "micro_f1": acc},
                },
                "decided_by_rules": 0.5,
                "prompt_tokens_mean": prompt,
                "tokens_exact": exact,
                "truncated_share": 0.0,
                "over_cap_share": None,
                "over_cap_n": 0,
                "wall_s_mean": 0.1,
            },
        }

    runs = [
        result("kw", "sections_first", "headings", 0.5, 100, exact=False),
        result("kw", "sections_first", "terms", 0.89, 200, exact=False),
        result("kw", "sections_first", "lead", 0.9, 300, exact=False),
        result("kw", "sections_first", "full", 0.9, 1000, exact=False),
    ]
    text = bench.render_report(runs)
    assert "Results pending" not in text
    assert "| kw | sections_first | terms |" in text
    assert "estimated" in text
    pick_line = next(line for line in text.splitlines() if line.startswith("| kw | sections_first"))
    assert pick_line  # the main table row
    assert "terms" in text.split("## Cheapest level within 2 points of full")[1].split("##")[0]
    path = tmp_path / "results"
    path.mkdir()
    for k, r in enumerate(runs):
        (path / f"r{k}.json").write_text(json.dumps(r))
    out = tmp_path / "report.md"
    assert bench.main(["report", "--results", str(path), "--out", str(out)]) == 0
    assert out.read_text() == text
