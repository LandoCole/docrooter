"""Benchmark harness: dataset adapters, systems registry, runner and Markdown report.

No network and no torch: loaders get in-memory rows, and the runner uses the keyword backend.
"""

import json

import pytest

from benchmarks import report as bench_report
from benchmarks import run as bench_run
from benchmarks import systems
from benchmarks.data import (
    INSTRUCTIONS,
    ITEM_PART,
    BenchData,
    edgar_taxonomy,
    load_b1,
    load_b2,
    load_b3,
    pool_shots,
    split_camel,
    strip_item_heading,
)
from docrooter import Dataset
from docrooter.evaluate.report import Report
from docrooter.train.dataset import Example

WORDS = "revenue grew while costs fell as the company expanded its operations abroad".split()


def prose(n_words: int) -> str:
    return " ".join(WORDS[i % len(WORDS)] for i in range(n_words))


# --- B1 ---------------------------------------------------------------------------------


def test_b1_part_item_mapping():
    filing = {
        "filename": "1_2020.txt",
        "section_1": prose(10),  # under 50 words: dropped
        "section_7": "  " + prose(60).replace(" ", "\n  ", 3) + "  ",
        "section_7A": prose(80),
        "section_15": prose(55),
        "section_2": "",
    }
    data = load_b1(rows=[filing], n_test=10)
    # a single filing lands in the test half
    by_item = {e.section_id: e for e in data.test}
    assert set(by_item) == {"7", "7A", "15"}
    assert by_item["7"].path == ["part-ii", "part-ii.item-7"]
    assert by_item["7A"].path == ["part-ii", "part-ii.item-7a"]
    assert by_item["15"].path == ["part-iv", "part-iv.item-15"]
    assert "\n" not in by_item["7"].text and "  " not in by_item["7"].text
    assert len(by_item["7"].text.split()) == 60


def test_b1_taxonomy_matches_item_mapping():
    taxonomy = edgar_taxonomy()
    assert {n.id for n in taxonomy.leaves()} == {
        f"{part}.item-{item.lower()}" for item, part in ITEM_PART.items()
    }
    assert all(n.description for n in taxonomy.leaves())
    data = load_b1(rows=[{"section_9A": prose(70)}], n_test=1)
    assert data.test.resolve(data.taxonomy).examples[0].path == ["part-ii", "part-ii.item-9a"]


def test_b1_splits_filings_between_test_and_pool():
    filings = [
        {"filename": str(i), "section_5": prose(60), "section_1A": prose(60)} for i in range(6)
    ]
    data = load_b1(rows=filings, n_test=100, pool_per_leaf=2)
    test_docs = {e.doc_id for e in data.test}
    pool_docs = {e.doc_id for e in data.train_pool}
    assert test_docs and pool_docs and not test_docs & pool_docs
    assert len(data.test) == 6 and len(data.train_pool) == 4  # 3 test filings x 2 items each
    assert data.subset_ids is None and data.subset() is None


def test_b1_strips_the_leading_item_heading():
    body = prose(70)
    cases = {
        "7A": "ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK ",
        "1": "Item 1. Business. ",
        "5": "PART II Item 5 - Market for Registrant\u2019s Common Equity, Related Stockholder "
        "Matters and Issuer Purchases of Equity Securities ",
        "9A": "ITEM 9A. CONTROLS AND ",  # a heading cut short keeps the rest of the text
    }
    for item, heading in cases.items():
        assert strip_item_heading(heading + body, item) == body, item
    assert strip_item_heading(body, "7") == body  # no heading: unchanged
    filings = [{"section_7A": cases["7A"] + body}]
    assert load_b1(rows=filings, n_test=1).test.examples[0].text == body
    kept = load_b1(rows=filings, n_test=1, strip_headings=False).test.examples[0].text
    assert kept.startswith("ITEM 7A.")
    short = [{"section_7A": cases["7A"] + prose(20)}]  # under 50 words once the heading is gone
    assert len(load_b1(rows=short, n_test=1).test) == 0


def test_b1_deterministic_for_a_seed():
    filings = [{"filename": str(i), "section_5": prose(60)} for i in range(20)]
    a = load_b1(rows=filings, n_test=4, seed=1)
    b = load_b1(rows=filings, n_test=4, seed=1)
    assert [e.doc_id for e in a.test] == [e.doc_id for e in b.test]


# --- B2 ---------------------------------------------------------------------------------


def test_b2_camelcase_labels():
    assert split_camel("AmusementParkAttraction") == "Amusement Park Attraction"
    assert split_camel("TVShow") == "TV Show"
    assert split_camel("Agent") == "Agent"
    rows = [
        {"text": "A roller coaster park.", "l1": "Place", "l2": "AmusementPark", "l3": "ThemePark"},
        {"text": "A senator.", "l1": "Agent", "l2": "Politician", "l3": "Senator"},
    ]
    data = load_b2(rows=rows, n_test=10)
    labels = {n.label for n in data.taxonomy.nodes.values()}
    assert {"Amusement Park", "Theme Park", "Place", "Senator"} <= labels
    example = next(e for e in data.test if e.text == "A senator.")
    assert [data.taxonomy.node(i).label for i in example.path] == ["Agent", "Politician", "Senator"]
    assert data.taxonomy.depth() == 3


def test_b2_pool_keeps_first_k_per_leaf_and_extends_taxonomy():
    rows = [{"text": "t", "l1": "Place", "l2": "Station", "l3": "RailwayStation"}]
    train = [
        {"text": f"s{i}", "l1": "Agent", "l2": "Politician", "l3": "Senator"} for i in range(5)
    ]
    data = load_b2(rows=rows, train_rows=train, n_test=10, pool_per_leaf=2)
    assert [e.text for e in data.train_pool] == ["s0", "s1"]
    assert "agent.politician.senator" in data.taxonomy
    assert data.train_pool.resolve(data.taxonomy)  # pool paths exist in the taxonomy


# --- B3 ---------------------------------------------------------------------------------


def test_b3_flat_taxonomy_and_fixed_subset():
    names = ["Governing Laws", "Notices", "Anti-Corruption Laws"]
    rows = [{"text": f"clause {i}", "label": i % 3} for i in range(40)]
    data = load_b3(rows=rows, label_names=names, n_test=30, subset_size=10)
    assert data.taxonomy.depth() == 1 and len(data.taxonomy) == 3
    assert data.subset_ids == list(range(10)) and len(data.subset()) == 10
    again = load_b3(rows=rows, label_names=names, n_test=30, subset_size=10)
    assert [e.text for e in data.subset()] == [e.text for e in again.subset()]
    example = data.test.examples[0]
    assert data.taxonomy.node(example.path[0]).label in names


def test_b3_subset_is_capped_by_test_size():
    rows = [{"text": "x", "label": 0}] * 5
    assert load_b3(rows=rows, label_names=["A"], n_test=3).subset_ids == [0, 1, 2]


def test_instructions_cover_every_benchmark():
    assert INSTRUCTIONS["b1"] == "Classify this section of a 10-K annual report"
    assert INSTRUCTIONS["b2"] == "Classify the topic of this Wikipedia article"
    assert INSTRUCTIONS["b3"] == "Classify the type of this contract clause"


# --- pool_shots -------------------------------------------------------------------------


def _pool() -> Dataset:
    return Dataset(
        [Example(text=f"a{i}", path=["p", "a"]) for i in range(5)]
        + [Example(text=f"b{i}", path=["p", "b"]) for i in range(2)]
        + [Example(text="c0", path=["q", "c"])]
    )


def test_pool_shots_k_per_leaf():
    shots = pool_shots(_pool(), 2, seed=0)
    counts: dict[str, int] = {}
    for e in shots:
        counts[e.path[-1]] = counts.get(e.path[-1], 0) + 1
    assert counts == {"a": 2, "b": 2, "c": 1}  # a leaf with fewer than k gives all it has
    assert [e.text for e in shots] == [e.text for e in pool_shots(_pool(), 2, seed=0)]
    assert {e.text for e in pool_shots(_pool(), 2, seed=0)} != {
        e.text for e in pool_shots(_pool(), 2, seed=3)
    }
    assert len(pool_shots(_pool(), 0, seed=0)) == 0
    with pytest.raises(ValueError):
        pool_shots(_pool(), -1, seed=0)


# --- systems ----------------------------------------------------------------------------


def test_registry_names_and_default_systems():
    names = [s.name for s in systems.available_systems(env={})]
    assert names == ["ultrabert-embed", "bge-small", "deberta-nli", "ultrabert-mlm"]


def test_optional_systems_need_their_environment(monkeypatch):
    env = {"DOCROOTER_BENCH_LLM_URL": "http://x", "DOCROOTER_BENCH_LLM_MODEL": "m"}
    with pytest.raises(systems.SystemUnavailable, match="DOCROOTER_BENCH_LLM_URL"):
        systems.REGISTRY["openai"]("openai", {})
    with pytest.raises(systems.SystemUnavailable, match="TYPESAFE_API_KEY"):
        systems.REGISTRY["jev"]("jev", {})
    # with the env set but the backend module absent they are still skipped, not fatal
    monkeypatch.setattr(systems.importlib.util, "find_spec", lambda name: None)
    names = [s.name for s in systems.available_systems(env={**env, "TYPESAFE_API_KEY": "k"})]
    assert "openai" not in names and "jev" not in names
    with pytest.raises(systems.SystemUnavailable, match="not available"):
        systems.available_systems(["jev"], env={"TYPESAFE_API_KEY": "k"})
    monkeypatch.setattr(systems.importlib.util, "find_spec", lambda name: object())
    built = systems.available_systems(env={**env, "TYPESAFE_API_KEY": "k"})
    specs = {s.name: s.spec for s in built}
    assert specs["openai"] == "openai:m" and specs["jev"] == "jev:jev-1.13.0"


def test_only_selects_systems_and_rejects_unknown():
    assert [s.name for s in systems.available_systems(["bge-small"], env={})] == ["bge-small"]
    with pytest.raises(KeyError, match="nope"):
        systems.available_systems(["nope"], env={})


def test_dataset_runs_per_system():
    by_name = {s.name: s for s in systems.available_systems(env={})}
    keys = ["b1", "b2", "b3"]
    assert systems.dataset_runs(by_name["bge-small"], keys) == ["b1", "b2", "b3", "b3-subset"]
    assert systems.dataset_runs(by_name["deberta-nli"], keys) == ["b1", "b2", "b3-subset"]
    assert systems.dataset_runs(by_name["ultrabert-mlm"], keys) == ["b1", "b2"]
    assert systems.dataset_runs(by_name["bge-small"], ["b2"]) == ["b2"]


def test_zero_shot_specs_are_the_briefed_ones():
    specs = {s.name: s.spec for s in systems.available_systems(env={})}
    assert specs == {
        "ultrabert-embed": "embedding:dmis-lab/UltraBERT-Embed-150M",
        "bge-small": "embedding:BAAI/bge-small-en-v1.5",
        "deberta-nli": "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        "ultrabert-mlm": "mlm-choice:dmis-lab/UltraBERT-380M",
    }


def test_build_classifier_uses_benchmark_settings():
    data = load_b3(rows=[{"text": "x", "label": 0}], label_names=["A"], n_test=1)
    clf = systems.build_classifier(systems.System("kw", "keyword"), data)
    assert clf.strategy == systems.Beam(3)
    assert clf.settings.threshold == 0.0
    assert clf.settings.instruction == INSTRUCTIONS["b3"]
    assert clf.view.fields == ("text",) and clf.view.max_tokens == 512


# --- runner -----------------------------------------------------------------------------


def _tiny_loaders():
    def b1(n_test, seed):
        return load_b1(
            rows=[
                {"filename": str(i), "section_1": prose(60), "section_7": prose(60) + " margin"}
                for i in range(6)
            ],
            n_test=n_test,
            seed=seed,
        )

    def b3(n_test, seed):
        rows = [
            {"text": f"{'tax' if i % 2 else 'notice'} clause {i}", "label": i % 2}
            for i in range(20)
        ]
        return load_b3(
            rows=rows, label_names=["Notices", "Taxes"], n_test=n_test, seed=seed, subset_size=4
        )

    return {"b1": b1, "b3": b3}


def test_run_suite_writes_rows_incrementally(tmp_path, monkeypatch):
    monkeypatch.setattr(
        systems, "REGISTRY", {"kw": systems._fixed("keyword", datasets=("b1", "b3"))}
    )
    seen: list[int] = []
    original = bench_run.ResultStore.add

    def spy(self, row):
        original(self, row)
        seen.append(len(json.loads(self.path.read_text())))

    monkeypatch.setattr(bench_run.ResultStore, "add", spy)
    rows = bench_run.run_suite(
        "zero-shot",
        None,
        6,
        out_dir=tmp_path,
        datasets=["b1", "b3"],
        loaders=_tiny_loaders(),
        environ={},
    )
    assert [r["dataset"] for r in rows] == ["b1", "b3", "b3-subset"]
    assert seen == [1, 2, 3]  # the file grows after every system x dataset
    row = rows[0]
    assert row["suite"] == "zero-shot" and row["system"] == "kw" and row["strategy"] == "Beam(3)"
    assert row["shots"] == 0 and row["report"]["n"] == row["n"]
    for key in ("macro_p", "macro_r", "macro_f1", "sections_per_sec", "seconds_total"):
        assert row[key] is not None
    assert row["seconds_per_example"] == pytest.approx(1 / row["sections_per_sec"])
    assert row["seconds_total"] == pytest.approx(row["n"] / row["sections_per_sec"])
    assert set(row["env"]) >= {
        "device",
        "torch",
        "transformers",
        "sentence_transformers",
        "docrooter",
    }
    assert json.loads((tmp_path / "zero-shot.json").read_text()) == rows


def test_rerun_replaces_the_same_row_and_keeps_others(tmp_path):
    def row(system, hf1):
        return {
            "suite": "zero-shot", "dataset": "b1", "system": system, "strategy": "Beam(3)",
            "shots": 0, "hf1": hf1,
        }  # fmt: skip

    store = bench_run.ResultStore(tmp_path, "zero-shot")
    store.add(row("a", 0.1))
    store.add(row("b", 0.2))
    store = bench_run.ResultStore(tmp_path, "zero-shot")  # a new process
    store.add(row("a", 0.9))
    saved = json.loads((tmp_path / "zero-shot.json").read_text())
    assert {r["system"]: r["hf1"] for r in saved} == {"a": 0.9, "b": 0.2}


def test_failed_cell_is_reported_but_finished_rows_are_kept(tmp_path, monkeypatch):
    registry = {
        "kw": systems._fixed("keyword", datasets=("b3",)),
        "bad": systems._fixed("nonexistent-kind", datasets=("b3",)),
    }
    monkeypatch.setattr(systems, "REGISTRY", registry)
    with pytest.raises(RuntimeError, match="bad/b3"):
        bench_run.run_suite(
            "zero-shot", None, 6, out_dir=tmp_path, datasets=["b3"],
            loaders=_tiny_loaders(), environ={},
        )  # fmt: skip
    saved = json.loads((tmp_path / "zero-shot.json").read_text())
    assert [(r["system"], r["dataset"]) for r in saved] == [("kw", "b3"), ("kw", "b3-subset")]


def test_unimplemented_suites_and_bad_arguments(tmp_path):
    with pytest.raises(NotImplementedError, match="Task 15"):
        bench_run.run_suite("few-shot", None, 5, out_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        bench_run.run_suite("zero-shot", None, 5, out_dir=tmp_path, datasets=["b9"])
    with pytest.raises(ValueError, match="unknown suite"):
        bench_run.run_suite("nope", None, 5, out_dir=tmp_path)  # type: ignore[arg-type]


def test_make_row_speed_fields():
    rep = Report(
        n=50, exact=0.5, hp=0.6, hr=0.6, hf1=0.6, macro_p=0.3, macro_r=0.4, macro_f1=0.35,
        stop_rate=0.0, selective_acc=0.5, sections_per_sec=25.0, device="cpu",
    )  # fmt: skip
    row = bench_run.make_row(
        suite="zero-shot", dataset="b1", system="s", strategy="Beam(3)", shots=0, report=rep,
        env={}, wall_seconds=9.0,
    )  # fmt: skip
    assert row["seconds_total"] == 2.0 and row["seconds_per_example"] == 0.04
    assert row["macro_f1"] == 0.35 and row["report"]["hf1"] == 0.6
    json.dumps(row)


def test_cli_arguments():
    args = bench_run._parse(
        ["--suite", "zero-shot", "--device", "mps", "--only", "a", "--only", "b", "--n-test", "7"]
    )
    assert args.only == ["a", "b"] and args.n_test == 7 and args.device == "mps"
    assert bench_run._parse([]).n_test == 1000


# --- markdown report --------------------------------------------------------------------


def _fixture_row(system, dataset, hf1, sps, device="mps", shots=0):
    report = {
        "n": 100, "exact": hf1 - 0.1, "hp": hf1, "hr": hf1, "hf1": hf1, "macro_p": 0.2,
        "macro_r": 0.3, "macro_f1": 0.25, "stop_rate": 0.0, "selective_acc": 0.5,
        "sections_per_sec": sps, "device": device,
    }  # fmt: skip
    return {
        "suite": "zero-shot", "dataset": dataset, "system": system, "strategy": "Beam(3)",
        "shots": shots, "n": 100, "macro_p": 0.2, "macro_r": 0.3, "macro_f1": 0.25,
        "sections_per_sec": sps, "seconds_total": 100 / sps, "seconds_per_example": 1 / sps,
        "report": report,
        "env": {"torch": "2.9.0", "transformers": "5.0.0", "docrooter": "0.1.0.dev0",
                "revisions": {"embedding:m": "367d5343b5a0164d57b810cedf5de4b8983d5984"}},
    }  # fmt: skip


def test_render_markdown_table(tmp_path):
    rows = [
        _fixture_row("alpha", "b1", 0.80, 50.0),
        _fixture_row("alpha", "b2", 0.70, 40.0),
        _fixture_row("alpha", "b3-subset", 0.60, 45.0),
        _fixture_row("beta", "b1", 0.90, 5.0),
        _fixture_row("beta", "b2", 0.50, 4.0),
        _fixture_row("beta", "b3-subset", 0.40, 6.0),
        _fixture_row("gamma|pipe", "b1", 0.10, 70.0, device="cpu"),
    ]
    (tmp_path / "zero-shot.json").write_text(json.dumps(rows))
    (tmp_path / "decisions.md").write_text("## Decisions\n\nPick alpha.\n")
    md = bench_report.render_markdown(tmp_path, link_base=tmp_path)
    assert md.startswith("# Benchmarks")
    assert "## zero-shot" in md and "[zero-shot.json](zero-shot.json)" in md
    assert "### B1: EDGAR-CORPUS" in md and "### B2: DBPedia" in md and "### B3 subset" in md
    # summary: alpha mean (0.8+0.7+0.6)/3 = 0.700, beta mean 0.600, incomplete systems have none
    summary = md.split("### Summary")[1].split("###")[0]
    lines = [ln for ln in summary.splitlines() if ln.startswith("| ") and "---" not in ln]
    assert lines[1].startswith("| alpha | Beam(3) | 0 | 0.800 | 0.700 | 0.600 | 0.700 | 40.0 |")
    assert lines[2].startswith("| beta |") and "| 0.600 | 4.0 |" in lines[2]
    assert lines[3].startswith("| gamma\\|pipe |") and "n/a" in lines[3]
    # per-dataset tables sort by hF1 and carry speed and device
    b1 = md.split("### B1")[1].split("###")[0]
    assert b1.index("| beta |") < b1.index("| alpha |") < b1.index("gamma")
    assert "| 50.0 | 0.0200 | mps |" in b1
    assert "revisions: embedding:m @ 367d5343" in md and "torch 2.9.0" in md
    assert md.rstrip().endswith("Pick alpha.")


def test_render_markdown_handles_empty_dir_and_missing_suites(tmp_path):
    md = bench_report.render_markdown(tmp_path, link_base=tmp_path)
    assert md.startswith("# Benchmarks") and "##" not in md.replace("# Benchmarks", "")


def test_benchdata_is_a_namedtuple_with_the_briefed_fields():
    assert BenchData._fields == ("name", "taxonomy", "test", "train_pool", "subset_ids")
