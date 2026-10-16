"""The few-shot and strategies suites of the benchmark harness, with a keyword backend standing
in for trained models (no torch, no network)."""

import json

import pytest

from benchmarks import report as bench_report
from benchmarks import run as bench_run
from benchmarks import systems
from benchmarks.data import load_b2, load_b3, pool_shots
from docrooter.backends import KeywordBackend

TOPICS = {
    ("Agent", "Politician", "Senator"): "senator election vote",
    ("Agent", "Athlete", "Boxer"): "boxer fight match",
    ("Place", "Station", "RailwayStation"): "railway train platform",
    ("Place", "Park", "ThemePark"): "roller coaster ride",
}


def _rows(n):
    return [
        {"text": f"{words} number {i}", "l1": l1, "l2": l2, "l3": l3}
        for i in range(n)
        for (l1, l2, l3), words in TOPICS.items()
    ]


def _loaders():
    def b2(n_test, seed):
        return load_b2(rows=_rows(3), train_rows=_rows(6), n_test=n_test, seed=seed)

    def b3(n_test, seed):
        rows = [
            {"text": f"{'tax' if i % 2 else 'notice'} clause {i}", "label": i % 2}
            for i in range(20)
        ]
        return load_b3(
            rows=rows, label_names=["Notices", "Taxes"], n_test=n_test, seed=seed, subset_size=4
        )

    return {"b2": b2, "b3": b3}


@pytest.fixture
def fake_training(monkeypatch):
    """Base ``kw`` whose 'trained model' is a keyword backend; records each training call."""
    calls = []

    def trained_model(
        data, base, k, variant, *, device=None, root=None, batch_size=None, max_triplets=None
    ):
        calls.append((data.name, base, k, variant))
        shots = pool_shots(data.train_pool, k, seed=0)
        return KeywordBackend(name=f"trained:{base}"), 1.5 + k, len(shots)

    monkeypatch.setattr(systems, "BASE_MODELS", {"kw": "keyword-model", "kw2": "keyword-model"})
    monkeypatch.setattr(bench_run, "BASE_MODELS", systems.BASE_MODELS)
    monkeypatch.setattr(systems, "trained_model", trained_model)
    monkeypatch.setattr(systems, "REGISTRY", {"kw": systems._fixed("keyword")})
    return calls


def test_few_shot_rows_per_base_shot_and_variant(tmp_path, fake_training):
    rows = bench_run.run_suite(
        "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
        only=["kw"], shots=[2, 4], bases=["kw", "kw2"],
    )  # fmt: skip
    assert [(r["system"], r["shots"]) for r in rows] == [
        ("kw/heads", 2), ("kw/finetuned", 2), ("kw/heads", 4), ("kw/finetuned", 4),
    ]  # fmt: skip
    assert fake_training == [("b2", "kw", k, v) for k in (2, 4) for v in ("heads", "finetuned")]
    row = rows[0]
    assert row["suite"] == "few-shot" and row["dataset"] == "b2" and row["strategy"] == "Beam(3)"
    assert row["train_seconds"] == 3.5 and row["n_train"] == 8  # 4 leaves x 2 shots
    assert json.loads((tmp_path / "few-shot.json").read_text()) == rows


def test_few_shot_b3_runs_the_full_test_and_the_subset(tmp_path, fake_training):
    rows = bench_run.run_suite(
        "few-shot", None, 8, out_dir=tmp_path, datasets=["b3"], loaders=_loaders(),
        shots=[2], bases=["kw"],
    )  # fmt: skip
    assert [(r["system"], r["dataset"]) for r in rows] == [
        ("kw/heads", "b3"), ("kw/heads", "b3-subset"),
        ("kw/finetuned", "b3"), ("kw/finetuned", "b3-subset"),
    ]  # fmt: skip
    assert len(fake_training) == 2  # one training per variant, not per evaluation set


def test_few_shot_writes_each_cell_and_reports_failures(tmp_path, fake_training, monkeypatch):
    good = systems.trained_model

    def flaky(data, base, k, variant, **kw):
        if k == 4:
            raise RuntimeError("out of memory")
        return good(data, base, k, variant, **kw)

    monkeypatch.setattr(systems, "trained_model", flaky)
    with pytest.raises(RuntimeError, match="kw/heads/4/b2"):
        bench_run.run_suite(
            "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
            shots=[2, 4], bases=["kw"],
        )  # fmt: skip
    saved = json.loads((tmp_path / "few-shot.json").read_text())
    assert [(r["system"], r["shots"]) for r in saved] == [("kw/heads", 2), ("kw/finetuned", 2)]


def test_few_shot_only_must_name_a_known_base(tmp_path, fake_training):
    with pytest.raises(KeyError, match="nope"):
        bench_run.run_suite(
            "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
            only=["nope"], shots=[2], bases=["kw"],
        )  # fmt: skip
    with pytest.raises(ValueError, match="selects none"):
        bench_run.run_suite(
            "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
            only=["kw2"], shots=[2], bases=["kw"],
        )  # fmt: skip


def test_strategies_suite_rows(tmp_path, fake_training, monkeypatch):
    monkeypatch.setattr(systems, "REGISTRY", {"kw": systems._fixed("embedding:fake")})
    from benchmarks.systems import build_classifier as real_build

    monkeypatch.setattr(
        bench_run,
        "build_classifier",
        lambda system, data, **kw: real_build(systems.System("kw", "keyword"), data, **kw),
    )
    rows = bench_run.run_suite(
        "strategies", None, 8, out_dir=tmp_path, loaders=_loaders(), environ={},
        trained_shots=4, zero_shot_system="kw", trained_system="kw/finetuned",
    )  # fmt: skip
    assert [(r["system"], r["strategy"], r["shots"]) for r in rows] == [
        ("kw", "Greedy", 0), ("kw", "Beam(3)", 0), ("kw", "Flat", 0),
        ("kw/finetuned", "Greedy", 4), ("kw/finetuned", "Beam(3)", 4),
    ]  # fmt: skip
    assert {r["dataset"] for r in rows} == {"b2"} and rows[-1]["train_seconds"] == 5.5
    assert fake_training == [("b2", "kw", 4, "finetuned")]


def test_strategies_flat_only_for_embedding_backends(tmp_path, fake_training):
    rows = bench_run.run_suite(
        "strategies", None, 8, out_dir=tmp_path, loaders=_loaders(), environ={},
        only=["kw"], zero_shot_system="kw", trained_system="kw/heads",
    )  # fmt: skip
    # the keyword system is not an embedding backend; --only kw also selects kw/heads
    assert [(r["system"], r["strategy"]) for r in rows] == [
        ("kw", "Greedy"), ("kw", "Beam(3)"), ("kw/heads", "Greedy"), ("kw/heads", "Beam(3)"),
    ]  # fmt: skip


def test_strategies_only_filters_systems(tmp_path, fake_training):
    rows = bench_run.run_suite(
        "strategies", None, 8, out_dir=tmp_path, loaders=_loaders(), environ={},
        only=["kw/heads"], zero_shot_system="kw", trained_system="kw/heads",
    )  # fmt: skip
    assert {r["system"] for r in rows} == {"kw/heads"}


def test_strategies_picks_the_best_systems_from_earlier_results(tmp_path):
    def row(suite, system, shots, hf1, strategy="Beam(3)", dataset="b2"):
        return {
            "suite": suite, "dataset": dataset, "system": system, "strategy": strategy,
            "shots": shots, "report": {"hf1": hf1},
        }  # fmt: skip

    (tmp_path / "zero-shot.json").write_text(
        json.dumps(
            [
                row("zero-shot", "bge-small", 0, 0.5),
                row("zero-shot", "ultrabert-embed", 0, 0.7),
                row("zero-shot", "deberta-nli", 0, 0.9, dataset="b1"),
            ]
        )
    )
    (tmp_path / "few-shot.json").write_text(
        json.dumps(
            [
                row("few-shot", "bge-small/heads", 16, 0.6),
                row("few-shot", "bge-small/finetuned", 16, 0.8),
                row("few-shot", "ultrabert-embed/heads", 4, 0.99),
            ]
        )
    )
    assert bench_run.pick_strategy_systems(tmp_path, 16, None, None) == (
        "ultrabert-embed",
        "bge-small/finetuned",
    )
    assert bench_run.pick_strategy_systems(tmp_path, 16, "bge-small", "bge-small/heads") == (
        "bge-small",
        "bge-small/heads",
    )
    with pytest.raises(ValueError, match="trained system"):
        bench_run.pick_strategy_systems(tmp_path, 16, None, "nonsense")
    empty = tmp_path / "empty"
    assert bench_run.pick_strategy_systems(empty, 16, None, None) == (
        bench_run.DEFAULT_STRATEGY_ZERO_SHOT,
        bench_run.DEFAULT_STRATEGY_TRAINED,
    )


def test_trained_bundles_are_cached_by_dataset_base_shots_and_variant(tmp_path):
    path = systems.bundle_dir("b2", "bge-small", 16, "heads", tmp_path)
    assert path == tmp_path / "b2" / "bge-small" / "16" / "heads"
    assert systems.bundle_dir("b1", "ultrabert-embed", 4, "finetuned").parts[-6:] == (
        ".cache", "bundles", "b1", "ultrabert-embed", "4", "finetuned",
    )  # fmt: skip


def test_cli_arguments_for_the_new_suites():
    args = bench_run._parse(
        ["--suite", "few-shot", "--shots", "4,8", "--bases", "bge-small", "--trained-shots", "8",
         "--zero-shot-system", "bge-small", "--trained-system", "bge-small/heads"]
    )  # fmt: skip
    assert args.shots == "4,8" and args.bases == "bge-small" and args.trained_shots == 8
    assert args.zero_shot_system == "bge-small" and args.trained_system == "bge-small/heads"
    defaults = bench_run._parse([])
    assert defaults.shots == "4,8,16,64" and defaults.bases == "ultrabert-embed,bge-small"
    assert defaults.trained_shots == 16


def test_report_shows_train_seconds_for_trained_rows(tmp_path, fake_training):
    bench_run.run_suite(
        "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
        shots=[2], bases=["kw"],
    )  # fmt: skip
    md = bench_report.render_markdown(tmp_path, link_base=tmp_path)
    assert "## few-shot" in md and "| Train s |" in md and "| kw/heads |" in md


def test_train_batch_size_reaches_training_and_the_rows(tmp_path, monkeypatch):
    seen = []

    def trained_model(
        data, base, k, variant, *, device=None, root=None, batch_size=None, max_triplets=None
    ):
        seen.append(batch_size)
        return KeywordBackend(name="trained"), 1.0, 4

    monkeypatch.setattr(systems, "BASE_MODELS", {"kw": "keyword-model"})
    monkeypatch.setattr(bench_run, "BASE_MODELS", systems.BASE_MODELS)
    monkeypatch.setattr(systems, "trained_model", trained_model)
    rows = bench_run.run_suite(
        "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
        shots=[2], bases=["kw"], train_batch_size=8,
    )  # fmt: skip
    assert seen == [8, 8] and rows[0]["env"]["train_batch_size"] == 8
    assert bench_run._parse(["--train-batch-size", "8"]).train_batch_size == 8
    assert bench_run._parse(["--train-max-triplets", "99"]).train_max_triplets == 99
    assert bench_run._parse([]).train_batch_size is None


def test_each_trained_classifier_loads_its_own_copy_of_the_model(tmp_path):
    pytest.importorskip("joblib")
    from docrooter.backends.recipes import get_recipe
    from docrooter.train.bundle import TrainedModel

    data = load_b3(rows=[{"text": "t", "label": 0}], label_names=["A"], n_test=1)
    taxonomy = data.taxonomy
    recipe = get_recipe("BAAI/bge-small-en-v1.5")

    class Body:
        name = "embedding:fake"

        def _load(self):
            class _Model:
                def save(self, path):
                    (tmp_path / "bundle" / "body").mkdir(parents=True, exist_ok=True)

            return _Model()

    manifest = {
        "base_model": "BAAI/bge-small-en-v1.5", "taxonomy_hash": taxonomy.content_hash(),
        "instruction": "x", "recipe": recipe.model_dump(), "heads": {},
    }  # fmt: skip
    model = TrainedModel(manifest, {}, Body())  # type: ignore[arg-type]
    model.save(tmp_path / "bundle")
    one = systems.build_trained_classifier(model, data)
    two = systems.build_trained_classifier(model, data)
    assert one.backend is not two.backend and one.backend is not model
    assert one.backend.path == two.backend.path == (tmp_path / "bundle").resolve()


def test_few_shot_only_can_pick_one_variant(tmp_path, fake_training):
    rows = bench_run.run_suite(
        "few-shot", None, 8, out_dir=tmp_path, datasets=["b2"], loaders=_loaders(),
        only=["kw/heads"], shots=[2], bases=["kw", "kw2"],
    )  # fmt: skip
    assert [r["system"] for r in rows] == ["kw/heads"]
    assert fake_training == [("b2", "kw", 2, "heads")]
