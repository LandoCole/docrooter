"""Element tagger evaluation harness (spec §11, task L11): gold loading, metrics, both scoring
modes, calibration, learning-curve subsets and the doc-type element-evidence mix.

Fakes only: fake taggers, a tiny corpus store in ``tmp_path``, the dependency-free keyword
backend. No model is loaded and no real gold text is read.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from benchmarks import _guard
from benchmarks import element_tags_bench as bench
from docrooter.corpus.store import Corpus, CorpusElement, ManifestRecord
from docrooter.elements.models import TagResult
from docrooter.taxonomies import builtin_taxonomy

AR = "annual_report"
FD = "fund_document"


def element(doc_id: str, element_id: str, text: str, kind: str = "paragraph") -> CorpusElement:
    return CorpusElement(
        doc_id=doc_id,
        element_id=element_id,
        kind=kind,
        text=text,
        heading_path=["Section"],
        section_title="Section",
        page=1,
    )


def item(doc_id, element_id, family, topic, functions=(), text=None):
    return bench.GoldItem(
        doc_id=doc_id,
        element_id=element_id,
        family=family,
        topic=topic,
        functions=tuple(functions),
        element=element(doc_id, element_id, text if text is not None else topic),
    )


def tag_result(labels, probs, confidence, family_probs=None):
    return TagResult(
        labels=labels, probs=probs, confidence=confidence, source="fake", family_probs=family_probs
    )


class EchoTagger:
    """Predicts the leaf its element's text names when that leaf is in the family asked for,
    else ``<family>.other``; functions: the tags named after ``|`` in the text."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def tag(self, elements, family):
        self.calls.append((family, len(elements)))
        out = []
        for e in elements:
            topic, _, functions = e.text.partition("|")
            leaf = topic if topic.startswith(f"{family}.") else f"{family}.other"
            tags = [t for t in functions.split(",") if t]
            out.append(
                {
                    "topic": tag_result([leaf], {leaf: 0.75}, 0.75),
                    "function": tag_result(tags, {t: 0.9 for t in tags}, 0.9),
                }
            )
        return out


# --- topic metrics, both modes ----------------------------------------------------------------


def test_topic_metrics_primary_and_end_to_end():
    # By hand: the classes are those in gold (rf, other, perf); the stray "properties"
    # prediction is only a miss of its gold class, not a zero class of its own.
    gold = [f"{AR}.risk_factors", f"{AR}.risk_factors", f"{AR}.other", f"{FD}.performance"]
    pred = [f"{AR}.risk_factors", f"{AR}.other", f"{AR}.other", f"{AR}.properties"]
    m = bench.topic_metrics(gold, pred)
    assert m["n"] == 4
    assert m["accuracy"] == pytest.approx(0.5)
    assert (m["macro_over"], m["classes"]) == ("gold_classes", 3)
    assert set(m["per_class"]) == set(gold)
    assert m["macro_p"] == pytest.approx((1 + 0.5 + 0) / 3)
    assert m["macro_r"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 3)
    # The class set depends on gold only: another system's stray labels do not change it.
    other = bench.topic_metrics(gold, [f"{AR}.risk_factors", AR, f"{AR}.other", "x.y"])
    assert other["classes"] == 3 and set(other["per_class"]) == set(gold)
    # Hierarchical over family -> leaf: overlaps 2 + 1 + 2 + 0 of 8 predicted and 8 gold nodes.
    assert m["hierarchical"] == pytest.approx({"p": 5 / 8, "r": 5 / 8, "f1": 5 / 8})
    # A family-level prediction (an untrained leaf) earns the family node only.
    h = bench.hierarchical_prf([f"{AR}.risk_factors"], [AR])
    assert h == pytest.approx({"p": 1.0, "r": 0.5, "f1": 2 / 3})
    assert bench.topic_metrics([f"{AR}.other"], [None])["accuracy"] == 0.0

    # Primary uses each gold row's family; end to end the predicted doc type's family.
    items = [
        item("docA", "e1", AR, f"{AR}.risk_factors"),
        item("docA", "e2", AR, f"{AR}.properties"),
        item("docA", "e3", AR, f"{AR}.other"),
        item("docA", "e4", AR, f"{AR}.exhibits"),
        item("docB", "e1", FD, f"{FD}.performance"),
        item("docB", "e2", FD, f"{FD}.principal_risks"),
    ]
    tagger = EchoTagger()
    result = bench.evaluate_tagger(
        tagger, items, predicted_families={"docA": AR, "docB": "generic"}
    )
    primary, e2e = result["primary"], result["end_to_end"]
    assert primary["topic"]["accuracy"] == 1.0
    assert primary["topic"]["macro_f1"] == 1.0
    assert set(primary["topic_by_family"]) == {AR, FD}
    assert primary["topic_by_family"][FD]["accuracy"] == 1.0
    assert e2e["topic"]["accuracy"] == pytest.approx(4 / 6)
    assert e2e["topic_by_family"][AR]["accuracy"] == 1.0  # grouped by the gold family
    assert e2e["topic_by_family"][FD]["accuracy"] == 0.0
    assert e2e["topic"]["hierarchical"]["p"] == pytest.approx(8 / 12)
    assert e2e["families"] == {"agree": 4, "n": 6}
    # docA's family was right, so its primary tags are reused: only docB is tagged again.
    assert tagger.calls.count(("generic", 2)) == 1
    assert [c for c in tagger.calls if c[0] == AR and c[1] > 1] == [(AR, 4)]
    rows = {(p["doc_id"], p["element_id"]): p for p in result["predictions"]}
    assert rows[("docB", "e1")]["topic"] == f"{FD}.performance"
    assert rows[("docB", "e1")]["e2e_topic"] == "generic.other"
    assert rows[("docB", "e1")]["e2e_family"] == "generic"
    assert result["n"] == 6 and result["elements_per_s"] > 0

    without = bench.evaluate_tagger(EchoTagger(), items)
    assert without["end_to_end"] is None
    with pytest.raises(ValueError, match="docB"):
        bench.evaluate_tagger(EchoTagger(), items, predicted_families={"docA": AR})


# --- function metrics -------------------------------------------------------------------------


def test_function_micro_macro_f1():
    gold = [{"risk_disclosure"}, {"risk_disclosure", "financial_data"}, set(), {"definition"}]
    pred = [{"risk_disclosure"}, {"financial_data"}, {"risk_disclosure"}, set()]
    m = bench.function_metrics(gold, pred)
    assert m["micro"] == pytest.approx({"p": 2 / 3, "r": 1 / 2, "f1": 4 / 7})
    per = m["per_tag"]
    assert per["risk_disclosure"]["f1"] == pytest.approx(0.5)
    assert per["financial_data"]["f1"] == pytest.approx(1.0)
    assert per["definition"]["f1"] == 0.0
    assert per["definition"]["support"] == 1 and per["risk_disclosure"]["predicted"] == 2
    # A tag nobody gold-labelled or predicted has no defined F1 and stays out of the macro.
    assert per["citation"]["f1"] is None
    assert m["tags_scored"] == 3
    assert m["macro_f1"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert list(per) == [n.id for n in builtin_taxonomy("element_functions").leaves()]
    with pytest.raises(ValueError):
        bench.function_metrics([set()], [set(), set()])


# --- calibration ------------------------------------------------------------------------------


def test_ece():
    # Two bins: confidence 0.85 with accuracy 0.5, and 0.55 with accuracy 1.0, each half the n.
    value = bench.ece([0.85, 0.85, 0.55, 0.55], [True, False, True, True], n_bins=10)
    assert value == pytest.approx(0.5 * 0.35 + 0.5 * 0.45)
    assert bench.ece([1.0, 1.0], [True, True]) == 0.0
    assert bench.ece([0.0], [False]) == 0.0  # confidence 0 falls in the first bin
    assert bench.ece([], []) is None
    # 0.2 x 15 is 3.0000000000000004 in floats: 0.2 still belongs to the bin (2/15, 3/15].
    assert bench.ece([0.2, 0.19], [True, False]) == pytest.approx(abs(0.5 - 0.195))
    assert bench.ECE_BINS == 15


class FakeStudent:
    """Like ElementTagger: ``tag`` embeds through ``self.embed`` and tempers with
    ``self.temperatures``. Topic: right on even elements; confidence 0.99 untempered, 0.6 at the
    calibrated temperature."""

    def __init__(self) -> None:
        self.temperatures = {"topic": 2.0, "function": 3.0}
        self.embed_calls = 0

    def embed(self, texts):
        self.embed_calls += 1
        return [[float(len(t))] for t in texts]

    def tag(self, elements, family):
        vectors = self.embed([e.text for e in elements])
        assert len(vectors) == len(elements)
        conf = 0.99 if self.temperatures["topic"] == 1.0 else 0.6
        p_yes = 0.95 if self.temperatures["function"] == 1.0 else 0.7
        out = []
        for e in elements:
            right = int(e.element_id[1:]) % 2 == 0
            leaf = e.text if right else f"{family}.other"
            out.append(
                {
                    "topic": tag_result([leaf], {leaf: conf}, conf),
                    "function": tag_result(["financial_data"], {"financial_data": p_yes}, 0.7),
                }
            )
        return out


def test_student_ece_before_and_after_temperature():
    items = [item("d", f"e{k}", AR, f"{AR}.risk_factors") for k in range(4)]
    student = FakeStudent()
    result = bench.evaluate_tagger(student, items)
    cal = result["calibration"]
    assert cal["temperatures"] == {"topic": 2.0, "function": 3.0}
    assert cal["ece_before"]["topic"] == pytest.approx(abs(0.99 - 0.5))
    assert cal["ece_after"]["topic"] == pytest.approx(abs(0.6 - 0.5))
    assert result["primary"]["ece"]["topic"] == pytest.approx(cal["ece_after"]["topic"])
    # No gold function: P(yes) 0.95 / 0.7 are wrong decisions at confidence 0.95 / 0.7.
    assert cal["ece_before"]["function"] == pytest.approx(0.95)
    assert cal["ece_after"]["function"] == pytest.approx(0.7)
    # The student is restored, and the untempered pass reused the embeddings.
    assert student.temperatures == {"topic": 2.0, "function": 3.0}
    assert "embed" not in vars(student)
    assert student.embed_calls == 2  # warm-up batch + the one family batch
    assert bench.evaluate_tagger(EchoTagger(), items)["calibration"] is None


# --- learning curve ---------------------------------------------------------------------------


def test_learning_curve_fractions_nested():
    groups = ["a"] * 40 + ["b"] * 8 + ["c"] * 3
    fractions = bench.LEARNING_CURVE_FRACTIONS
    assert fractions == (0.125, 0.25, 0.5, 1.0)
    subsets = bench.nested_fraction_indices(groups, fractions, seed=0)
    ordered = [set(subsets[f]) for f in fractions]
    for small, large in zip(ordered, ordered[1:], strict=False):
        assert small < large  # strictly nested
    assert ordered[-1] == set(range(len(groups)))

    def counts(indices):
        return {g: sum(groups[i] == g for i in indices) for g in "abc"}

    # Stratified: each class keeps its share (rounded), never fewer than one example.
    assert counts(subsets[0.125]) == {"a": 5, "b": 1, "c": 1}
    assert counts(subsets[0.25]) == {"a": 10, "b": 2, "c": 1}
    assert counts(subsets[0.5]) == {"a": 20, "b": 4, "c": 2}
    assert subsets == bench.nested_fraction_indices(groups, fractions, seed=0)  # deterministic
    assert subsets[0.125] != bench.nested_fraction_indices(groups, fractions, seed=1)[0.125]
    assert bench.parse_fraction("1/8") == 0.125 and bench.parse_fraction("1") == 1.0
    with pytest.raises(ValueError):
        bench.parse_fraction("0")
    with pytest.raises(ValueError):
        bench.nested_fraction_indices(groups, (0.5, 1.5))


# --- doc type with element evidence ---------------------------------------------------------


def test_element_evidence_mix_fixed_weights():
    assert (bench.VIEW_WEIGHT, bench.ELEMENT_WEIGHT) == (0.7, 0.3)
    # Never tuned: neither the mix nor the driver's command line takes a weight.
    params = inspect.signature(bench.mix_doc_type).parameters
    assert not [p for p in params if "weight" in p or p in ("alpha", "beta", "w")]
    assert "weight" not in bench.build_parser().format_help().lower()
    for action in bench.build_parser()._subparsers._group_actions[0].choices.values():
        assert not [o for a in action._actions for o in a.option_strings if "weight" in o]

    view = {
        "financial.annual_report": 0.5,
        "financial.quarterly_report": 0.1,
        "research.research_paper": 0.4,
    }
    evidence = {AR: 0.2, "research_paper": 0.8}
    implied = bench.implied_doc_types(evidence, view)
    # Within a family the family's mass follows the view: 0.2 x 0.5/0.6, 0.2 x 0.1/0.6.
    assert implied["financial.annual_report"] == pytest.approx(0.2 * 0.5 / 0.6)
    assert implied["financial.quarterly_report"] == pytest.approx(0.2 * 0.1 / 0.6)
    assert implied["financial.proxy_statement"] == 0.0
    assert implied["research.research_paper"] == pytest.approx(0.8)
    assert sum(implied.values()) == pytest.approx(1.0)
    mixed = bench.mix_doc_type(view, evidence)
    assert mixed["financial.annual_report"] == pytest.approx(0.7 * 0.5 + 0.3 * 0.2 * 0.5 / 0.6)
    assert mixed["research.research_paper"] == pytest.approx(0.7 * 0.4 + 0.3 * 0.8)
    assert max(mixed, key=mixed.get) == "research.research_paper"  # the evidence flips it
    # A family the view gives nothing spreads its mass evenly over its doc types.
    spread = bench.implied_doc_types({FD: 1.0}, view)
    assert spread["financial.prospectus"] == pytest.approx(0.5)
    assert spread["financial.fact_sheet"] == pytest.approx(0.5)

    # A view that sums to less than 1 (leaf_probs keep the top 10 nodes) is renormalised
    # first, so the mix is 0.7/0.3 in effect: 0.7 x 0.3/0.5 and 0.7 x 0.2/0.5 + 0.3.
    short = {"financial.annual_report": 0.3, "research.research_paper": 0.2}
    assert bench.normalise_view(short, list(view))["financial.annual_report"] == pytest.approx(0.6)
    mixed = bench.mix_doc_type(short, {"research_paper": 1.0})
    assert mixed["financial.annual_report"] == pytest.approx(0.42)
    assert mixed["research.research_paper"] == pytest.approx(0.58)
    assert sum(mixed.values()) == pytest.approx(1.0)

    # The evidence is the mean unconstrained family distribution of the tagged elements.
    tags = [
        tag_result(["x"], {"x": 1.0}, 1.0, family_probs={AR: 0.8, FD: 0.2}),
        tag_result(["x"], {"x": 1.0}, 1.0, family_probs={AR: 0.4, FD: 0.6}),
        tag_result(["x"], {"x": 1.0}, 1.0),  # no family_probs (zero-shot): left out
        None,
    ]
    assert bench.element_family_evidence(tags) == pytest.approx({AR: 0.6, FD: 0.4})
    assert bench.element_family_evidence([None]) is None


# --- gold set ---------------------------------------------------------------------------------


def make_store(tmp_path: Path, docs: dict[str, tuple[str, list[CorpusElement]]]) -> Corpus:
    corpus = Corpus(tmp_path / "store")
    for doc_id, (split, elements) in docs.items():
        sha = hashlib.sha256(doc_id.encode()).hexdigest()
        corpus.add_document(ManifestRecord(id=doc_id, sha256=sha, split=split), elements, [1])
    return Corpus(tmp_path / "store")


def gold_row(e: CorpusElement, family: str, topic: str, functions=(), sha=None) -> dict:
    return {
        "doc_id": e.doc_id,
        "element_id": e.element_id,
        "text_sha1": sha or e.text_sha1,
        "family": family,
        "topic": topic,
        "functions": list(functions),
        "annotator": "A1",
        "adjudicated": False,
    }


def write_jsonl(path: Path, rows) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_gold_items_join_text_by_key_and_verify_sha1(tmp_path):
    a = [element("doca", "e1", "Risk factors text"), element("doca", "e2", "Other text")]
    t = [element("docb", "e1", "Training text")]
    corpus = make_store(tmp_path, {"doca": ("gold", a), "docb": ("train", t)})
    rows = [gold_row(a[1], AR, f"{AR}.other"), gold_row(a[0], AR, f"{AR}.risk_factors")]
    items = bench.gold_items(rows, corpus)
    assert [(i.element_id, i.element.text) for i in items] == [
        ("e2", "Other text"),
        ("e1", "Risk factors text"),
    ]
    assert items[1].topic == f"{AR}.risk_factors" and items[1].functions == ()

    with pytest.raises(bench.GoldTextError, match="text_sha1"):
        bench.gold_items([gold_row(a[0], AR, f"{AR}.other", sha="0" * 40)], corpus)
    missing = {**gold_row(a[0], AR, f"{AR}.other"), "element_id": "e9"}
    with pytest.raises(bench.GoldTextError, match="e9"):
        bench.gold_items([missing], corpus)
    with pytest.raises(bench.GoldTextError, match="gold split"):
        bench.gold_items([gold_row(t[0], AR, f"{AR}.other")], corpus)
    with pytest.raises(bench.GoldTextError, match="twice"):
        bench.gold_items([gold_row(a[0], AR, f"{AR}.other")] * 2, corpus)


def test_read_gold_checks_fields(tmp_path):
    good = gold_row(element("d", "e1", "x"), AR, f"{AR}.other")
    assert bench.read_gold(write_jsonl(tmp_path / "g.jsonl", [good])) == [good]
    bad = {k: v for k, v in good.items() if k != "topic"}
    with pytest.raises(bench.GoldTextError, match="topic"):
        bench.read_gold(write_jsonl(tmp_path / "b.jsonl", [bad]))


def test_gold_subset_is_stratified_and_deterministic():
    items = [item("d", f"a{k}", AR, f"{AR}.other") for k in range(60)]
    items += [item("d", f"f{k}", FD, f"{FD}.other") for k in range(30)]
    items += [item("d", f"g{k}", "generic", "generic.other") for k in range(10)]
    subset = bench.gold_subset(items, 30)
    assert len(subset) == 30
    families = [i.family for i in subset]
    assert (families.count(AR), families.count(FD), families.count("generic")) == (18, 9, 3)
    reversed_ids = {i.element_id for i in bench.gold_subset(list(reversed(items)), 30)}
    assert {i.element_id for i in subset} == reversed_ids  # by key, not by input order
    order = [items.index(i) for i in subset]
    assert order == sorted(order)  # kept in gold order
    assert bench.gold_subset(items, 500) == items


# --- systems ----------------------------------------------------------------------------------


def test_enriched_zero_shot_reads_the_mined_keywords():
    from docrooter.backends.fake import KeywordBackend
    from docrooter.teach.enrich import enrich_taxonomy

    enriched = enrich_taxonomy(
        builtin_taxonomy("element_topics"), {f"{AR}.properties": ["zebra", "giraffe"]}
    )
    e = [element("d", "e1", "zebra giraffe zebra okapi")]
    plain = bench.build_tagger("keyword")
    rich = bench.EnrichedEmbeddingTagger("keyword", enriched, backend=KeywordBackend())
    assert plain.tag(e, AR)[0]["topic"].labels != [f"{AR}.properties"]
    out = rich.tag(e, AR)[0]
    assert out["topic"].labels == [f"{AR}.properties"]
    assert set(out["function"].probs) == set(bench.function_tags())
    assert rich.source == "keyword+enriched"
    other = enrich_taxonomy(builtin_taxonomy("doc_types"), {})
    with pytest.raises(ValueError, match="leaves"):
        bench.EnrichedEmbeddingTagger("keyword", other, backend=KeywordBackend())


def test_build_tagger_refuses_jev():
    from docrooter.errors import TeacherNotAllowedError

    with pytest.raises(TeacherNotAllowedError):
        bench.build_tagger("teacher:jev:jev-1.13.0")


def test_guard_counts_this_harness_as_a_model_process():
    table = [(10, 1, "python3.12", "python -m benchmarks.element_tags_bench run --system x")]
    assert [p for p, _ in _guard.other_model_processes(table, pid=99, ppid=98)] == [10]


def test_run_keyword_system_writes_results_with_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")  # restored after the test; main() sets it
    calls = []
    monkeypatch.setattr(
        _guard, "single_model_process", lambda name, allow_concurrent=False: calls.append(name)
    )
    a = [element("doca", f"e{k}", f"risk factors uncertainty {k}") for k in range(3)]
    b = [element("docb", "e1", "fees and expenses of the fund")]
    corpus = make_store(tmp_path, {"doca": ("gold", a), "docb": ("gold", b)})
    rows = [gold_row(x, AR, f"{AR}.risk_factors", ["risk_disclosure"]) for x in a]
    rows.append(gold_row(b[0], FD, f"{FD}.fees_and_expenses", ["financial_data"]))
    gold = write_jsonl(tmp_path / "gold.jsonl", rows)
    out = tmp_path / "results"
    common = ["--store", str(corpus.dir), "--gold", str(gold), "--out", str(out)]

    assert bench.main(["run", "--system", "keyword", "--device", "cpu", *common]) == 0
    assert calls == ["benchmarks.element_tags_bench"]
    saved = json.loads((out / bench.RESULTS_FILE).read_text(encoding="utf-8"))
    assert saved["disclosure"] == "Claude-annotated gold"
    (run,) = saved["runs"]
    assert run["system"] == "keyword" and run["spec"] == "keyword" and run["device"] == "cpu"
    prov = run["provenance"]
    assert prov["gold_sha256"] == hashlib.sha256(gold.read_bytes()).hexdigest()
    assert "commit" in prov and prov["bundle_sha256"] is None and prov["teacher"] is None
    assert run["n"] == 4 and run["end_to_end"] is None
    assert run["primary"]["topic"]["n"] == 4
    assert {"peak_total_gb", "peak_rss_gb", "elements_per_s"} <= set(run)
    assert not any("text" in p for p in run["predictions"])  # no gold text in results

    # End to end with the predicted doc types; a file made for other gold is refused.
    doc_types = {
        "gold_sha256": prov["gold_sha256"],
        "documents": {
            "doca": {"doc_type": "financial.annual_report", "family": AR},
            "docb": {"doc_type": "research.research_paper", "family": "research_paper"},
        },
    }
    path = tmp_path / "doctypes.json"
    path.write_text(json.dumps(doc_types), encoding="utf-8")
    args = ["run", "--system", "keyword", "--device", "cpu", "--doc-types", str(path), *common]
    assert bench.main(args) == 0
    (run,) = json.loads((out / bench.RESULTS_FILE).read_text(encoding="utf-8"))["runs"]
    assert run["end_to_end"]["families"] == {"agree": 3, "n": 4}
    assert run["provenance"]["doc_types_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    path.write_text(json.dumps({**doc_types, "gold_sha256": "f" * 64}), encoding="utf-8")
    with pytest.raises(SystemExit, match="gold"):
        bench.main(args)

    # A subset run is its own row.
    assert bench.main(["run", "--system", "kw-2", "--tagger", "keyword", "--subset", "2",
                       "--device", "cpu", *common]) == 0  # fmt: skip
    runs = json.loads((out / bench.RESULTS_FILE).read_text(encoding="utf-8"))["runs"]
    got = [(r["system"], r["n"], r["subset"]) for r in runs]
    assert got == [("keyword", 4, None), ("kw-2", 2, 2)]
    # A subset run of the same system is its own row too: it never replaces the full run.
    assert (
        bench.main(["run", "--system", "keyword", "--subset", "2", "--device", "cpu", *common]) == 0
    )
    runs = json.loads((out / bench.RESULTS_FILE).read_text(encoding="utf-8"))["runs"]
    got = [(r["system"], r["n"], r["subset"]) for r in runs]
    assert got == [("keyword", 4, None), ("kw-2", 2, 2), ("keyword", 2, 2)]


# --- drivers, with the models faked ---------------------------------------------------------


@pytest.fixture
def no_guard(monkeypatch):
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")
    monkeypatch.setattr(_guard, "single_model_process", lambda name, allow_concurrent=False: None)


def gold_store(tmp_path):
    a = [element("doca", f"e{k}", f"{AR}.risk_factors") for k in range(3)]
    b = [element("docb", "e1", f"{FD}.performance")]
    corpus = make_store(tmp_path, {"doca": ("gold", a), "docb": ("gold", b)})
    rows = [gold_row(x, AR, f"{AR}.risk_factors") for x in a]
    rows.append(gold_row(b[0], FD, f"{FD}.performance"))
    return corpus, write_jsonl(tmp_path / "gold.jsonl", rows)


class FakeBundle(EchoTagger):
    """What train_tagger returns: an EchoTagger with temperatures, info, save and close."""

    def __init__(self, info=None) -> None:
        super().__init__()
        self.temperatures = {"topic": 1.0, "function": 1.0}
        self.info = dict(info or {})
        self.closed = False

    def save(self, out):
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "tagger.json").write_text(json.dumps({"info": self.info}), encoding="utf-8")

    def close(self):
        self.closed = True


def curve_setup(tmp_path, monkeypatch, doc_id="x"):
    """A gold store, a train-elements work dir (24 confident examples of ``doc_id``, 10
    function rows, fingerprint ``abc``) and faked training; returns the argv to extend with
    ``--fractions``, the examples, the trained calls and the store."""
    from docrooter.tagger import bundle as bundle_module
    from docrooter.tagger import train as train_module
    from docrooter.tagger.functions import FunctionRow
    from docrooter.train import Example

    corpus, gold = gold_store(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    examples = [
        Example(text=f"t{k}", path=[AR, f"{AR}.{leaf}"], doc_id=doc_id, section_id=f"e{k}")
        for k, leaf in enumerate(["other"] * 16 + ["properties"] * 8)
    ]
    write_jsonl(work / "train.jsonl", [e.model_dump(exclude_none=True) for e in examples])
    write_jsonl(work / "calib.jsonl", [examples[0].model_dump(exclude_none=True)])
    rows = [FunctionRow(text=f"f{k}", answers={"definition": 0.9}) for k in range(10)]
    write_jsonl(work / "functions-train.jsonl", [r.model_dump() for r in rows])
    write_jsonl(work / "functions-calib.jsonl", [rows[0].model_dump()])
    (work / "dataset.json").write_text(json.dumps({"fingerprint": "abc"}), encoding="utf-8")
    trained = []

    def fake_train(dataset, function_rows, **options):
        trained.append((sorted(e.section_id for e in dataset), len(function_rows), options))
        return FakeBundle({"untrained_leaves": []})

    monkeypatch.setattr(train_module, "train_tagger", fake_train)
    monkeypatch.setattr(train_module, "calibrate_tagger", lambda tagger, calib, rows: {})
    monkeypatch.setattr(
        bundle_module.ElementTagger,
        "load",
        classmethod(lambda cls, path, device=None: FakeBundle()),
    )
    argv = ["learning-curve", "--stage", "A", "--work", str(work)]
    argv += ["--bundle-root", str(tmp_path / "bundles"), "--store", str(corpus.dir)]
    argv += ["--gold", str(gold), "--out", str(tmp_path / "results"), "--device", "cpu"]
    return argv, examples, trained, corpus


def write_teacher_labels(store: Path, rows) -> None:
    (store / "labels").mkdir(parents=True, exist_ok=True)
    write_jsonl(store / "labels" / "teacher.jsonl", rows)


def teacher_label(doc_id, element_id, teacher="mlx:org/model@rev1#float16", source=None):
    provenance = {"teacher": teacher, **({"source": source} if source else {})}
    return {"doc_id": doc_id, "element_id": element_id, "text_sha1": "0" * 40,
            "facet": "topic", "probs": {"a": 1.0}, "provenance": provenance}  # fmt: skip


def test_learning_curve_point_trains_the_nested_subset_once(tmp_path, monkeypatch, no_guard):
    base, examples, trained, corpus = curve_setup(tmp_path, monkeypatch)
    out = tmp_path / "results"
    write_teacher_labels(corpus.dir, [teacher_label("x", "e0")])
    argv = [*base, "--fractions", "1/2"]
    assert bench.main(argv) == 0
    ((ids, n_rows, options),) = trained
    assert len(ids) == 12 and n_rows == 5  # stratified: 8 of 16 "other", 4 of 8 "properties"
    expected = bench.nested_fraction_indices([">".join(e.path) for e in examples], [0.5])[0.5]
    assert ids == sorted(examples[i].section_id for i in expected)
    assert options["stage"] == "A" and options["device"] == "cpu" and options["seed"] == 0
    (run,) = json.loads((out / bench.CURVE_FILE).read_text(encoding="utf-8"))["runs"]
    assert (run["stage"], run["fraction"], run["examples"], run["how"]) == (
        "A",
        "1/2",
        12,
        "trained",
    )
    assert run["macro_f1"] == 1.0 and run["dataset"] == "abc"
    # Provenance: the labelling teacher with its dtype, the label files' hashes, no gold.
    (teacher,) = run["training"]["teachers"]
    assert (teacher["name"], teacher["dtype"]) == ("mlx:org/model@rev1#float16", "float16")
    assert set(run["training"]["label_files_sha256"]) == {"teacher"}
    assert run["gold_leakage_check"] == {"checked": 25, "without_doc_id": 0}
    assert run["active_learning_examples"] == 0
    info = json.loads((tmp_path / "bundles" / "stage-a-1of2" / "tagger.json").read_text())["info"]
    assert info["learning_curve"]["examples"] == 12

    assert bench.main(argv) == 0  # the same point again: the saved bundle is reused
    assert len(trained) == 1
    (run,) = json.loads((out / bench.CURVE_FILE).read_text(encoding="utf-8"))["runs"]
    assert run["how"] == "cached"


def test_learning_curve_runs_each_fraction_in_its_own_process(tmp_path, monkeypatch, no_guard):
    commands = []

    class Done:
        returncode = 0

    monkeypatch.setattr(
        bench.subprocess, "run", lambda command, check: commands.append(command) or Done()
    )
    assert bench.main(["learning-curve", "--stage", "B", "--out", str(tmp_path)]) == 0
    fractions = [c[c.index("--fractions") + 1] for c in commands]
    assert fractions == ["1/8", "1/4", "1/2", "1"]
    assert all(c[c.index("--stage") + 1] == "B" for c in commands)
    assert all(c[c.index("--out") + 1] == str(tmp_path) for c in commands)


def test_learning_curve_refuses_gold_split_training_data(tmp_path, monkeypatch, no_guard):
    base, _, trained, _ = curve_setup(tmp_path, monkeypatch, doc_id="doca")  # a gold document
    with pytest.raises(bench.GoldTextError, match="gold-split"):
        bench.main([*base, "--fractions", "1/2"])
    assert trained == []


def test_full_bundle_must_be_the_pre_active_learning_confident_set(tmp_path, monkeypatch, no_guard):
    base, _, trained, corpus = curve_setup(tmp_path, monkeypatch)
    full = tmp_path / "full"

    def full_bundle(info):
        full.mkdir(exist_ok=True)
        (full / "tagger.json").write_text(json.dumps({"info": info}), encoding="utf-8")
        return [*base, "--fractions", "1", "--full-bundle", str(full)]

    current = {"stage": "A", "train_elements": {"stage": "A", "dataset": "abc"}}
    with pytest.raises(SystemExit, match="confident-set fingerprint"):
        bench.main(full_bundle({**current, "train_elements": {"stage": "A", "dataset": "old"}}))
    with pytest.raises(SystemExit, match="stage B bundle"):
        bench.main(full_bundle({"stage": "B", "train_elements": {"stage": "B", "dataset": "abc"}}))
    with pytest.raises(SystemExit, match="pre-active-learning"):
        bench.main(full_bundle({**current, "active_learning": {"n": 3000}}))
    # The confident set itself holds an element the active-learning round labelled.
    write_teacher_labels(corpus.dir, [teacher_label("x", "e3", source="active_learning")])
    with pytest.raises(SystemExit, match="pre-active-learning"):
        bench.main(full_bundle(current))
    assert trained == []

    write_teacher_labels(corpus.dir, [teacher_label("x", "e3")])
    assert bench.main(full_bundle(current)) == 0
    assert trained == []  # reused, not trained
    (run,) = json.loads((tmp_path / "results" / bench.CURVE_FILE).read_text())["runs"]
    assert (run["fraction"], run["how"]) == ("1", "reused")


class FakeEngine:
    """A teacher engine: picks each item's first option, answers every question 0.1."""

    name = "mlx:fake/model@abc#float16"

    def choose(self, items, *, instruction, prefix_key=None, prefix_options=None):
        return [{item.options[0][0]: 1.0} for item in items]

    def yes_no(self, items, *, question):
        return [0.1] * len(items)

    def close(self):
        pass


def test_teacher_items_per_second_is_counted(tmp_path):
    from docrooter.tagger.element_tags import TeacherTagger

    engine = FakeEngine()
    tagger = TeacherTagger("mlx:fake/model@abc", teacher=engine)
    items = [item("d", f"e{k}", AR, f"{AR}.shareholder_letter") for k in range(3)]
    result = bench.evaluate_tagger(tagger, items)
    # The primary pass only (not the warm-up): one topic choice and 12 questions per element.
    assert result["teacher_items"] == 3 * 13
    assert result["teacher_items_per_s"] > 0
    assert tagger.teacher is engine  # restored
    assert bench.evaluate_tagger(EchoTagger(), items)["teacher_items"] is None
    gold = write_jsonl(tmp_path / "g.jsonl", [])
    prov = bench.provenance("teacher:mlx:fake/model@abc", tagger, gold_path=gold)
    assert (prov["teacher"]["name"], prov["teacher"]["dtype"]) == (FakeEngine.name, "float16")


def test_student_run_records_the_labelling_teacher(tmp_path, monkeypatch, no_guard):
    corpus, gold = gold_store(tmp_path)
    write_teacher_labels(
        corpus.dir,
        [teacher_label("x", "e1"), teacher_label("x", "e2", source="active_learning")],
    )
    (corpus.dir / "labels" / "prescreen.jsonl").write_text("", encoding="utf-8")
    work = corpus.dir / bench.WORK_DIR
    work.mkdir()
    (work / "dataset.json").write_text(json.dumps({"fingerprint": "fp1"}), encoding="utf-8")
    settings = {"teacher": "mlx:org/model@rev1"}
    (work / "label.json").write_text(json.dumps({"settings": settings}), encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "tagger.json").write_text("{}", encoding="utf-8")
    student = FakeBundle({"stage": "B", "train_elements": {"stage": "B", "dataset": "fp1"}})
    student.path = bundle_dir
    monkeypatch.setattr(bench, "build_tagger", lambda spec, **options: student)
    out = tmp_path / "results"
    argv = ["run", "--system", "stage-b", "--tagger", str(bundle_dir), "--store", str(corpus.dir)]
    assert bench.main([*argv, "--gold", str(gold), "--out", str(out), "--device", "cpu"]) == 0
    (run,) = json.loads((out / bench.RESULTS_FILE).read_text(encoding="utf-8"))["runs"]
    prov = run["provenance"]
    assert prov["bundle_sha256"] == bench.bundle_sha256(bundle_dir)
    assert prov["bundle_info"]["train_elements"] == {"stage": "B", "dataset": "fp1"}
    training = prov["training"]
    (teacher,) = training["teachers"]
    assert teacher == {
        "name": "mlx:org/model@rev1#float16",
        "rows": 2,
        "engine": "mlx",
        "model": "org/model",
        "revision": "rev1",
        "dtype": "float16",
    }
    assert training["active_learning_rows"] == 1
    assert set(training["label_files_sha256"]) == {"teacher", "prescreen"}
    assert training["label_settings"] == settings
    assert training["dataset_fingerprint"] == "fp1" and prov["bundle_matches_dataset"] is True
    assert run["calibration"] is not None  # a student: ECE before and after temperature


def fake_analysis(leaf, leaf_probs, elements=()):
    from types import SimpleNamespace

    best = SimpleNamespace(node_ids=[leaf] if leaf else [])
    doc_type = SimpleNamespace(best=best, confidence=0.9, leaf_probs=leaf_probs, source="fake")
    return SimpleNamespace(doc_type=doc_type, document=SimpleNamespace(elements=list(elements)))


def data_inputs_ok(out):
    data = json.loads((out / bench.GOLD_DOCTYPES_FILE).read_text(encoding="utf-8"))
    return "neutral" in data["inputs"]


def test_gold_doctypes_writes_what_run_reads(tmp_path, monkeypatch, no_guard):
    import docrooter.rooter

    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    corpus = Corpus(tmp_path / "store")
    a = [element("doca", "e1", f"{AR}.other")]
    b = [element("docb", "e1", f"{FD}.performance")]
    for doc_id, elements in (("doca", a), ("docb", b)):
        (pdfs / f"{doc_id}.pdf").write_bytes(f"%PDF-{doc_id}".encode())
        sha = hashlib.sha256(f"%PDF-{doc_id}".encode()).hexdigest()
        corpus.add_document(ManifestRecord(id=doc_id, sha256=sha, split="gold"), elements, [1])
    gold = write_jsonl(
        tmp_path / "gold.jsonl",
        [gold_row(a[0], AR, f"{AR}.other"), gold_row(b[0], FD, f"{FD}.performance")],
    )
    seen = {}

    def fake_process_many(paths, **config):
        paths = list(paths)
        seen.update(config, paths=[Path(p).name for p in paths])
        seen.update(full_paths=[str(p) for p in paths], readable=[Path(p).is_file() for p in paths])
        yield fake_analysis("financial.annual_report", {"financial.annual_report": 0.9})
        yield fake_analysis(None, {})

    monkeypatch.setattr(docrooter.rooter, "process_many", fake_process_many)
    out = tmp_path / "results"
    common = ["--store", str(corpus.dir), "--gold", str(gold), "--out", str(out), "--device", "cpu"]
    assert bench.main(["gold-doctypes", "--pdf-dir", str(pdfs), *common]) == 0
    # The model sees neutral copies, never a source-revealing id (the view's title falls back
    # to the file stem).
    shas = [hashlib.sha256(f"%PDF-{d}".encode()).hexdigest() for d in ("doca", "docb")]
    assert seen["paths"] == [f"doc_{sha[:12]}.pdf" for sha in shas]
    assert not [p for p in seen["full_paths"] if "doca" in p or "docb" in p]
    assert all(bench.NEUTRAL_NAME.fullmatch(name) for name in seen["paths"])
    assert seen["readable"] == [True, True]  # the copies exist while the model reads them
    assert data_inputs_ok(out)
    assert seen["profile"] == "balanced" and seen["escalate"] == "none"
    assert seen["element_tags"] == "none"
    data = json.loads((out / bench.GOLD_DOCTYPES_FILE).read_text(encoding="utf-8"))
    assert data["documents"]["docb"]["family"] == "generic"  # no doc type: the generic family
    assert data["doc_type_sources"] == ["fake"]
    assert data["family_agreement"] == {"agree": 1, "n": 2}
    assert bench.read_predicted_families(out / bench.GOLD_DOCTYPES_FILE, gold) == {
        "doca": AR,
        "docb": "generic",
    }

    (pdfs / "docb.pdf").write_bytes(b"%PDF-changed")
    with pytest.raises(bench.GoldTextError, match="not the PDF"):
        bench.main(["gold-doctypes", "--pdf-dir", str(pdfs), *common])


def test_doctype_evidence_reports_view_and_fixed_mix(tmp_path, monkeypatch, no_guard):
    from types import SimpleNamespace

    import docrooter.rooter
    from benchmarks import corpora

    dev = tmp_path / "dev"
    (dev / "pdfs").mkdir(parents=True)
    for name in ("doc_1", "doc_2"):
        (dev / "pdfs" / f"{name}.pdf").write_bytes(f"%PDF-{name}".encode())
    write_jsonl(
        dev / "manifest.jsonl",
        [
            {"id": "doc_1", "file": "pdfs/doc_1.pdf", "label": "research.research_paper"},
            {"id": "doc_2", "file": "pdfs/doc_2.pdf", "label": "financial.annual_report"},
        ],
    )
    monkeypatch.setattr(corpora, "DEV", dev)
    paper_tags = [
        SimpleNamespace(
            tags={"topic": tag_result(["x"], {"x": 1.0}, 1.0, {"research_paper": 0.9, AR: 0.1})}
        ),
        SimpleNamespace(tags=None),  # a heading: not tagged
    ]
    seen = {}

    def fake_process_many(paths, **config):
        seen.update(config, names=[Path(p).name for p in paths])
        view = {"financial.annual_report": 0.5, "research.research_paper": 0.4}
        yield fake_analysis("financial.annual_report", view, paper_tags)
        # The rooter abstains (low confidence) although its view's argmax is right.
        yield fake_analysis(None, {"financial.annual_report": 0.3})

    monkeypatch.setattr(docrooter.rooter, "process_many", fake_process_many)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "tagger.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "results"
    argv = ["doctype-evidence", "--bundle", str(bundle), "--out", str(out), "--device", "cpu"]
    assert bench.main(argv) == 0
    assert seen["element_tags"] == str(bundle) and seen["profile"] == "accurate"
    assert all(bench.NEUTRAL_NAME.fullmatch(name) for name in seen["names"])
    assert not [n for n in seen["names"] if n in ("doc_1.pdf", "doc_2.pdf")]
    data = json.loads((out / bench.EVIDENCE_FILE).read_text(encoding="utf-8"))
    assert data["weights"] == {"view": 0.7, "elements": 0.3, "tuned": False}
    manifest_sha = hashlib.sha256((dev / "manifest.jsonl").read_bytes()).hexdigest()
    assert data["dev_manifest_sha256"] == manifest_sha
    first, second = data["rows"]
    # pred_view is the rooter's own doc type; the view is renormalised before the mix.
    assert (first["pred_view"], first["pred_mix"], first["n_tagged"]) == (
        "financial.annual_report",
        "research.research_paper",
        1,
    )
    assert first["view_mass"] == pytest.approx(0.9) and first["view_leaves"] == 2
    assert first["view_top3"]["financial.annual_report"] == pytest.approx(0.5 / 0.9, abs=1e-4)
    p_paper = 0.7 * 0.4 / 0.9 + 0.3 * 0.9
    assert first["mix_top3"]["research.research_paper"] == pytest.approx(p_paper, abs=1e-4)
    assert (second["pred_view"], second["pred_view_argmax"]) == (None, "financial.annual_report")
    assert second["evidence"] == {} and second["pred_mix"] == "financial.annual_report"
    assert second["view_top3"] == {"financial.annual_report": 1.0}
    summary = data["summary"]
    assert (summary["view_accuracy"], summary["view_argmax_accuracy"]) == (0.0, 0.5)
    assert summary["mix_accuracy"] == 1.0
    assert (summary["changed"], summary["fixed"], summary["broken"]) == (2, 2, 0)
    assert summary["no_evidence"] == 1
    assert data["bundle_sha256"] == bench.bundle_sha256(bundle)
