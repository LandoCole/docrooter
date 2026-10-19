"""Term mining, the similarity-gap filter and the training set (spec §8, §4.0 ruling 7), with a
fake embedder: no model, torch or network."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from docrooter import Taxonomy
from docrooter.corpus import Corpus, CorpusElement, LabelRow, ManifestRecord, text_sha1
from docrooter.errors import TeacherNotAllowedError
from docrooter.teach.confidence import (
    TrainingSet,
    class_centroids,
    confident_set,
    core_classes,
    gap_core_classes,
)
from docrooter.teach.dataset import build_training_set, input_text, to_dataset
from docrooter.teach.enrich import (
    TermCounts,
    count_terms,
    element_ngrams,
    enrich_taxonomy,
    mine_terms,
    teacher_choices,
    term_affinity,
)

# --- fixtures ----------------------------------------------------------------------------------

AXES = ("fee", "expense", "risk", "loss", "method", "fund", "paper")


class FakeEmbedder:
    """Counts a few stems per axis, plus a small constant axis, L2-normalised: "management fees"
    and the label "Fees" point the same way, "Other" only along the constant axis."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str]] = []

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        self.calls.append((list(texts), prompt))
        rows = []
        for text in texts:
            v = np.zeros(len(AXES) + 1)
            v[-1] = 0.1
            for word in re.findall(r"[a-z]+", text.lower()):
                for i, axis in enumerate(AXES):
                    if word.startswith(axis):
                        v[i] += 1.0
            rows.append(v / np.linalg.norm(v))
        return np.asarray(rows, dtype=np.float32)


def _taxonomy() -> Taxonomy:
    return Taxonomy.from_dict(
        {
            "nodes": [
                {"id": "fund", "label": "Fund document"},
                {"id": "fund.fees", "label": "Fees", "parent_id": "fund"},
                {"id": "fund.risks", "label": "Risks", "parent_id": "fund"},
                {"id": "fund.other", "label": "Other", "parent_id": "fund"},
                {"id": "paper", "label": "Research paper"},
                {"id": "paper.method", "label": "Method", "parent_id": "paper"},
                {"id": "paper.other", "label": "Other", "parent_id": "paper"},
            ]
        }
    )


def _sha(n: int) -> str:
    return f"{n:064x}"


def _corpus(tmp_path, docs, *, split="train", start=1) -> Corpus:
    """``docs``: {doc_id: [(text, kind, heading_path), ...]}."""
    corpus = Corpus(tmp_path / "corpus")
    for n, (doc_id, items) in enumerate(docs.items(), start):
        record = ManifestRecord(id=doc_id, sha256=_sha(n), split=split)
        elements = [
            CorpusElement(
                doc_id=doc_id,
                element_id=f"e{i}",
                kind=kind,
                text=text,
                heading_path=list(path),
                section_title=path[-1] if path else None,
                page=1,
            )
            for i, (text, kind, path) in enumerate(items)
        ]
        corpus.add_document(record, elements, [1])
    return corpus


def _row(doc_id, element_id, text, probs, *, facet="topic", **provenance) -> LabelRow:
    return LabelRow(
        doc_id=doc_id,
        element_id=element_id,
        text_sha1=text_sha1(text),
        facet=facet,
        probs=probs,
        provenance=provenance
        or {"teacher": "mlx:mlx-community/Qwen3-4B-Instruct-2507-4bit@50d4277"},
    )


def _key(doc_id, element_id, text="x"):
    return _row(doc_id, element_id, text, {"fund.fees": 1.0}).key


# --- term mining -------------------------------------------------------------------------------


def test_term_scores_formula():
    taxonomy = _taxonomy()
    classes = ["fund", "paper", "fund.fees", "fund.risks", "fund.other"]
    terms = ["fee", "market risk", "net"]
    tf = np.array([[9, 4, 3], [0, 1, 2], [8, 0, 1], [1, 4, 1], [0, 0, 1]])
    df = np.array([[6, 3, 3], [0, 1, 2], [5, 0, 1], [1, 3, 1], [0, 0, 1]])
    length = np.array([300, 120, 100, 150, 50])
    cos = np.array(
        [[0.2, 0.1, 0.3], [0.1, 0.0, 0.3], [0.9, 0.1, -0.2], [0.1, 0.8, 0.3], [0.2, 0.2, 0.3]]
    )
    counts = TermCounts(terms, classes, tf, df, length)

    aff = term_affinity(counts, cos, taxonomy)

    parent = {c: taxonomy.node(c).parent_id for c in classes}
    for c, cname in enumerate(classes):
        group = [g for g, gname in enumerate(classes) if parent[gname] == parent[cname]]
        avg = sum(length[g] for g in group) / len(group)
        for t in range(len(terms)):
            n_holding = sum(1 for g in group if tf[g, t] > 0)
            idf = math.log(1 + (len(group) - n_holding + 0.5) / (n_holding + 0.5))

            def bm(g, t=t, idf=idf, avg=avg):
                norm = 1 - 0.75 + 0.75 * length[g] / avg
                return idf * tf[g, t] * 2.5 / (tf[g, t] + 1.5 * norm)

            dist = math.exp(bm(c)) / (1 + sum(math.exp(bm(g)) for g in group if g != c))
            pop = math.log(1 + df[c, t])
            expected = pop**0.5 * dist**0.5 * max(0.0, cos[c, t])
            assert aff[c, t] == pytest.approx(expected), (cname, terms[t])

    assert aff[classes.index("fund.fees"), terms.index("net")] == 0.0  # negative cosine
    assert aff[classes.index("paper"), terms.index("fee")] == 0.0  # absent: pop = 0
    fees = classes.index("fund.fees")
    assert aff[fees, 0] > aff[fees, 2]


def test_count_terms_keeps_frequent_ngrams_without_stopword_edges():
    texts = (
        ["The management fee of the fund. Risk"] * 5  # "fee risk" spans punctuation
        + ["rare widget here"] * 4  # df 4 < 5
        + ["fund of funds; 2023 2023"] * 5
    )
    source = lambda: [(t, ["fund.fees"]) for t in texts]  # noqa: E731

    counts = count_terms(source, ["fund.fees", "fund.risks"])

    terms = set(counts.terms)
    assert {"management fee", "management", "fee", "fund", "fund of funds", "funds"} <= terms
    assert "the management fee" not in terms and "fee of the" not in terms  # stop-word edges
    assert "fee risk" not in terms  # never across punctuation
    assert not {"rare", "widget", "rare widget"} & terms  # df 4
    assert "2023" not in terms  # bare numbers make no edge
    i = counts.terms.index("fund")
    assert counts.df[0, i] == 10 and counts.tf[0, i] == 10
    assert counts.df[1].sum() == 0 and counts.length[1] == 0
    assert counts.length[0] == sum(len(re.findall(r"[a-z0-9]+", t.lower())) for t in texts)
    # the sketch only screens: with every n-gram colliding, the counts are the same
    crowded = count_terms(source, ["fund.fees", "fund.risks"], sketch_width=2)
    assert crowded.terms == counts.terms
    assert (crowded.tf == counts.tf).all() and (crowded.df == counts.df).all()


def _fund_corpus(tmp_path):
    fees = [f"The management fees are {n} percent of net assets." for n in range(8)]
    risks = [f"Market risk may cause a loss of {n} percent of net assets." for n in range(8)]
    other = [f"Contact the transfer agent at office {n} for net assets." for n in range(6)]
    items = [(t, "paragraph", ["Fees"]) for t in fees] + [
        (t, "paragraph", ["Risks"]) for t in risks
    ]
    items += [(t, "paragraph", ["Contact"]) for t in other]
    corpus = _corpus(tmp_path, {"doc1": items})
    labels = []
    for i, (text, _, _) in enumerate(items):
        leaf = "fund.fees" if i < 8 else "fund.risks" if i < 16 else "fund.other"
        others = [x for x in ("fund.fees", "fund.risks", "fund.other") if x != leaf]
        labels.append(_row("doc1", f"e{i}", text, {leaf: 0.7, others[0]: 0.2, others[1]: 0.1}))
    return corpus, labels, items


def test_mine_terms_ranks_class_terms_and_enriches(tmp_path):
    corpus, labels, _ = _fund_corpus(tmp_path)
    embedder = FakeEmbedder()

    terms = mine_terms(corpus, labels, _taxonomy(), embedder, k=3)

    assert set(terms) == set(_taxonomy().nodes)
    assert terms["fund.fees"][0] in {"management fees", "fees"}
    assert set(terms["fund.fees"]) <= {"management fees", "fees", "management", "net assets"}
    assert terms["fund.risks"][0] in {"market risk", "risk", "loss"}
    assert terms["paper.method"] == []  # no element chose it
    assert len(terms["fund.fees"]) <= 3
    assert all(prompt == "" for _, prompt in embedder.calls)

    enriched = enrich_taxonomy(_taxonomy(), terms)
    assert enriched.nodes["fund.fees"].keywords == terms["fund.fees"]
    assert _taxonomy().nodes["fund.fees"].keywords == []
    assert enriched.content_hash() != _taxonomy().content_hash()
    with pytest.raises(KeyError):
        enrich_taxonomy(_taxonomy(), {"nope": ["x"]})


def test_mine_terms_uses_only_labelled_train_elements(tmp_path):
    corpus, labels, items = _fund_corpus(tmp_path)
    # unlabelled train elements and calib elements full of one term are never mined
    _corpus(
        tmp_path, {"cal": [("zebra crossing fees", "paragraph", [])] * 6}, split="calib", start=9
    )
    extra = [("zebra crossing fees", "paragraph", [])] * 6
    _corpus(tmp_path, {"doc2": extra}, start=20)
    corpus = Corpus(tmp_path / "corpus")  # a fresh reader sees all three documents
    assert len(list(corpus.iter_elements())) == len(items) + 12
    labels += [_row("cal", f"e{i}", "zebra crossing fees", {"fund.fees": 0.9}) for i in range(6)]

    terms = mine_terms(corpus, labels, _taxonomy(), FakeEmbedder(), k=10)

    assert not any("zebra" in t for ts in terms.values() for t in ts)


def test_teacher_choices_take_the_argmax_and_check_leaves():
    rows = [
        _row("d", "e1", "a", {"fund.fees": 0.3, "fund.risks": 0.6, "fund.other": 0.1}),
        _row("d", "e2", "b", {"forward_looking": 0.9}, facet="function"),
    ]
    choices = teacher_choices(rows, _taxonomy())
    assert list(choices.values()) == [("fund.risks", 0.6)]
    with pytest.raises(ValueError, match="not a leaf"):
        teacher_choices([_row("d", "e1", "a", {"fund": 1.0})], _taxonomy())


@pytest.mark.parametrize(
    "provenance",
    [{"teacher": "jev:typesafe/jev-1"}, {"engine": "jev"}, {"model": "jev-2"}],
)
def test_jev_labels_are_refused(provenance):
    rows = [_row("d", "e1", "a", {"fund.fees": 0.9}, **provenance)]
    with pytest.raises(TeacherNotAllowedError):
        teacher_choices(rows, _taxonomy())
    with pytest.raises(TeacherNotAllowedError):
        confident_set(rows, {rows[0].key: (["fund.fees"], 0.5)}, taxonomy=_taxonomy())


# --- centroids and the gap ---------------------------------------------------------------------


def test_core_classes_above_largest_gap():
    assert gap_core_classes({"d": 0.35, "a": 0.9, "c": 0.4, "b": 0.85}) == (
        ["a", "b"],
        pytest.approx(0.45),
    )
    assert gap_core_classes({"a": 0.9, "b": 0.2, "c": 0.15}) == (["a"], pytest.approx(0.7))
    assert gap_core_classes({"a": 0.3}) == (["a"], 0.0)
    assert gap_core_classes({}) == ([], 0.0)
    # equal largest gaps: the first (smallest core)
    assert gap_core_classes({"a": 0.9, "b": 0.6, "c": 0.3})[0] == ["a"]

    centroids = {"x": np.array([1.0, 0.0]), "y": np.array([0.0, 1.0]), "z": np.array([0.6, 0.8])}
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
    cores = core_classes(vectors, [["x", "y", "z"], ["x", "y", "missing"]], centroids)
    assert cores[0] == (["x", "z"], pytest.approx(0.6))
    assert cores[1] == (["y"], pytest.approx(1.0))  # a class with no centroid is not scored


def test_centroids_mean_of_keyword_elements_with_fallback():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])
    centroids = class_centroids(vectors, ["a", "a", "b", "b"], [True, False, False, False])
    np.testing.assert_allclose(centroids["a"], [1.0, 0.0])  # only the keyword element
    expected = np.array([1.6, 0.8]) / np.linalg.norm([1.6, 0.8])
    np.testing.assert_allclose(centroids["b"], expected)  # no keyword element: all of them


# --- the confident set -------------------------------------------------------------------------


def _labelled(cls, n, *, prob=0.6, doc="d", start=0):
    others = [c for c in ("fund.fees", "fund.risks", "fund.other") if c != cls]
    rest = (1 - prob) / 2
    return [
        _row(doc, f"{cls}-{i}", f"{cls} {i}", {cls: prob, others[0]: rest, others[1]: rest})
        for i in range(start, start + n)
    ]


def _probs(cls, p):
    others = [c for c in ("fund.fees", "fund.risks", "fund.other") if c != cls]
    return {cls: p, others[0]: (1 - p) / 2, others[1]: (1 - p) / 2}


def test_keep_rule_core_or_prob_override():
    # fund.fees: 11 elements at 0.50, 0.55, ..., 1.00 -> floor(10% of 11) = 1 may bypass
    fees = [_row("d", f"fees-{i}", f"f{i}", _probs("fund.fees", 0.5 + 0.05 * i)) for i in range(11)]
    # fund.risks: a less sure class, 0.40 ... 0.60 -> its top one, at 0.60, well below 0.8
    risks = [
        _row("d", f"risks-{i}", f"r{i}", _probs("fund.risks", 0.4 + 0.02 * i)) for i in range(11)
    ]
    cores = {row.key: (["fund.other"], 0.2) for row in fees + risks}  # nobody in core...
    cores[fees[0].key] = (["fund.fees", "fund.risks"], 0.3)  # ...but the least sure fees element
    rows = fees + risks

    ts = confident_set(rows, cores, keep=1.0, min_per_class=1, taxonomy=_taxonomy())

    kept = sorted(e.element_id for e in ts.examples)
    # in core whatever its probability; else only the class's top decile, rounded down
    assert kept == ["fees-0", "fees-10", "risks-10"]
    report = ts.classes["fund.fees"]
    assert (report.labelled, report.confident, report.kept, report.trained) == (11, 2, 2, True)
    assert report.override_prob == pytest.approx(1.0)  # the last element admitted
    assert ts.classes["fund.risks"].override_prob == pytest.approx(0.6)
    assert ts.classes["fund.other"].override_prob is None
    assert ts.params["prob_override_quantile"] == 0.9
    in_core = next(e for e in ts.examples if e.element_id == "fees-0")
    assert in_core.in_core and in_core.path == ["fund", "fund.fees"]
    assert not next(e for e in ts.examples if e.element_id == "fees-10").in_core
    # the quantile is a parameter: at 0.5, floor(5.5) = 5 per class bypass the gap check
    half = confident_set(
        rows, cores, keep=1.0, min_per_class=1, prob_override_quantile=0.5, taxonomy=_taxonomy()
    )
    assert len(half.examples) == 1 + 5 + 5
    assert half.classes["fund.fees"].override_prob == pytest.approx(0.8)
    # fewer than 10 elements: floor(10% of n) = 0, nobody bypasses
    few = confident_set(fees[1:9], cores, keep=1.0, min_per_class=1, taxonomy=_taxonomy())
    assert few.examples == [] and few.classes["fund.fees"].override_prob is None
    # rows without a core (another split, or never scored) are not candidates
    assert len(confident_set(rows, {}, taxonomy=_taxonomy()).examples) == 0


def test_overconfident_teacher_does_not_bypass_the_gap():
    # every teacher probability >= 0.95: an absolute 0.8 override would keep all 40
    in_core = [
        _row("d", f"core-{i}", f"c{i}", _probs("fund.fees", 0.97 + 0.001 * i)) for i in range(20)
    ]
    off_core = [
        _row("d", f"off-{i}", f"o{i}", _probs("fund.fees", 0.95 + 0.001 * i)) for i in range(20)
    ]
    cores = {row.key: (["fund.fees"], 0.3) for row in in_core}
    cores.update({row.key: (["fund.risks"], 0.3) for row in off_core})

    ts = confident_set(in_core + off_core, cores, keep=1.0, taxonomy=_taxonomy())

    assert {e.element_id for e in ts.examples} == {f"core-{i}" for i in range(20)}
    report = ts.classes["fund.fees"]
    assert (report.labelled, report.confident) == (40, 20)
    assert report.override_prob == pytest.approx(0.986)  # 4 admitted, all already in core

    # an off-core element in the class's top decile (4 of 45) still gets through
    top = [
        _row("d", f"top-{i}", f"t{i}", _probs("fund.fees", 0.9995 + 0.0001 * i)) for i in range(5)
    ]
    cores.update({row.key: (["fund.risks"], 0.3) for row in top})
    ts = confident_set(in_core + off_core + top, cores, keep=1.0, taxonomy=_taxonomy())
    kept = {e.element_id for e in ts.examples}
    assert {"top-1", "top-2", "top-3", "top-4"} <= kept and "top-0" not in kept
    assert not any(e.element_id.startswith("off-") for e in ts.examples)


def test_saturated_teacher_ties_admit_exactly_the_top_decile():
    # every probability is 1.0 and nothing is in core: exactly floor(10% of 30) = 3 bypass,
    # the first three by element key
    rows = [_row("d", f"x-{i:02d}", f"x{i}", {"fund.fees": 1.0}) for i in reversed(range(30))]
    cores = {row.key: (["fund.risks"], 0.3) for row in rows}

    ts = confident_set(rows, cores, keep=1.0, min_per_class=1, taxonomy=_taxonomy())

    assert sorted(e.element_id for e in ts.examples) == ["x-00", "x-01", "x-02"]
    report = ts.classes["fund.fees"]
    assert (report.labelled, report.confident, report.override_prob) == (30, 3, 1.0)


def test_top75_per_class_with_floor():
    a = _labelled("fund.fees", 40, prob=0.9)
    b = _labelled("fund.risks", 22, prob=0.9)
    gaps = {row.key: 0.01 * (i + 1) for i, row in enumerate(a + b)}
    cores = {key: (["fund.fees", "fund.risks"], gap) for key, gap in gaps.items()}

    ts = confident_set(a + b, cores, taxonomy=_taxonomy())

    kept_a = [e for e in ts.examples if e.label == "fund.fees"]
    kept_b = [e for e in ts.examples if e.label == "fund.risks"]
    assert len(kept_a) == 30  # ceil(0.75 · 40)
    assert len(kept_b) == 20  # ceil(0.75 · 22) = 17, raised to the floor of 20
    assert {e.element_id for e in kept_a} == {f"fund.fees-{i}" for i in range(10, 40)}
    assert {e.element_id for e in kept_b} == {f"fund.risks-{i}" for i in range(2, 22)}
    top = max(ts.examples, key=lambda e: e.gap)
    assert top.confidence == pytest.approx(0.9 * 1.0)  # teacher prob × gap rank percentile
    low = min(kept_a, key=lambda e: e.gap)  # 11th smallest of 62 gaps
    assert low.confidence == pytest.approx(0.9 * 11 / 62)
    assert ts.untrained == ["fund.other", "paper.method", "paper.other"]
    assert ts.classes["fund.fees"].kept == 30 and ts.classes["fund.risks"].kept == 20


def test_sparse_class_reported_untrained(caplog):
    a = _labelled("fund.fees", 25, prob=0.9)
    b = _labelled("fund.risks", 19, prob=0.9)
    b_weak = _labelled(
        "fund.risks", 5, prob=0.5, start=19
    )  # not in core, below the class's 0.9 quantile: dropped
    rows = a + b + b_weak
    cores = {row.key: (["fund.fees", "fund.risks"], 0.1) for row in a + b}
    cores.update({row.key: (["fund.other"], 0.1) for row in b_weak})

    with caplog.at_level("WARNING", logger="docrooter.teach"):
        ts = confident_set(rows, cores, taxonomy=_taxonomy())

    assert "fund.risks" in ts.untrained and "fund.fees" not in ts.untrained
    report = ts.classes["fund.risks"]
    assert (report.labelled, report.confident, report.trained) == (24, 19, False)
    sparse = [e for e in ts.examples if e.label == "fund.risks"]
    assert len(sparse) == 19 and all(e.path == ["fund"] for e in sparse)  # the family only
    assert all(e.path == ["fund", "fund.fees"] for e in ts.examples if e.label == "fund.fees")
    assert ts.classes["paper.method"].labelled == 0 and "paper.method" in ts.untrained
    assert "fund.risks" in caplog.text

    empty = confident_set([], {}, taxonomy=_taxonomy())  # nothing at all: no crash
    assert empty.examples == [] and set(empty.untrained) == {n.id for n in _taxonomy().leaves()}
    assert TrainingSet.model_validate_json(ts.model_dump_json()) == ts


# --- the dataset -------------------------------------------------------------------------------


def test_dataset_text_format(tmp_path):
    items = [
        ("Prices may fall.", "paragraph", ["Item 1A", "Market risk"]),
        ("Preamble text.", "paragraph", []),
        ("Fee 1%", "table", ["Fees"]),
        ("Not kept.", "paragraph", ["Fees"]),
    ]
    corpus = _corpus(tmp_path, {"doc1": items})
    table = next(e for e in corpus.iter_elements() if e.kind == "table")
    table_md = "| Fee |\n| --- |\n| 1% |"
    stored = list(corpus.iter_elements())
    stored[2] = table.model_copy(update={"table_markdown": table_md})

    assert input_text(stored[0]) == "Item 1A > Market risk\n\n[paragraph] Prices may fall."
    assert input_text(stored[1]) == "\n\n[paragraph] Preamble text."
    assert input_text(stored[2]) == f"Fees\n\n[table] {table_md}"
    assert input_text(table) == "Fees\n\n[table] Fee 1%"  # no Markdown: the text

    rows = [
        _row("doc1", "e0", items[0][0], {"fund.risks": 0.9, "fund.fees": 0.1}),
        _row("doc1", "e1", items[1][0], {"fund.other": 0.95, "fund.fees": 0.05}),
    ]
    cores = {rows[0].key: (["fund.risks"], 0.5), rows[1].key: (["fund.other"], 0.4)}
    ts = confident_set(rows, cores, min_per_class=1, taxonomy=_taxonomy())

    dataset = to_dataset(ts, corpus)

    assert [(e.text, e.path, e.doc_id, e.section_id) for e in dataset] == [
        (
            "Item 1A > Market risk\n\n[paragraph] Prices may fall.",
            ["fund", "fund.risks"],
            "doc1",
            "e0",
        ),
        ("\n\n[paragraph] Preamble text.", ["fund", "fund.other"], "doc1", "e1"),
    ]
    assert dataset.resolve(_taxonomy()).examples[0].path == ["fund", "fund.risks"]

    stale = confident_set(
        [_row("doc1", "e0", "changed text", {"fund.risks": 0.9})],
        {_row("doc1", "e0", "changed text", {}).key: (["fund.risks"], 0.5)},
        min_per_class=1,
        taxonomy=_taxonomy(),
    )
    with pytest.raises(ValueError, match="not in the corpus"):
        to_dataset(stale, corpus)


def test_build_training_set_end_to_end(tmp_path):
    corpus, labels, items = _fund_corpus(tmp_path)
    embedder = FakeEmbedder()

    enriched, ts = build_training_set(corpus, labels, _taxonomy(), embedder, k=3, min_per_class=3)

    assert enriched.nodes["fund.fees"].keywords  # enriched before the centroids
    first_element_call = next(
        i for i, (texts, _) in enumerate(embedder.calls) if any("[paragraph]" in t for t in texts)
    )
    assert all(
        "[paragraph]" not in t for texts, _ in embedder.calls[:first_element_call] for t in texts
    )
    assert ts.classes["fund.fees"].labelled == 8 and ts.classes["fund.fees"].trained
    assert ts.classes["fund.risks"].trained
    fees = [e for e in ts.examples if e.label == "fund.fees"]
    assert fees and all(e.in_core for e in fees)  # fees elements sit near the fees centroid
    assert len(fees) == 6  # ceil(0.75 · 8)
    dataset = to_dataset(ts, corpus)
    assert len(dataset) == len(ts.examples)
    assert all(
        e.text.startswith(("Fees\n\n[paragraph] ", "Risks\n\n", "Contact\n\n")) for e in dataset
    )


def test_element_ngrams_match_keywords():
    grams = element_ngrams("The Management Fees, and expenses.")
    assert {"management fees", "the management fees", "expenses", "and expenses"} <= grams
    assert "fees and" not in grams  # never across punctuation
