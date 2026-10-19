"""Gold set (v0.5 L5): the stratified sampler over gold-split documents, the annotator view,
label validation against the element taxonomies, Cohen's kappa and the adjudication merge.
Every corpus here is a synthetic store built in a temporary directory; no model is loaded."""

from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from benchmarks.training_corpus import gold, gold_sample
from docrooter.corpus import Corpus, CorpusElement, ManifestRecord

KINDS = ("paragraph", "paragraph", "paragraph", "list_item", "table", "heading", "caption")

# source, doc_type_hint, family_hint for each kind of synthetic document
SOURCES = {
    "ir": ("ir_annual_reports", "financial.annual_report", "annual_report"),
    "fund": ("fund_docs", "financial.prospectus", "fund_document"),
    "arxiv": ("arxiv", "research.research_paper", "research_paper"),
    "gao": ("govinfo_gao", "government.government_report", "generic"),
    "court": ("uscourts", "legal.court_opinion", "generic"),
    "cuad": ("cuad", "legal.contract", "contract"),
    "rfc": ("rfc", "research.technical_standard", "generic"),
}


def _corpus(tmp_path, plan, *, per_doc=40):
    """A store holding, for each ``(prefix, split, count)`` in ``plan``, ``count`` documents of
    that source and split with ``per_doc`` elements each (kinds cycling through KINDS)."""
    corpus = Corpus(tmp_path / "store")
    n = 0
    for prefix, split, count in plan:
        source, doc_type, family = SOURCES[prefix]
        for k in range(count):
            n += 1
            doc_id = f"{prefix}-{split}-{k:02d}"
            record = ManifestRecord(
                id=doc_id,
                url=f"https://example.org/{doc_id}.pdf",
                source=source,
                sha256=f"{n:064x}",
                doc_type_hint=doc_type,
                family_hint=family,
                split=split,
            )
            elements = [
                CorpusElement(
                    doc_id=doc_id,
                    element_id=f"e{j}",
                    kind=KINDS[j % len(KINDS)],
                    text=f"{doc_id} element {j} text",
                    heading_path=["Part I", f"Section {j // 10}"],
                    section_title=f"Section {j // 10}",
                    page=1 + j // 8,
                )
                for j in range(per_doc)
            ]
            corpus.add_document(record, elements, range(1, 6))
    return corpus


FULL_PLAN = [
    ("ir", "gold", 12),
    ("fund", "gold", 8),
    ("arxiv", "gold", 8),
    ("gao", "gold", 4),
    ("court", "gold", 4),
    ("cuad", "gold", 6),
    ("rfc", "gold", 6),
    # documents of the other splits, which the sampler must never use
    ("ir", "train", 10),
    ("rfc", "train", 10),
    ("cuad", "calib", 3),
]


def test_gold_sample_stratified_per_spec(tmp_path):
    corpus = _corpus(tmp_path, FULL_PLAN)
    sample = gold_sample.sample_gold(corpus)
    rows = sample.rows

    assert len(rows) == 900
    by_group = Counter(r["group"] for r in rows)
    assert by_group == {
        "annual_report": 300,
        "fund_document": 150,
        "research_paper": 150,
        "government_legal": 150,
        "technical": 150,
    }
    gov = Counter(r["family"] for r in rows if r["group"] == "government_legal")
    assert gov == {"generic": 100, "contract": 50}
    assert {r["family"] for r in rows if r["group"] == "technical"} == {"generic"}

    # gold-split documents only
    gold_ids = {rec.id for rec in corpus.manifest() if rec.split == "gold"}
    assert {r["doc_id"] for r in rows} <= gold_ids

    # one row per element, keyed and hashed like the corpus labels; items are unique
    stored = {(e.doc_id, e.element_id): e for e in corpus.iter_elements(split="gold")}
    assert len({(r["doc_id"], r["element_id"]) for r in rows}) == 900
    assert len({r["item"] for r in rows}) == 900
    for r in rows:
        element = stored[(r["doc_id"], r["element_id"])]
        assert r["text_sha1"] == element.text_sha1
        assert (r["kind"], r["text"], r["heading_path"]) == (
            element.kind,
            element.text,
            element.heading_path,
        )

    # no document dominates its cell: at most max_doc_share of the cell's quota
    for cell in gold_sample.CELLS:
        per_doc = Counter(r["doc_id"] for r in rows if gold_sample.cell_of_row(r) == cell.name)
        assert max(per_doc.values()) <= int(cell.quota * gold_sample.MAX_DOC_SHARE)

    # stratified by kind: each cell carries every kind its documents have, roughly in proportion
    for cell in gold_sample.CELLS:
        kinds = Counter(r["kind"] for r in rows if gold_sample.cell_of_row(r) == cell.name)
        assert set(kinds) == set(KINDS)
        share = kinds["paragraph"] / sum(kinds.values())
        assert 0.3 < share < 0.55  # 3/7 of the pool are paragraphs

    # 20% double-annotated, stratified by cell
    double = Counter(r["group"] for r in rows if r["double"])
    assert sum(double.values()) == 180
    assert double == {
        "annual_report": 60,
        "fund_document": 30,
        "research_paper": 30,
        "government_legal": 30,
        "technical": 30,
    }

    summary = sample.summary
    assert summary["selected"] == 900 and summary["shortfall"] == 0
    assert summary["seed"] == 0

    # deterministic with seed 0
    again = gold_sample.sample_gold(Corpus(corpus.dir))
    assert again.rows == rows


def test_gold_sample_reports_shortfall_without_borrowing(tmp_path):
    plan = [
        ("ir", "gold", 12),
        ("fund", "gold", 8),
        ("arxiv", "gold", 8),
        ("gao", "gold", 8),
        ("cuad", "gold", 2),  # 2 docs x cap 10 = 20 of 50 contract elements
        ("rfc", "gold", 1),  # 1 doc x cap 30 = 30 of 150
        ("cuad", "train", 10),
        ("rfc", "train", 10),
    ]
    corpus = _corpus(tmp_path, plan)
    sample = gold_sample.sample_gold(corpus)
    cells = sample.summary["cells"]
    assert cells["government_legal.contract"]["selected"] == 20
    assert cells["government_legal.contract"]["shortfall"] == 30
    assert cells["technical"]["selected"] == 30
    assert cells["technical"]["shortfall"] == 120
    assert sample.summary["shortfall"] == 150
    assert len(sample.rows) == 750
    assert all(r["doc_id"].split("-")[1] == "gold" for r in sample.rows)


def test_gold_sample_skips_blank_and_duplicate_text(tmp_path):
    corpus = Corpus(tmp_path / "store")
    record = ManifestRecord(
        id="ir-gold-00",
        source="ir_annual_reports",
        sha256="1" * 64,
        doc_type_hint="financial.annual_report",
        family_hint="annual_report",
        split="gold",
    )
    texts = ["  ", "Same line", "Same line", "Unique"]
    elements = [
        CorpusElement(
            doc_id=record.id,
            element_id=f"e{j}",
            kind="paragraph",
            text=t,
            heading_path=[],
            section_title=None,
            page=1,
        )
        for j, t in enumerate(texts)
    ]
    corpus.add_document(record, elements, [1])
    sample = gold_sample.sample_gold(corpus)
    assert sorted(r["text"] for r in sample.rows) == ["Same line", "Unique"]


def test_sample_files_and_annotator_view(tmp_path):
    corpus = _corpus(tmp_path, FULL_PLAN)
    out = tmp_path / "gold_sample.jsonl"
    summary_path = tmp_path / "gold_sample_summary.json"
    code = gold_sample.main(
        ["--corpus", str(corpus.dir), "--out", str(out), "--summary", str(summary_path)]
    )
    assert code == 0
    rows = gold.read_sample(out)
    assert len(rows) == 900
    assert json.loads(summary_path.read_text())["selected"] == 900

    view = gold.annotator_view(rows[0])
    assert set(view) == {"item", "kind", "family", "heading_path", "text"}

    batches = tmp_path / "batches"
    assert gold.main(["batches", "--sample", str(out), "--out-dir", str(batches)]) == 0
    first = sorted((batches / "first").glob("*.jsonl"))
    second = sorted((batches / "second").glob("*.jsonl"))
    assert len(first) == 18 and len(second) == 4  # 900 / 50 and 180 / 50
    items = [json.loads(line) for path in first for line in path.read_text().splitlines()]
    assert len(items) == 900 and all(set(i) == set(view) for i in items)
    seconds = [json.loads(line) for path in second for line in path.read_text().splitlines()]
    assert {i["item"] for i in seconds} == {r["item"] for r in rows if r["double"]}


# --- labels ----------------------------------------------------------------------------------


def _sample_rows():
    families = ["annual_report", "contract", "generic", "fund_document"]
    return [
        {
            "item": f"g{i:04d}",
            "doc_id": f"doc-{i % 3}",
            "element_id": f"e{i}",
            "text_sha1": f"{i:040x}",
            "kind": "paragraph",
            "text": f"text {i}",
            "heading_path": [],
            "family": families[i % 4],
            "group": "x",
            "double": i < 4,
        }
        for i in range(8)
    ]


def test_gold_rows_valid_against_taxonomies():
    leaves = gold.topic_leaves()
    assert set(leaves) == {
        "annual_report",
        "fund_document",
        "contract",
        "research_paper",
        "technical_document",
        "generic",
    }
    assert all(f"{family}.other" in ids for family, ids in leaves.items())
    assert len(gold.function_tags()) == 12

    assert gold.label_errors("contract", "contract.payment", ["obligation", "financial_data"]) == []
    assert gold.label_errors("contract", "contract.other", []) == []
    assert gold.label_errors("contract", "annual_report.risk_factors", [])  # wrong family
    assert gold.label_errors("contract", "contract.nope", [])  # unknown leaf
    assert gold.label_errors("contract", "contract", [])  # the family node is not a leaf
    assert gold.label_errors("contract", "contract.payment", ["nope"])  # unknown function
    assert gold.label_errors("contract", "contract.payment", ["definition", "definition"])
    assert gold.label_errors("nope", "nope.other", [])  # unknown family

    sample = _sample_rows()
    first, second, adjudication = _annotations(sample)
    rows = gold.merge_gold(sample, first, second, adjudication)
    assert gold.gold_row_errors(rows, sample) == []

    bad = [dict(r) for r in rows]
    bad[0]["topic"] = "research_paper.abstract"  # outside the element's family
    bad[1]["functions"] = ["marketing", "bogus"]
    bad[2]["adjudicated"] = "yes"
    del bad[3]["annotator"]
    bad.append(dict(rows[4]))  # duplicate key
    errors = gold.gold_row_errors(bad, sample)
    assert len(errors) == 5
    assert gold.gold_row_errors(rows[:-1], sample)  # an element left out


def test_read_annotations_rejects_invalid_rows():
    sample = _sample_rows()
    rows = [
        {"item": "g0000", "topic": "annual_report.other", "functions": [], "annotator": "a1"},
        {"item": "g0001", "topic": "contract.term", "functions": ["obligation"], "annotator": "a1"},
    ]
    labels = gold.read_annotations(rows, sample)
    assert labels["g0001"] == gold.Label("contract.term", ("obligation",), "a1")
    for broken in (
        {"item": "g9999", "topic": "contract.term", "functions": [], "annotator": "a1"},
        {"item": "g0001", "topic": "generic.body", "functions": [], "annotator": "a1"},
        {"item": "g0001", "topic": "contract.term", "functions": "obligation", "annotator": "a1"},
        {"item": "g0001", "topic": "contract.term", "functions": []},
    ):
        with pytest.raises(gold.AnnotationError):
            gold.read_annotations([rows[0], broken], sample)
    with pytest.raises(gold.AnnotationError):
        gold.read_annotations([rows[0], rows[0]], sample)  # one item twice


# --- agreement -------------------------------------------------------------------------------


def test_kappa_computation():
    # Textbook 2x2: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.7, pe 0.5, kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert gold.cohen_kappa(a, b) == pytest.approx(0.4)
    assert gold.cohen_kappa(["x", "y", "z"], ["x", "y", "z"]) == pytest.approx(1.0)
    # three categories: po 3/6, pa = pb = (1/3, 1/3, 1/3) -> pe 1/3, kappa (1/2 - 1/3) / (2/3)
    assert gold.cohen_kappa(list("aabbcc"), list("abbcca")) == pytest.approx(0.25)
    # systematic disagreement is below chance
    assert gold.cohen_kappa(["y", "n"], ["n", "y"]) == pytest.approx(-1.0)
    # one category used throughout by both: chance agreement is 1, kappa undefined
    assert gold.cohen_kappa(["n"] * 5, ["n"] * 5) is None
    assert gold.cohen_kappa([], []) is None
    with pytest.raises(ValueError):
        gold.cohen_kappa(["a"], ["a", "b"])

    sample = _sample_rows()
    first, second, _ = _annotations(sample)
    report = gold.agreement(sample, first, second)
    assert report["double_annotated"] == 4
    topic = report["topic"]
    assert topic["n"] == 4
    assert topic["observed_agreement"] == pytest.approx(0.75)
    a_topics = [first[f"g{i:04d}"].topic for i in range(4)]
    b_topics = [second[f"g{i:04d}"].topic for i in range(4)]
    assert topic["kappa"] == pytest.approx(gold.cohen_kappa(a_topics, b_topics))
    functions = report["functions"]
    assert set(functions["per_tag"]) == set(gold.function_tags())
    oblig = functions["per_tag"]["obligation"]
    a_ob = ["obligation" in first[f"g{i:04d}"].functions for i in range(4)]
    b_ob = ["obligation" in second[f"g{i:04d}"].functions for i in range(4)]
    assert oblig["kappa"] == pytest.approx(gold.cohen_kappa(a_ob, b_ob))
    assert (oblig["positives_first"], oblig["positives_second"]) == (sum(a_ob), sum(b_ob))
    assert functions["per_tag"]["signature"]["kappa"] is None  # never used by either
    assert report["disagreements"] == 2


# --- merge -----------------------------------------------------------------------------------


def _annotations(sample):
    """First annotator on every item; second on the 4 double items (g0000-g0003), agreeing on
    g0000 and g0001 and disagreeing on g0002 (topic) and g0003 (functions); an adjudicator
    for the two disagreements."""
    topics = {
        "annual_report": "annual_report.risk_factors",
        "contract": "contract.payment",
        "generic": "generic.body",
        "fund_document": "fund_document.performance",
    }
    first_rows = [
        {
            "item": r["item"],
            "topic": topics[r["family"]],
            "functions": ["obligation"] if r["family"] == "contract" else [],
            "annotator": "first-1" if i < 4 else "first-2",
        }
        for i, r in enumerate(sample)
    ]
    second_rows = [dict(row, annotator="second-1") for row in first_rows[:4]]
    second_rows[2]["topic"] = "generic.conclusion"
    second_rows[3]["functions"] = ["financial_data"]
    adjudication_rows = [
        {"item": "g0002", "topic": "generic.body", "functions": [], "annotator": "adj-1"},
        {
            "item": "g0003",
            "topic": "fund_document.performance",
            "functions": ["financial_data"],
            "annotator": "adj-1",
        },
    ]
    return (
        gold.read_annotations(first_rows, sample),
        gold.read_annotations(second_rows, sample),
        gold.read_annotations(adjudication_rows, sample),
    )


def test_merge_gold_uses_adjudication_for_disagreements():
    sample = _sample_rows()
    first, second, adjudication = _annotations(sample)
    assert gold.disagreements(sample, first, second) == ["g0002", "g0003"]
    rows = gold.merge_gold(sample, first, second, adjudication)
    assert [r["element_id"] for r in rows] == [r["element_id"] for r in sample]
    assert set(rows[0]) == {
        "doc_id",
        "element_id",
        "text_sha1",
        "family",
        "topic",
        "functions",
        "annotator",
        "adjudicated",
    }
    by_id = {r["element_id"]: r for r in rows}
    assert by_id["e0"]["annotator"] == "first-1+second-1" and not by_id["e0"]["adjudicated"]
    assert by_id["e2"]["topic"] == "generic.body" and by_id["e2"]["adjudicated"] is True
    assert by_id["e2"]["annotator"] == "adj-1"
    assert by_id["e3"]["functions"] == ["financial_data"] and by_id["e3"]["adjudicated"]
    assert by_id["e5"]["annotator"] == "first-2" and by_id["e5"]["adjudicated"] is False
    assert by_id["e5"]["functions"] == ["obligation"]

    with pytest.raises(gold.AnnotationError, match="adjudication"):
        gold.merge_gold(sample, first, second, {})
    missing_first = {k: v for k, v in first.items() if k != "g0006"}
    with pytest.raises(gold.AnnotationError, match="g0006"):
        gold.merge_gold(sample, missing_first, second, adjudication)
    missing_second = {k: v for k, v in second.items() if k != "g0001"}
    with pytest.raises(gold.AnnotationError, match="g0001"):
        gold.merge_gold(sample, first, missing_second, adjudication)
    extra_second = dict(second, g0005=first["g0005"]._replace(annotator="second-1"))
    with pytest.raises(gold.AnnotationError, match="g0005"):
        gold.merge_gold(sample, first, extra_second, adjudication)
    not_third = dict(adjudication, g0002=adjudication["g0002"]._replace(annotator="first-1"))
    with pytest.raises(gold.AnnotationError, match="third"):
        gold.merge_gold(sample, first, second, not_third)
    same_annotator = dict(second, g0000=first["g0000"])
    with pytest.raises(gold.AnnotationError, match="independent"):
        gold.merge_gold(sample, first, same_annotator, adjudication)


def test_merge_cli_writes_gold_and_agreement(tmp_path):
    sample = _sample_rows()
    first, second, adjudication = _annotations(sample)

    def dump(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return str(path)

    def rows_of(labels):
        return [
            {"item": k, "topic": v.topic, "functions": list(v.functions), "annotator": v.annotator}
            for k, v in labels.items()
        ]

    sample_path = dump(tmp_path / "sample.jsonl", sample)
    first_a = dump(tmp_path / "first_a.jsonl", rows_of(first)[:5])
    first_b = dump(tmp_path / "first_b.jsonl", rows_of(first)[5:])
    second_path = dump(tmp_path / "second.jsonl", rows_of(second))
    adj_path = dump(tmp_path / "adj.jsonl", rows_of(adjudication))

    assert gold.main(["validate", "--sample", sample_path, first_a, first_b]) == 0
    to_adjudicate = tmp_path / "adjudicate.jsonl"
    code = gold.main(
        [
            "adjudication-batch",
            "--sample",
            sample_path,
            "--first",
            first_a,
            first_b,
            "--second",
            second_path,
            "--out",
            str(to_adjudicate),
        ]
    )
    assert code == 0
    pending = [json.loads(line) for line in to_adjudicate.read_text().splitlines()]
    assert [p["item"] for p in pending] == ["g0002", "g0003"]
    assert all(len(p["labels"]) == 2 and "annotator" not in p["labels"][0] for p in pending)

    gold_path = tmp_path / "gold.jsonl"
    agreement_path = tmp_path / "gold_agreement.json"
    code = gold.main(
        [
            "merge",
            "--sample",
            sample_path,
            "--first",
            first_a,
            first_b,
            "--second",
            second_path,
            "--adjudication",
            adj_path,
            "--gold",
            str(gold_path),
            "--agreement",
            str(agreement_path),
        ]
    )
    assert code == 0
    written = [json.loads(line) for line in gold_path.read_text().splitlines()]
    assert written == gold.merge_gold(sample, first, second, adjudication)
    report = json.loads(agreement_path.read_text())
    assert report["disclosure"] == gold.DISCLOSURE
    assert report["elements"] == 8 and report["adjudicated"] == 2

    bad = dump(tmp_path / "bad.jsonl", [{"item": "g0000", "topic": "x", "functions": []}])
    assert gold.main(["validate", "--sample", sample_path, bad]) == 1


# --- the annotation guide --------------------------------------------------------------------


def test_annotation_guide_covers_taxonomies_and_examples_are_valid():
    guide = (gold.HERE / "annotation_guide.md").read_text(encoding="utf-8")
    for ids in gold.topic_leaves().values():
        for leaf in ids:
            assert f"`{leaf}`" in guide, leaf
    for tag in gold.function_tags():
        assert f"`{tag}`" in guide, tag
    answers = [json.loads(m) for m in re.findall(r"^- answer: `(\{.*\})`$", guide, re.M)]
    assert Counter(a["family"] for a in answers) == dict.fromkeys(gold.topic_leaves(), 3)
    for answer in answers:
        assert gold.label_errors(answer["family"], answer["topic"], answer["functions"]) == []


def test_sample_defaults_to_the_git_ignored_cache():
    cache = (gold_sample.HERE.parent / ".cache").resolve()
    assert gold_sample.SAMPLE_PATH.resolve().is_relative_to(cache)
    assert gold.SAMPLE_PATH == gold_sample.SAMPLE_PATH
    ignore = (gold_sample.HERE.parent.parent / ".gitignore").read_text().splitlines()
    assert "benchmarks/.cache/" in ignore
    assert "benchmarks/training_corpus/gold_sample*.jsonl" in ignore
