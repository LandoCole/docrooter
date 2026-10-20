"""Section gold tooling (v1 V5): the section sampler, the annotator view, label validation
against the section catalogue, Cohen's kappa and the adjudication merge. Documents are tiny
in-memory fixtures: no PDF is read and no model is loaded."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from benchmarks.classification import section_gold, section_sample
from benchmarks.training_corpus.gold import AnnotationError
from docrooter.document import Document

SECRET = "TEXT-OF-A-NEVER-REDISTRIBUTED-DOCUMENT"
LEAVES = list(section_gold.catalogue_leaves())


def _doc(n_top: int, *, subs: int = 2) -> Document:
    """``n_top`` level-1 sections, each with ``subs`` level-2 ones, and a level-3 one."""
    entries: list[dict] = [{"title": "Preamble", "level": 0, "text": "front"}]
    for i in range(n_top):
        entries.append({"title": f"Part {i}", "level": 1, "text": f"{SECRET} part {i}"})
        for j in range(subs):
            entries.append({"title": f"Part {i}.{j}", "level": 2, "text": f"{SECRET} {i}.{j}"})
            entries.append({"title": f"Deep {i}.{j}", "level": 3, "text": "deep"})
    return Document.from_sections(entries, source="<fixture>")


def test_sample_deterministic_and_capped() -> None:
    doc = _doc(10, subs=3)  # 10 level-1 and 30 level-2 candidates
    first = section_sample.sample_sections(doc, max_per_doc=12)
    again = section_sample.sample_sections(_doc(10, subs=3), max_per_doc=12)
    assert [s.id for s in first] == [s.id for s in again]
    assert len(first) == 12
    assert {s.level for s in first} <= {1, 2}
    order = [s.id for s in doc.walk()]
    positions = [order.index(s.id) for s in first]
    assert positions == sorted(positions)
    assert len(set(positions)) == 12
    # spread: picks come from the whole document, not its start
    assert positions[0] < len(order) // 4 and positions[-1] > 3 * len(order) // 4
    # a small document is taken whole; the cap is the argument
    small = _doc(2, subs=1)
    assert len(section_sample.sample_sections(small, max_per_doc=12)) == 4
    assert len(section_sample.sample_sections(doc, max_per_doc=5)) == 5
    with pytest.raises(ValueError):
        section_sample.sample_sections(doc, max_per_doc=0)


def _sample(n: int = 10) -> list[dict]:
    rows = [
        {
            "doc_id": f"d{k // 5}",
            "section_id": f"s{k}",
            "title_sha1": section_sample.title_sha1(f"title {k}"),
            "level": 1 + k % 2,
            "title": f"title {k}",
            "heading_path": ["Part"] if k % 2 else [],
            "text": f"{SECRET} {k}",
        }
        for k in range(n)
    ]
    return section_sample.assign_items(rows, double_fraction=0.4)


def _labels(sample, who: str, labels: dict[str, str]) -> dict:
    rows = [{"item": i, "label": lab, "annotator": who} for i, lab in labels.items()]
    return section_gold.read_annotations(rows, sample)


def test_kappa_and_merge() -> None:
    sample = _sample(10)
    double = [r["item"] for r in sample if r["double"]]
    assert len(double) == 4
    a, b, c = "general.body", "general.summary", "research.method"
    first = _labels(sample, "A1", {r["item"]: a for r in sample})
    # B1 agrees on two double items, disagrees on two (one within the group, one across)
    second_labels = {double[0]: a, double[1]: a, double[2]: b, double[3]: c}
    second = _labels(sample, "B1", second_labels)

    assert section_gold.disagreements(sample, first, second) == double[2:]
    report = section_gold.agreement(sample, first, second)
    assert report["double_annotated"] == 4 and report["disagreements"] == 2
    assert report["leaf"]["n"] == 4 and report["leaf"]["observed_agreement"] == 0.5
    # A1 used one label throughout: chance agreement is 0.5*... computed by hand below
    # po = 0.5; A1 marginals {a: 4}, B1 {a: 2, b: 1, c: 1}; pe = 4*2/16 = 0.5; kappa = 0
    assert report["leaf"]["kappa"] == pytest.approx(0.0)
    assert report["group"]["observed_agreement"] == 0.75  # b is in a's group, c is not

    with pytest.raises(AnnotationError, match="no adjudication"):
        section_gold.merge_gold(sample, first, second, {})
    own = _labels(sample, "A1", {double[2]: b, double[3]: c})
    with pytest.raises(AnnotationError, match="not a third annotator"):
        section_gold.merge_gold(sample, first, second, own)

    adjudication = _labels(sample, "C1", {double[2]: b, double[3]: a})
    rows = section_gold.merge_gold(sample, first, second, adjudication)
    assert [r["label"] for r in rows if r["adjudicated"]] == [b, a]
    by_key = {(r["doc_id"], r["section_id"]): r for r in rows}
    sec = {r["item"]: (r["doc_id"], r["section_id"]) for r in sample}
    assert by_key[sec[double[0]]]["annotator"] == "A1+B1"
    assert by_key[sec[double[2]]]["annotator"] == "C1"
    single = next(r["item"] for r in sample if not r["double"])
    assert by_key[sec[single]]["annotator"] == "A1" and not by_key[sec[single]]["adjudicated"]
    assert len(rows) == 10 and section_gold.gold_row_errors(rows, sample) == []


def test_merge_rejects_bad_roles_and_labels() -> None:
    sample = _sample(10)
    every = {r["item"]: "general.body" for r in sample}
    first = _labels(sample, "A1", every)
    double = {r["item"]: "general.body" for r in sample if r["double"]}
    with pytest.raises(AnnotationError, match="not independent"):
        section_gold.agreement(sample, first, _labels(sample, "A1", double))
    with pytest.raises(AnnotationError, match="missing"):
        section_gold.agreement(sample, first, _labels(sample, "B1", {}))
    for bad, text in (("general", "is a group"), ("general.nope", "unknown"), (3, "string")):
        rows = [{"item": "s0000", "label": bad, "annotator": "A1"}]
        with pytest.raises(AnnotationError, match=text):
            section_gold.read_annotations(rows, sample)
    with pytest.raises(AnnotationError, match="not an item"):
        section_gold.read_annotations(
            [{"item": "zz", "label": "general.body", "annotator": "x"}], sample
        )
    with pytest.raises(AnnotationError, match="annotated twice"):
        section_gold.read_annotations(
            [{"item": "s0000", "label": "general.body", "annotator": "x"}] * 2, sample
        )


def test_gold_rows_hold_no_text(tmp_path: Path) -> None:
    sample = _sample(10)
    first = _labels(sample, "A1", {r["item"]: "general.body" for r in sample})
    second = _labels(sample, "B1", {r["item"]: "general.body" for r in sample if r["double"]})
    rows = section_gold.merge_gold(sample, first, second, {})
    assert all(tuple(r) == section_gold.GOLD_FIELDS for r in rows)
    assert not {"text", "title", "heading_path"} & set(rows[0])
    blob = json.dumps(rows)
    assert SECRET not in blob and "title 3" not in blob
    # a row that carries text is refused
    leaky = [{**rows[0], "text": SECRET}, *rows[1:]]
    assert any("unexpected" in e for e in section_gold.gold_row_errors(leaky, sample))

    # the command line writes the same: gold without text, agreement with the reading
    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text("".join(json.dumps(r) + "\n" for r in sample), encoding="utf-8")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"reading": section_sample.reading_metadata()}), encoding="utf-8")
    for name, who, items in (
        ("first", "A1", [r["item"] for r in sample]),
        ("second", "B1", [r["item"] for r in sample if r["double"]]),
    ):
        text = "".join(
            json.dumps({"item": i, "label": "general.body", "annotator": who}) + "\n" for i in items
        )
        (tmp_path / f"{name}.jsonl").write_text(text, encoding="utf-8")
    (tmp_path / "adj.jsonl").write_text("", encoding="utf-8")
    gold, agreement = tmp_path / "gold.jsonl", tmp_path / "agreement.json"
    argv = ["merge", "--sample", str(sample_path), "--sample-summary", str(summary)]
    argv += ["--first", str(tmp_path / "first.jsonl"), "--second", str(tmp_path / "second.jsonl")]
    argv += ["--adjudication", str(tmp_path / "adj.jsonl"), "--gold", str(gold)]
    argv += ["--agreement", str(agreement)]
    assert section_gold.main(argv) == 0
    written = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 10 and all(tuple(r) == section_gold.GOLD_FIELDS for r in written)
    assert SECRET not in gold.read_text(encoding="utf-8")
    meta = json.loads(agreement.read_text(encoding="utf-8"))
    assert meta["reading"]["profile"] == "fast" and meta["reading"]["elements"] is False
    assert SECRET not in agreement.read_text(encoding="utf-8")


def test_annotator_view_hides_document_and_batches_stay_in_cache(tmp_path: Path) -> None:
    sample = _sample(10)
    view = section_gold.annotator_view(sample[1])
    assert tuple(view) == section_gold.ANNOTATOR_FIELDS
    assert not {"doc_id", "section_id", "title_sha1", "double"} & set(view)

    sample_path = tmp_path / "sample.jsonl"
    sample_path.write_text("".join(json.dumps(r) + "\n" for r in sample), encoding="utf-8")
    out = tmp_path / "batches"
    assert section_gold.main(["batches", "--sample", str(sample_path), "--out-dir", str(out)]) == 0
    assert len(list((out / "first").glob("*.jsonl"))) == 1
    second = [
        json.loads(line)
        for f in (out / "second").glob("*.jsonl")
        for line in f.read_text(encoding="utf-8").splitlines()
    ]
    assert len(second) == sum(r["double"] for r in sample)
    catalogue = json.loads((out / "catalogue.json").read_text(encoding="utf-8"))
    assert sum(len(g["leaves"]) for g in catalogue) == len(LEAVES) == 112

    tracked = section_sample.HERE / "annotation"
    with pytest.raises(ValueError, match="git-ignored"):
        section_sample.require_untracked(tracked)
    section_sample.require_untracked(section_sample.CACHE_DIR / "annotation")
    section_sample.require_untracked(tmp_path)


def test_build_sample_records_the_fast_reading() -> None:
    docs = [("d2", _doc(3)), ("d1", _doc(20, subs=1)), ("d0", Document.from_sections([]))]
    rows, summary = section_sample.build_sample(docs, max_per_doc=6, double_fraction=0.25)
    assert [r["doc_id"] for r in rows[:6]] == ["d1"] * 6  # documents in id order
    assert summary["documents_without_sections"] == ["d0"]
    assert summary["selected"] == len(rows) == 6 + 6 and summary["double_annotated"] == 3
    assert summary["reading"]["profile"] == "fast" and summary["reading"]["elements"] is False
    assert summary["reading"]["settings"]["layout"] == "none"
    assert all(r["title_sha1"] == section_sample.title_sha1(r["title"]) for r in rows)
    top = next(r for r in rows if r["level"] == 1)
    assert top["heading_path"] == [] and top["text"].startswith(SECRET)
    deep = section_sample.build_sample([("d", _doc(1))], max_per_doc=12)[0]
    child = next(r for r in deep if r["title"] == "Part 0.1")
    assert child["heading_path"] == ["Part 0"] and child["level"] == 2
    assert len(section_sample.opening(_doc(1).section("Part 0"), 10)) == 10


def test_guide_names_only_catalogue_labels() -> None:
    guide = section_gold.GUIDE_PATH.read_text(encoding="utf-8")
    groups = set(section_gold.catalogue_leaves().values())
    named = set(re.findall(r"`([a-z_]+(?:\.[a-z_]+)?)`", guide))
    ids = {n for n in named if "." in n} - {"group.leaf", "catalogue.json"}
    assert ids <= set(LEAVES), sorted(ids - set(LEAVES))
    assert groups <= named, sorted(groups - named)  # every group is named
    # the other leaves are explained: every group's `other` is listed by id
    assert {leaf for leaf in LEAVES if leaf.endswith(".other")} <= ids
    assert len(section_gold.catalogue_leaves()) == 112 and len(groups) == 12
