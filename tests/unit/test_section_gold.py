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
    doc = _doc(10, subs=3)  # 10 level-1, 30 level-2 and 30 level-3 candidates
    first = section_sample.sample_sections(doc, max_per_doc=12)
    again = section_sample.sample_sections(_doc(10, subs=3), max_per_doc=12)
    assert [s.id for s in first] == [s.id for s in again]
    assert len(first) == 12
    assert {s.level for s in first} == {1, 2, 3}  # every depth, as classify tags them
    order = [s.id for s in doc.walk()]
    positions = [order.index(s.id) for s in first]
    assert positions == sorted(positions)
    assert len(set(positions)) == 12
    # spread: picks come from the whole document, not its start
    assert positions[0] < len(order) // 4 and positions[-1] > 3 * len(order) // 4
    # a small document is taken whole; the cap is the argument
    small = _doc(2, subs=1)
    assert len(section_sample.sample_sections(small, max_per_doc=12)) == 6  # 2 + 2 + 2 deep
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
    a, b, c = "government.findings", "government.highlights", "research.method"
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
    every = {r["item"]: "none" for r in sample}
    first = _labels(sample, "A1", every)
    double = {r["item"]: "none" for r in sample if r["double"]}
    with pytest.raises(AnnotationError, match="not independent"):
        section_gold.agreement(sample, first, _labels(sample, "A1", double))
    with pytest.raises(AnnotationError, match="missing"):
        section_gold.agreement(sample, first, _labels(sample, "B1", {}))
    for bad, text in (("financial", "is a group"), ("financial.nope", "unknown"), (3, "string")):
        rows = [{"item": "s0000", "label": bad, "annotator": "A1"}]
        with pytest.raises(AnnotationError, match=text):
            section_gold.read_annotations(rows, sample)
    with pytest.raises(AnnotationError, match="not an item"):
        section_gold.read_annotations([{"item": "zz", "label": "none", "annotator": "x"}], sample)
    with pytest.raises(AnnotationError, match="annotated twice"):
        section_gold.read_annotations(
            [{"item": "s0000", "label": "none", "annotator": "x"}] * 2, sample
        )


def test_gold_rows_hold_no_text(tmp_path: Path) -> None:
    sample = _sample(10)
    first = _labels(sample, "A1", {r["item"]: "none" for r in sample})
    second = _labels(sample, "B1", {r["item"]: "none" for r in sample if r["double"]})
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
            json.dumps({"item": i, "label": "none", "annotator": who}) + "\n" for i in items
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
    assert sum(len(g["leaves"]) for g in catalogue) == len(LEAVES)

    tracked = section_sample.HERE / "annotation"
    with pytest.raises(ValueError, match="git-ignored"):
        section_sample.require_untracked(tracked)
    section_sample.require_untracked(section_sample.CACHE_DIR / "annotation")
    section_sample.require_untracked(tmp_path)


def test_build_sample_records_the_fast_reading() -> None:
    docs = [("d2", _doc(3)), ("d1", _doc(20, subs=1)), ("d0", Document.from_sections([]))]
    rows, summary = section_sample.build_sample(docs, max_per_doc=6, double_fraction=0.25)
    assert {r["doc_id"] for r in rows} == {"d1", "d2"}
    assert summary["documents_without_sections"] == ["d0"]
    assert summary["selected"] == len(rows) == 6 + 6 and summary["double_annotated"] == 3
    assert summary["reading"]["profile"] == "fast" and summary["reading"]["elements"] is False
    assert summary["reading"]["settings"]["layout"] == "none"
    assert all(r["title_sha1"] == section_sample.title_sha1(r["title"]) for r in rows)
    top = next(r for r in rows if r["level"] == 1 and r["doc_id"] == "d1")
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


def _levels(doc: Document, max_per_doc: int = 12) -> list[int]:
    return [s.level for s in section_sample.sample_sections(doc, max_per_doc=max_per_doc)]


def test_sample_stratifies_levels() -> None:
    # 10 level-1 and 60 deeper: a third of the cap goes to level 1, the rest to the deeper ones
    levels = _levels(_doc(10, subs=3))
    assert levels.count(1) == 4 and len(levels) == 12 and {2, 3} <= set(levels)
    # unused slots pass to the other side
    assert _levels(_doc(2, subs=20)).count(1) == 2  # both level-1, ten deeper
    assert len(_levels(_doc(2, subs=20))) == 12
    assert _levels(_doc(20, subs=0)).count(1) == 12  # nothing deeper: all twelve are level 1
    assert _levels(_doc(3, subs=0)) == [1, 1, 1]
    deep = _doc(40, subs=1)
    assert sorted(_levels(deep, 4))[:2] == [1, 2] and len(_levels(deep, 4)) == 4  # one slot
    # level 1 is picked in proportion to the cap, not to how common it is
    assert _levels(_doc(5, subs=8)).count(1) == 4


def test_every_document_with_level_one_gets_one() -> None:
    import random

    rng = random.Random(7)
    for _ in range(200):
        doc = _doc(rng.randint(1, 15), subs=rng.randint(0, 8))
        n1 = sum(1 for s in doc.walk() if s.level == 1)
        got = _levels(doc)
        assert got.count(1) >= min(4, n1)
        assert len(got) == min(12, len([s for s in doc.walk() if s.level >= 1]))


def test_sample_is_not_fooled_by_periodic_structure() -> None:
    # a level-1 heading and three level-2 ones, each with a level-3 one, repeated
    doc = _doc(12, subs=3)
    picked = section_sample.sample_sections(doc, max_per_doc=12)
    assert [s.level for s in picked].count(1) == 4
    assert len({s.title for s in picked if s.level > 1}) == 8
    offsets = {int(s.title.split(".")[-1]) for s in picked if s.level > 1}
    assert len(offsets) > 1  # not always the same child of a part


def test_repeated_titles_are_sampled_once() -> None:
    entries = [{"title": "Annual Report 2025", "level": 1, "text": "running header"}] * 5
    entries += [{"title": "Real", "level": 1, "text": "x"}]
    doc = Document.from_sections(entries)
    assert [s.title for s in section_sample.sample_sections(doc)] == ["Annual Report 2025", "Real"]


def test_items_are_shuffled_across_documents_and_double_is_a_fifth() -> None:
    rows = [{"doc_id": f"d{k // 6}", "section_id": f"s{k}", "title": "t"} for k in range(60)]
    items = section_sample.assign_items(rows)
    again = section_sample.assign_items(rows)
    assert items == again  # deterministic
    assert [r["item"] for r in items] == [f"s{k:04d}" for k in range(60)]
    assert sum(r["double"] for r in items) == 12
    assert {r["section_id"] for r in items} == {r["section_id"] for r in rows}
    docs = [r["doc_id"] for r in items]
    assert docs != sorted(docs)  # not in document order
    assert sum(a == b for a, b in zip(docs, docs[1:], strict=False)) < 20  # few neighbours
    assert section_sample.assign_items(rows, seed=1) != items


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _annotated(tmp_path: Path):
    sample = _sample(10)
    sample_path = _write(tmp_path / "sample.jsonl", sample)
    double = [r["item"] for r in sample if r["double"]]
    every = [{"item": r["item"], "label": "none", "annotator": "A1"} for r in sample]
    other = "government.highlights"
    second = [
        {"item": i, "label": other if k == 0 else "none", "annotator": "B1"}
        for k, i in enumerate(double)
    ]
    first_path = _write(tmp_path / "first.jsonl", every)
    second_path = _write(tmp_path / "second.jsonl", second)
    return sample, sample_path, first_path, second_path, double


def test_validate_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, sample_path, first_path, _, _ = _annotated(tmp_path)
    assert section_gold.main(["validate", "--sample", str(sample_path), str(first_path)]) == 0
    assert "10 valid" in capsys.readouterr().out
    bad = _write(
        tmp_path / "bad.jsonl",
        [{"item": "s0000", "label": "financial", "annotator": "A1"}],
    )
    assert section_gold.main(["validate", "--sample", str(sample_path), str(bad)]) == 1
    assert "is a group" in capsys.readouterr().err


def test_adjudication_batch_and_undecided_merge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample, sample_path, first_path, second_path, double = _annotated(tmp_path)
    out = tmp_path / "adjudication.jsonl"
    argv = ["adjudication-batch", "--sample", str(sample_path), "--first", str(first_path)]
    argv += ["--second", str(second_path), "--out", str(out)]
    assert section_gold.main(argv) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["item"] for r in rows] == double[:1]  # only the disputed item
    assert sorted(rows[0]["labels"]) == ["government.highlights", "none"]
    assert "A1" not in out.read_text(encoding="utf-8")  # no annotator names
    assert "B1" not in out.read_text(encoding="utf-8")
    assert not {"doc_id", "section_id", "title_sha1", "double"} & set(rows[0])
    # the label order does not give away who was first: it varies by item, and is fixed per item
    many = [{**r, "double": True} for r in _sample(40)]
    firsts = {r["item"]: section_gold.Label("none", "A1") for r in many}
    seconds = {r["item"]: section_gold.Label("government.highlights", "B1") for r in many}
    shown = section_gold._adjudication_rows(many, firsts, seconds)
    assert {tuple(r["labels"]) for r in shown} == {
        ("none", "government.highlights"),
        ("government.highlights", "none"),
    }
    assert shown == section_gold._adjudication_rows(many, firsts, seconds)

    gold, agreement = tmp_path / "g.jsonl", tmp_path / "a.json"
    merge = ["merge", "--sample", str(sample_path), "--first", str(first_path)]
    merge += ["--second", str(second_path), "--gold", str(gold), "--agreement", str(agreement)]
    empty = _write(tmp_path / "empty.jsonl", [])
    summary = _write(tmp_path / "summary.json", [])
    summary.write_text(json.dumps({"reading": {"profile": "fast"}}), encoding="utf-8")
    undecided = [*merge, "--adjudication", str(empty), "--sample-summary", str(summary)]
    assert section_gold.main(undecided) == 1
    assert "no adjudication" in capsys.readouterr().err
    assert not gold.exists()
    own = _write(tmp_path / "own.jsonl", [{"item": double[0], "label": "none", "annotator": "A1"}])
    assert (
        section_gold.main([*merge, "--adjudication", str(own), "--sample-summary", str(summary)])
        == 1
    )
    assert "third annotator" in capsys.readouterr().err
    third = _write(
        tmp_path / "third.jsonl",
        [{"item": double[0], "label": "none", "annotator": "C1"}],
    )
    no_reading = _write(tmp_path / "noreading.json", [])
    no_reading.write_text(json.dumps({"counts": 1}), encoding="utf-8")
    assert (
        section_gold.main(
            [*merge, "--adjudication", str(third), "--sample-summary", str(no_reading)]
        )
        == 1
    )
    assert "reading" in capsys.readouterr().err
    assert not gold.exists()
    assert (
        section_gold.main([*merge, "--adjudication", str(third), "--sample-summary", str(summary)])
        == 0
    )
    assert len(gold.read_text(encoding="utf-8").splitlines()) == 10


def test_gold_documents_reads_the_gold_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": i, "split": split}) + "\n"
            for i, split in (("b", "gold"), ("a", "gold"), ("c", "train"), ("d", "calib"))
        ),
        encoding="utf-8",
    )
    assert section_sample.gold_documents(manifest) == ["a", "b"]
