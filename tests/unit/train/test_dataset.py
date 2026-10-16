import json
from pathlib import Path

import pytest

import docrooter
from docrooter import SectionView
from docrooter.classify.results import DocumentResult, PathResult, SectionResult
from docrooter.errors import DatasetError
from docrooter.taxonomy import Taxonomy
from docrooter.train import Dataset, Example

FIXTURES = Path(__file__).parents[2] / "fixtures"
SMALL_JSONL = FIXTURES / "datasets" / "small.jsonl"
SMALL_CSV = FIXTURES / "datasets" / "small.csv"


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


def test_exported_from_package():
    assert docrooter.Dataset is Dataset


def test_dataset_jsonl_csv_equal():
    a, b = Dataset.load(SMALL_JSONL), Dataset.load(SMALL_CSV)
    assert len(a) == len(b) == 8
    assert list(a) == list(b)
    assert ["Legal", "Litigation"] in [e.path for e in a]  # the one path given as labels


def test_save_roundtrip(tmp_path):
    original = Dataset.load(SMALL_JSONL)
    for name in ("out.jsonl", "out.csv"):
        original.save(tmp_path / name)
        assert list(Dataset.load(tmp_path / name)) == list(original)


def test_csv_with_bom_crlf_and_padding(tmp_path):
    f = tmp_path / "d.csv"
    f.write_bytes("﻿text,path\r\nhello world,  Financial > Tax \r\n\r\n".encode())
    assert [e.path for e in Dataset.load(f)] == [["Financial", "Tax"]]


def test_resolve_to_ids_mixes_ids_and_labels(nested):
    resolved = Dataset.load(SMALL_JSONL).resolve(nested)
    assert {tuple(e.path) for e in resolved} == {
        ("financial", "financial.tax"),
        ("financial", "financial.audit"),
        ("legal", "legal.contracts"),
        ("legal", "legal.litigation"),
    }


def test_resolve_labels_by_parent_context():
    t = Taxonomy.from_dict(
        {
            "nodes": [
                {"id": "a", "label": "A", "children": [{"id": "a.o", "label": "Other"}]},
                {"id": "b", "label": "B", "children": [{"id": "b.o", "label": "Other"}]},
            ]
        }
    )
    out = Dataset([Example(text="x", path=["B", "Other"]), Example(text="y", path=["a", "other"])])
    assert [e.path for e in out.resolve(t)] == [["b", "b.o"], ["a", "a.o"]]


def test_resolve_error_has_line(tmp_path, nested):
    f = tmp_path / "d.jsonl"
    good = json.dumps({"text": "t", "path": ["financial", "financial.tax"]})
    bad = json.dumps({"text": "t", "path": ["financial", "nope"]})
    f.write_text(f"{good}\n\n{good}\n{bad}\n{bad}\n")  # the blank line is not counted
    with pytest.raises(DatasetError) as e:
        Dataset.load(f).resolve(nested)
    assert "line 3" in str(e.value) and "'nope' not found" in str(e.value)
    assert "line 4" not in str(e.value)  # only the first failure is reported


def test_resolve_error_line_in_csv(tmp_path, nested):
    f = tmp_path / "d.csv"
    f.write_text("text,path\nok,financial > Tax\nbad,financial > zzz\n")
    with pytest.raises(DatasetError, match="line 3"):
        Dataset.load(f).resolve(nested)


def test_load_errors(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        Dataset.load(tmp_path / "missing.jsonl")
    f = tmp_path / "d.jsonl"
    f.write_text('{"text": "ok", "path": ["a"]}\n{not json}\n')
    with pytest.raises(DatasetError, match="line 2"):
        Dataset.load(f)
    f.write_text('{"text": "ok"}\n')
    with pytest.raises(DatasetError, match="line 1"):
        Dataset.load(f)
    g = tmp_path / "d.csv"
    g.write_text("body,path\nx,a\n")
    with pytest.raises(DatasetError, match="text"):
        Dataset.load(g)


def test_split_stratified_deterministic(nested):
    data = Dataset.load(SMALL_JSONL).resolve(nested)
    train, test = data.split(test_size=0.25, seed=3)
    again_train, again_test = data.split(test_size=0.25, seed=3)
    assert list(train) == list(again_train) and list(test) == list(again_test)
    assert len(train) + len(test) == 8
    assert {e.path[0] for e in train} == {e.path[0] for e in test} == {"financial", "legal"}
    assert {e.text for e in train}.isdisjoint({e.text for e in test})
    other = [data.split(test_size=0.25, seed=s)[1] for s in range(8)]
    assert len({tuple(e.text for e in t) for t in other}) > 1  # the seed matters


def test_split_small_class_stays_in_train():
    rare = Example(text="solo", path=["z"])
    data = Dataset([Example(text=f"a{i}", path=["a"]) for i in range(5)] + [rare])
    train, test = data.split(test_size=0.4, seed=0)
    assert rare in list(train) and rare not in list(test)
    assert len(test) == 2 and {e.path[0] for e in test} == {"a"}
    with pytest.raises(ValueError):
        data.split(test_size=1.5)


def test_split_stratify_depth_two(nested):
    data = Dataset.load(SMALL_JSONL).resolve(nested)
    train, test = data.split(test_size=0.5, stratify_depth=2, seed=1)
    assert len(train) == len(test) == 4
    assert {tuple(e.path) for e in train} == {tuple(e.path) for e in test}


def test_node_examples(nested):
    data = Dataset.load(SMALL_JSONL)  # includes the path given as labels
    texts, labels = data.node_examples(nested, "financial")
    assert len(texts) == len(labels) == 4
    assert sorted(set(labels)) == ["financial.audit", "financial.tax"]
    assert all("tax" in t.lower() or "audit" in t.lower() or "control" in t.lower() for t in texts)
    texts, labels = data.node_examples(nested, "legal")
    assert sorted(set(labels)) == ["legal.contracts", "legal.litigation"] and len(texts) == 4
    assert data.node_examples(nested, "financial.tax") == ([], [])  # nothing below a leaf
    texts, labels = data.node_examples(nested, Taxonomy.ROOT_ID)
    assert sorted(set(labels)) == ["financial", "legal"] and len(texts) == 8


def test_node_examples_skips_paths_ending_at_the_node(nested):
    data = Dataset(
        [Example(text="t", path=["financial"]), Example(text="u", path=["Financial", "Tax"])]
    )
    assert data.node_examples(nested, "financial") == (["u"], ["financial.tax"])
    assert data.node_examples(nested, Taxonomy.ROOT_ID) == (["t", "u"], ["financial", "financial"])


def test_from_results_roundtrip(tmp_path):
    doc = docrooter.load(FIXTURES / "docs" / "simple.md")
    ids = list(doc.sections)

    def result(sid, node_ids):
        best = None if node_ids is None else PathResult(node_ids=node_ids)
        return SectionResult(section_id=sid, title=doc.sections[sid].title, best=best, n_tokens=5)

    sections = {
        ids[0]: result(ids[0], ["legal", "legal.contracts"]),
        ids[1]: result(ids[1], ["financial"]),
        ids[2]: result(ids[2], []),  # empty best path
        ids[3]: result(ids[3], None),  # unclassified
    }
    view = SectionView(fields=("title", "text"))
    data = Dataset.from_results(DocumentResult(source=doc.source, sections=sections), doc, view)
    rendered = view.render_all(doc, [ids[0], ids[1]])
    assert [(e.text, e.path, e.section_id) for e in data] == [
        (rendered[ids[0]].text, ["legal", "legal.contracts"], ids[0]),
        (rendered[ids[1]].text, ["financial"], ids[1]),
    ]
    assert all(e.doc_id == doc.source for e in data)
    data.save(tmp_path / "d.jsonl")
    assert list(Dataset.load(tmp_path / "d.jsonl")) == list(data)


def test_jsonl_roundtrip_with_unicode_line_separators(tmp_path):
    text = "first\u2028second\u2029third\u0085fourth\x0bfifth\x0csixth\x1csev"
    data = Dataset([Example(text=text, path=["a"]), Example(text="plain", path=["b"])])
    data.save(tmp_path / "d.jsonl")
    loaded = Dataset.load(tmp_path / "d.jsonl")
    assert list(loaded) == list(data)
    assert loaded._lines == [1, 2]
    data.save(tmp_path / "d.csv")
    assert list(Dataset.load(tmp_path / "d.csv")) == list(data)
