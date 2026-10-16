import json
from pathlib import Path

import pytest

from docrooter.errors import TaxonomyError
from docrooter.taxonomy import Taxonomy
from docrooter.taxonomy.io import from_csv, read_nodes, slug

FIX = Path(__file__).resolve().parents[2] / "fixtures" / "taxonomies"
FORMATS = [
    "nested.json",
    "flat.json",
    "adjacency.csv",
    "levels.csv",
    "path.csv",
    "excel_export.csv",
]


@pytest.mark.parametrize("name", FORMATS)
def test_all_formats_same_tree(name):
    t = Taxonomy.load(FIX / name)
    assert {n.id: [c.id for c in t.children(n.id)] for n in t.children()} == {
        "financial": ["financial.tax", "financial.audit"],
        "legal": ["legal.contracts", "legal.litigation"],
    }
    assert t.depth() == 2


def test_content_hash_is_stable_across_formats():
    hashes = {name: Taxonomy.load(FIX / name).content_hash() for name in FORMATS}
    assert len(set(hashes.values())) == 1, hashes


def test_excel_export_is_stripped_and_keeps_descriptions():
    t = Taxonomy.load(FIX / "excel_export.csv")
    assert t.node("financial.audit").description == "Audit reports and findings"
    assert t.node("financial").label == "Financial"
    assert (FIX / "excel_export.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_validation_errors_have_locations():
    with pytest.raises(TaxonomyError) as e:
        Taxonomy.load(FIX / "broken.csv")
    codes = {(i.code, i.location) for i in e.value.issues}
    assert ("missing_parent", "row 4") in codes and any(c == "cycle" for c, _ in codes)
    assert ("cycle", "row 5") in codes


def test_unknown_csv_format():
    with pytest.raises(TaxonomyError) as e:
        from_csv("name,thing\nx,y\n")
    assert [i.code for i in e.value.issues] == ["unknown_csv_format"]


def test_from_csv_accepts_text_and_path_and_custom_separator():
    text = "path\nA / B\nA / C\n"
    t = from_csv(text, sep="/")
    assert [n.id for n in t.children("a")] == ["a.b", "a.c"]
    assert (
        from_csv(FIX / "path.csv").content_hash() == Taxonomy.load(FIX / "path.csv").content_hash()
    )
    assert from_csv(str(FIX / "levels.csv")).depth() == 2


def test_levels_gap_and_duplicate_rows():
    with pytest.raises(TaxonomyError) as e:
        from_csv("level_1,level_2,level_3\nA,,C\n")
    assert e.value.issues[0].code == "empty_path_segment" and e.value.issues[0].location == "row 2"
    with pytest.raises(TaxonomyError) as e:
        from_csv("level_1,level_2\nA,B\nA,B\n")
    assert ("duplicate_id", "row 3") in {(i.code, i.location) for i in e.value.issues}


def test_adjacency_optional_columns_and_generated_ids():
    text = (
        "parent_id,label,examples,threshold,multi_label,backend\n"
        ",Top,one | two,0.4,true,jev\n"
        "top,Child,,,no,\n"
    )
    t = from_csv(text)
    top = t.node("top")
    assert top.examples == ["one", "two"] and top.threshold == 0.4
    assert top.multi_label is True and top.backend == "jev"
    assert t.node("top.child").multi_label is False and t.node("top.child").parent_id == "top"


def test_adjacency_bad_numbers_are_located():
    with pytest.raises(TaxonomyError) as e:
        from_csv("id,parent_id,label,threshold\na,,A,high\n")
    assert (e.value.issues[0].code, e.value.issues[0].location) == ("invalid_value", "row 2")


def test_nested_json_generates_ids_from_label_path():
    t = Taxonomy.from_dict(
        {"nodes": [{"label": "Risk Factors", "children": [{"label": "Market & Credit"}]}]}
    )
    assert t.node("risk-factors.market-credit").parent_id == "risk-factors"


def test_nested_json_parent_comes_from_nesting():
    t = Taxonomy.from_dict(
        {"nodes": [{"id": "p", "label": "P", "children": [{"id": "c", "label": "C"}]}]}
    )
    assert t.node("c").parent_id == "p"


def test_nested_json_parent_mismatch():
    with pytest.raises(TaxonomyError) as e:
        Taxonomy.from_dict(
            {
                "nodes": [
                    {
                        "id": "p",
                        "label": "P",
                        "children": [{"id": "c", "label": "C", "parent_id": "q"}],
                    }
                ]
            }
        )
    assert (e.value.issues[0].code, e.value.issues[0].location) == (
        "parent_mismatch",
        "/nodes/0/children/0",
    )


def test_json_pointer_locations():
    data = {
        "nodes": [
            {
                "id": "a",
                "label": "A",
                "children": [
                    {"id": "a.1", "label": "One"},
                    {"id": "a.2", "label": "Two", "parent_id": "a"},
                    {
                        "id": "a.3",
                        "label": "Three",
                        "children": [
                            {
                                "id": "a.3.x",
                                "label": "X",
                                "children": [{"id": "dup", "label": "D1"}],
                            }
                        ],
                    },
                ],
            },
            {"id": "dup", "label": "D2"},
        ]
    }
    with pytest.raises(TaxonomyError) as e:
        Taxonomy.from_dict(data)
    assert ("duplicate_id", "/nodes/1") in {(i.code, i.location) for i in e.value.issues}
    nodes, locs = read_nodes_from_dict(data)
    assert locs[2] == "/nodes/0/children/1"


def read_nodes_from_dict(data):
    from docrooter.taxonomy.io import nodes_from_dict

    return nodes_from_dict(data)


def test_bare_list_and_malformed_json():
    t = Taxonomy.from_dict([{"id": "a", "label": "A"}])
    assert [n.id for n in t.children()] == ["a"]
    for bad in ({"nodes": "x"}, {"nope": []}, 3, {"nodes": [5]}, {"nodes": [{"id": "a"}]}):
        with pytest.raises(TaxonomyError) as e:
            Taxonomy.from_dict(bad)
        assert e.value.issues


def test_load_invalid_json_file(tmp_path):
    p = tmp_path / "t.json"
    p.write_text("{not json")
    with pytest.raises(TaxonomyError) as e:
        Taxonomy.load(p)
    assert e.value.issues[0].code == "invalid_json"


def test_load_sniffs_format_without_known_suffix(tmp_path):
    (tmp_path / "t.txt").write_text(json.dumps({"nodes": [{"id": "a", "label": "A"}]}))
    (tmp_path / "u.dat").write_text("id,parent_id,label\na,,A\n")
    assert Taxonomy.load(tmp_path / "t.txt").node("a")
    assert Taxonomy.load(tmp_path / "u.dat").node("a")


def test_read_nodes_returns_nodes_without_validating():
    nodes, locs = read_nodes(FIX / "broken.csv")
    assert len(nodes) == 5 and locs[0] == "row 2"


def test_slug():
    assert slug("  Risk & Return!  ") == "risk-return"
    assert slug("Tax") == "tax"
    assert slug("!!!") == "untitled"


def test_warnings_logged_on_load(tmp_path, caplog):
    p = tmp_path / "t.csv"
    p.write_text("id,parent_id,label\na,,A\nb,a,B\n")
    with caplog.at_level("WARNING", logger="docrooter.taxonomy"):
        t = Taxonomy.load(p)
    assert [w.code for w in t.warnings] == ["single_child"]
    assert any("single_child" in r.getMessage() for r in caplog.records)
