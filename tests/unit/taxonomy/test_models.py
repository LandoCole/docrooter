import pytest

from docrooter.errors import TaxonomyError
from docrooter.taxonomy import Taxonomy, TaxonomyNode


def _n(id, parent=None, label=None):
    return TaxonomyNode(id=id, label=label or id.upper(), parent_id=parent)


def test_resolve_path_by_label_uses_parent_context():
    t = Taxonomy.from_dict(
        {
            "nodes": [
                {"id": "a", "label": "A", "children": [{"id": "a.o", "label": "Other"}]},
                {"id": "b", "label": "B", "children": [{"id": "b.o", "label": "Other"}]},
            ]
        }
    )
    assert t.resolve_path(["B", "other"]) == ["b", "b.o"]


def test_resolve_path_prefers_id_and_names_options_on_failure():
    t = Taxonomy([_n("a"), _n("a.x", "a"), _n("a.y", "a", label="Why")])
    assert t.resolve_path(["a", "a.x"]) == ["a", "a.x"]
    assert t.resolve_path([]) == []
    with pytest.raises(KeyError) as e:
        t.resolve_path(["a", "nope"])
    assert "nope" in str(e.value) and "a.x" in str(e.value) and "Why" in str(e.value)
    with pytest.raises(KeyError):
        t.resolve_path(["a.x"])  # not a top-level node


def test_deep_and_wide():
    nodes = [
        TaxonomyNode(id=f"n{d}", label=f"L{d}", parent_id=f"n{d - 1}" if d else None)
        for d in range(30)
    ]
    nodes += [TaxonomyNode(id=f"w{i}", label=f"W{i}", parent_id="n0") for i in range(500)]
    t = Taxonomy(nodes)
    assert t.depth() == 30 and len(t.children("n0")) == 501 and len(t.path("n29")) == 30
    assert [n.id for n in t.path("n29")][:2] == ["n0", "n1"]
    assert t.node_depth("n29") == 30 and t.node_depth(Taxonomy.ROOT_ID) == 0


def test_depth_counts_nodes_on_longest_path():
    assert Taxonomy([_n("a"), _n("b")]).depth() == 1
    assert Taxonomy([_n("a"), _n("b", "a")]).depth() == 2


def test_children_leaves_and_derived_child_ids():
    t = Taxonomy([_n("a"), _n("b"), _n("a.1", "a"), _n("a.2", "a")])
    assert [n.id for n in t.children()] == ["a", "b"]
    assert [n.id for n in t.children("a")] == ["a.1", "a.2"]
    assert t.node("a").children == ["a.1", "a.2"]
    assert [n.id for n in t.leaves()] == ["b", "a.1", "a.2"]
    assert len(t) == 4 and "a.1" in t


def test_constructor_does_not_mutate_input_and_ignores_stale_children():
    a = TaxonomyNode(id="a", label="A", children=["stale"])
    t = Taxonomy([a, _n("b", "a")])
    assert a.children == ["stale"]
    assert t.node("a").children == ["b"]


def test_cycle_is_rejected_and_cannot_hang():
    with pytest.raises(TaxonomyError) as e:
        Taxonomy([_n("a", "b"), _n("b", "a")])
    assert any(i.code == "cycle" for i in e.value.issues)
    with pytest.raises(TaxonomyError):
        Taxonomy([_n("a", "a")])


def test_unknown_node_lookups_raise_key_error():
    t = Taxonomy([_n("a")])
    for call in (t.node, t.children, t.path, t.node_depth):
        with pytest.raises(KeyError):
            call("zzz")


def test_to_dict_round_trip_and_hash_independent_of_node_order():
    nodes = [_n("a"), _n("a.1", "a"), _n("b")]
    t = Taxonomy(nodes)
    again = Taxonomy.from_dict(t.to_dict())
    assert again.to_dict() == t.to_dict()
    assert again.content_hash() == t.content_hash()
    reordered = Taxonomy([nodes[0], nodes[2], nodes[1]])  # same top-level order
    assert reordered.content_hash() == t.content_hash()
    changed = Taxonomy([_n("a"), _n("a.1", "a", label="Different"), _n("b")])
    assert changed.content_hash() != t.content_hash()
    assert len(t.content_hash()) == 64


def test_explicit_root_parent_means_top_level():
    t = Taxonomy([TaxonomyNode(id="a", label="A", parent_id="__root__")])
    assert [n.id for n in t.children()] == ["a"]
    assert t.node("a").parent_id is None


def test_node_rejects_unknown_fields():
    with pytest.raises(ValueError):
        TaxonomyNode(id="a", label="A", descripton="typo")


def test_root_node_is_synthetic_with_top_level_children():
    from docrooter.taxonomy.validate import ROOT_ID

    t = Taxonomy([_n("b"), _n("a"), _n("a.1", "a"), _n("a.2", "a")])
    root = t.node(ROOT_ID)
    assert root.id == ROOT_ID and root.label == "(root)" and root.parent_id is None
    assert root.children == ["b", "a"]
    assert root.multi_label is False and root.threshold is None and root.backend is None
    assert t.path(ROOT_ID) == [] and t.node_depth(ROOT_ID) == 0
    assert ROOT_ID not in t.nodes and ROOT_ID not in t and len(t) == 4
    assert t.to_dict().get("root") is None


def test_root_settings_applied_round_tripped_and_hashed():
    nodes = [_n("a"), _n("b")]
    t = Taxonomy(nodes, root={"multi_label": True, "threshold": 0.25, "description": "top"})
    root = t.node(Taxonomy.ROOT_ID)
    assert (root.multi_label, root.threshold, root.description) == (True, 0.25, "top")
    assert t.to_dict()["root"] == {"multi_label": True, "threshold": 0.25, "description": "top"}
    again = Taxonomy.from_dict(t.to_dict())
    assert again.node(Taxonomy.ROOT_ID).multi_label is True
    assert again.content_hash() == t.content_hash()
    assert t.content_hash() != Taxonomy(nodes).content_hash()
    assert (
        Taxonomy(nodes, root={"multi_label": False}).content_hash()
        == Taxonomy(nodes).content_hash()
    )


def test_top_level_order_changes_hash():
    assert (
        Taxonomy([_n("a"), _n("b")]).content_hash() != Taxonomy([_n("b"), _n("a")]).content_hash()
    )


@pytest.mark.parametrize("root", [{"colour": "red"}, {"threshold": "high"}, ["multi_label"]])
def test_invalid_root_settings_raise(root):
    with pytest.raises(TaxonomyError) as e:
        Taxonomy([_n("a")], root=root)
    assert [i.code for i in e.value.issues] == ["invalid_root"]
