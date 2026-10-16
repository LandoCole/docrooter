from docrooter.taxonomy import TaxonomyNode
from docrooter.taxonomy.validate import validate_nodes


def _n(id, parent=None, label=None):
    return TaxonomyNode(id=id, label=label if label is not None else id.upper(), parent_id=parent)


def _codes(issues):
    return {(i.severity, i.code) for i in issues}


def test_clean_taxonomy_has_no_issues():
    assert validate_nodes([_n("a"), _n("a.1", "a"), _n("a.2", "a")]) == []


def test_duplicate_id_located_at_second_occurrence():
    issues = validate_nodes([_n("a"), _n("a", label="Again")], locations=["row 2", "row 3"])
    dup = [i for i in issues if i.code == "duplicate_id"]
    assert [(i.severity, i.location) for i in dup] == [("error", "row 3")]


def test_missing_parent():
    issues = validate_nodes([_n("a"), _n("b", "ghost")], locations=["row 2", "row 3"])
    (issue,) = [i for i in issues if i.code == "missing_parent"]
    assert issue.severity == "error" and issue.location == "row 3" and "ghost" in issue.message


def test_cycle_reported_once_per_cycle_at_first_member():
    nodes = [
        _n("ok"),
        _n("a", "b"),
        _n("b", "c"),
        _n("c", "a"),
        _n("tail", "a"),
        _n("self", "self"),
    ]
    issues = validate_nodes(nodes, locations=[f"row {i}" for i in range(2, 8)])
    cycles = sorted((i.location, i.message) for i in issues if i.code == "cycle")
    assert [loc for loc, _ in cycles] == ["row 3", "row 7"]
    assert "a" in cycles[0][1] and "b" in cycles[0][1] and "c" in cycles[0][1]


def test_empty_label():
    issues = validate_nodes([_n("a", label="   ")])
    assert ("error", "empty_label") in _codes(issues)


def test_duplicate_sibling_labels_after_normalisation():
    nodes = [_n("p"), _n("x", "p", label="Risk  Factors"), _n("y", "p", label=" risk factors ")]
    issues = validate_nodes(nodes, locations=["row 2", "row 3", "row 4"])
    (issue,) = [i for i in issues if i.code == "duplicate_sibling_label"]
    assert issue.severity == "error" and issue.location == "row 4"


def test_same_label_under_different_parents_is_fine():
    nodes = [_n("a"), _n("b"), _n("a.o", "a", label="Other"), _n("b.o", "b", label="Other")]
    assert not [i for i in validate_nodes(nodes) if i.code == "duplicate_sibling_label"]


def test_single_child_is_only_a_warning():
    issues = validate_nodes([_n("a"), _n("a.1", "a")], locations=["row 2", "row 3"])
    (issue,) = issues
    assert (issue.severity, issue.code, issue.location) == ("warning", "single_child", "row 2")


def test_reserved_and_empty_taxonomy():
    assert ("error", "reserved_id") in _codes(validate_nodes([_n("__root__")]))
    assert ("error", "empty_taxonomy") in _codes(validate_nodes([]))
