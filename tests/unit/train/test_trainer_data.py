from pathlib import Path

import pytest

from docrooter.taxonomy import Taxonomy, TaxonomyNode
from docrooter.train import Dataset, Example
from docrooter.train.trainer import build_triplets, node_groups, trainable_nodes

FIXTURES = Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def nested() -> Taxonomy:
    return Taxonomy.load(FIXTURES / "taxonomies" / "nested.json")


def _dataset(counts: dict[tuple[str, ...], int]) -> Dataset:
    """Examples whose text encodes their path: ``"a/b#3"``."""
    return Dataset(
        [
            Example(text=f"{'/'.join(path)}#{i}", path=list(path))
            for path, n in counts.items()
            for i in range(n)
        ]
    )


def _path(text: str) -> list[str]:
    return text.split("#")[0].split("/")


COUNTS = {
    ("financial", "financial.tax"): 4,
    ("financial", "financial.audit"): 3,
    ("legal", "legal.contracts"): 5,
    ("legal", "legal.litigation"): 4,
    ("legal",): 2,  # stops at an internal node: a decision at the root only
}


def test_triplets_negatives_are_siblings(nested):
    triplets = build_triplets(nested, _dataset(COUNTS), seed=1)
    assert triplets
    for anchor, positive, negative in triplets:
        a, p, n = _path(anchor), _path(positive), _path(negative)
        assert anchor != positive
        # some node where all three pass through, anchor and positive take the same child
        # and the negative takes a different (sibling) child
        assert any(
            a[:k] == p[:k] == n[:k] and len(a) > k and len(n) > k and a[k] == p[k] and n[k] != a[k]
            for k in range(len(a))
        ), (anchor, positive, negative)


def test_triplets_cap_and_determinism(nested):
    data = _dataset(COUNTS)
    full = build_triplets(nested, data, seed=0)
    # root: 18 anchors; financial: 7; legal: 9 (the two examples stopping at legal are skipped)
    assert len(full) == 18 + 7 + 9
    capped = build_triplets(nested, data, max_triplets=5, seed=0)
    assert len(capped) == 5
    assert capped == build_triplets(nested, data, max_triplets=5, seed=0)
    assert full == build_triplets(nested, data, seed=0)
    assert full != build_triplets(nested, data, seed=1)


def test_triplets_cap_is_shared_across_nodes(nested):
    # the cap takes from every node in turn, so the big root pool cannot crowd the others out
    capped = build_triplets(nested, _dataset(COUNTS), max_triplets=6, seed=0)
    depths = [next(k for k in range(3) if _path(a)[k] != _path(n)[k]) for a, _, n in capped]
    assert sorted(depths) == [0, 0, 1, 1, 1, 1]  # root, financial, legal: two each


def test_triplets_skip_missing_sides(nested):
    # one example per child: no positive anywhere; one child only: no negative
    assert build_triplets(nested, _dataset({("financial", "financial.tax"): 1})) == []
    lone = _dataset({("financial", "financial.tax"): 1, ("legal", "legal.contracts"): 1})
    assert build_triplets(nested, lone) == []
    assert build_triplets(nested, _dataset({("financial", "financial.tax"): 3})) == []


def test_trainable_nodes_threshold(nested):
    data = _dataset(COUNTS)
    nodes = trainable_nodes(nested, data, min_examples_per_child=4)
    # financial.audit has 3 examples, so financial is excluded at min 4
    assert nodes == {
        Taxonomy.ROOT_ID: {"financial": 7, "legal": 11},
        "legal": {"legal.contracts": 5, "legal.litigation": 4},
    }
    at3 = trainable_nodes(nested, data, min_examples_per_child=3)
    assert at3["financial"] == {"financial.tax": 4, "financial.audit": 3}


def test_trainable_nodes_need_every_child(nested):
    # a head must cover every child the classifier will offer it
    data = _dataset({("legal", "legal.contracts"): 9, ("financial", "financial.tax"): 9})
    assert trainable_nodes(nested, data, min_examples_per_child=4) == {
        Taxonomy.ROOT_ID: {"financial": 9, "legal": 9}
    }


def test_node_groups_match_node_examples(nested):
    data = _dataset(COUNTS)
    groups = node_groups(nested, data)
    for node_id in (Taxonomy.ROOT_ID, "financial", "legal", "legal.contracts"):
        texts, children = data.node_examples(nested, node_id)
        assert groups.get(node_id, []) == list(zip(texts, children, strict=True))


def test_deep_taxonomy_is_iterative():
    nodes = [TaxonomyNode(id="n0", label="n0")]
    for i in range(1, 31):
        nodes.append(TaxonomyNode(id=f"n{i}", label=f"n{i}", parent_id=f"n{i - 1}"))
    nodes.append(TaxonomyNode(id="side", label="side", parent_id="n29"))
    tax = Taxonomy(nodes)
    chain = [f"n{i}" for i in range(31)]
    data = Dataset(
        [Example(text=f"deep {i}", path=chain) for i in range(4)]
        + [Example(text=f"side {i}", path=[*chain[:30], "side"]) for i in range(4)]
    )
    assert trainable_nodes(tax, data, min_examples_per_child=4) == {"n29": {"n30": 4, "side": 4}}
    triplets = build_triplets(tax, data)
    assert len(triplets) == 8
