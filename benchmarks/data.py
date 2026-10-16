"""Dataset adapters: B1 (EDGAR-CORPUS), B2 (DBPedia Classes) and B3 (LEDGAR) as ``BenchData``.

Every loader takes its rows through an injectable parameter, so the mapping logic is testable
without network access; with ``rows=None`` the real dataset is fetched from the Hugging Face Hub
(needs the ``bench`` extra).
"""

from __future__ import annotations

import functools
import random
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from docrooter import Dataset, SectionView, Taxonomy, TaxonomyNode
from docrooter.train.dataset import Example

TAXONOMY_DIR = Path(__file__).parent / "taxonomies"

DATASET_KEYS = ("b1", "b2", "b3")

# Same text-only view for every system, so the comparison is about the backend alone.
BENCH_VIEW = SectionView(fields=("text",), max_tokens=512)

INSTRUCTIONS = {
    "b1": "Classify this section of a 10-K annual report",
    "b2": "Classify the topic of this Wikipedia article",
    "b3": "Classify the type of this contract clause",
}

MIN_ITEM_WORDS = 50
MAX_ITEM_CHARS = 10_000  # far beyond the 512-token view; keeps memory bounded
POOL_PER_LEAF = 20
B3_SUBSET_SIZE = 300

PART_ITEMS: dict[str, tuple[str, ...]] = {
    "part-i": ("1", "1A", "1B", "2", "3", "4"),
    "part-ii": ("5", "6", "7", "7A", "8", "9", "9A", "9B"),
    "part-iii": ("10", "11", "12", "13", "14"),
    "part-iv": ("15",),
}
ITEM_PART = {item: part for part, items in PART_ITEMS.items() for item in items}


class BenchData(NamedTuple):
    """One benchmark: its taxonomy, the test examples and a pool to draw few-shot examples from.

    ``subset_ids`` indexes ``test`` (B3 only: the fixed like-for-like subset); ``None`` otherwise.
    """

    name: str
    taxonomy: Taxonomy
    test: Dataset
    train_pool: Dataset
    subset_ids: list[int] | None

    def subset(self) -> Dataset | None:
        """The like-for-like subset of ``test``, or ``None`` when the benchmark has none."""
        if self.subset_ids is None:
            return None
        return Dataset([self.test.examples[i] for i in self.subset_ids])


# --- helpers ----------------------------------------------------------------------------


def slug(label: str) -> str:
    return re.sub(r"[\W_]+", "-", label.lower()).strip("-") or "untitled"


_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_camel(label: str) -> str:
    """``"AmusementParkAttraction"`` -> ``"Amusement Park Attraction"`` (acronyms stay whole)."""
    return _CAMEL_BOUNDARY.sub(" ", label)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def pool_shots(pool: Dataset, k: int, seed: int) -> Dataset:
    """``k`` examples per leaf (the last path element) from ``pool``; fewer if a leaf has fewer.

    Deterministic for a seed; the result keeps the pool's order.
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    groups: dict[str, list[int]] = {}
    for i, example in enumerate(pool.examples):
        groups.setdefault(example.path[-1], []).append(i)
    rng = random.Random(seed)
    keep: list[int] = []
    for leaf in sorted(groups):
        members = groups[leaf][:]
        rng.shuffle(members)
        keep.extend(members[:k])
    return Dataset([pool.examples[i] for i in sorted(keep)])


def _path_taxonomy(paths: Iterable[tuple[tuple[str, str], ...]]) -> Taxonomy:
    """Taxonomy from ``((id_part, label), ...)`` paths; node ids are the dotted id parts."""
    nodes: dict[str, TaxonomyNode] = {}
    for path in sorted(set(paths)):
        parent: str | None = None
        for part, label in path:
            node_id = f"{parent}.{part}" if parent else part
            if node_id not in nodes:
                nodes[node_id] = TaxonomyNode(id=node_id, label=label, parent_id=parent)
            parent = node_id
    return Taxonomy(list(nodes.values()))


def _cap_per_leaf(examples: Iterable[Example], per_leaf: int) -> list[Example]:
    """The first ``per_leaf`` examples of every leaf, in input order."""
    counts: dict[str, int] = {}
    kept: list[Example] = []
    for example in examples:
        leaf = example.path[-1]
        if counts.get(leaf, 0) < per_leaf:
            counts[leaf] = counts.get(leaf, 0) + 1
            kept.append(example)
    return kept


def _sample(items: list[Any], n: int, rng: random.Random) -> list[Any]:
    return rng.sample(items, min(n, len(items)))


# --- B1: EDGAR-CORPUS -------------------------------------------------------------------


def edgar_taxonomy() -> Taxonomy:
    return Taxonomy.load(TAXONOMY_DIR / "edgar.json")


def edgar_item_path(item: str) -> list[str]:
    """``"7A"`` -> ``["part-ii", "part-ii.item-7a"]``."""
    part = ITEM_PART[item]
    return [part, f"{part}.item-{item.lower()}"]


_HEADING = re.compile(r"\s*(?:part\s+[ivx]+\W+)?item\s*\d+[a-c]?\b\W*", re.IGNORECASE)


def _words(text: str) -> list[str]:
    return text.split()


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


@functools.cache
def _item_titles() -> dict[str, list[str]]:
    """Normalised words of each standard item title, e.g. ``"7A"`` -> ``["quantitative", ...]``."""
    titles = {}
    for item in ITEM_PART:
        node = edgar_taxonomy().node(edgar_item_path(item)[-1])
        titles[item] = [w for w in map(_norm, _words(node.label.split(". ", 1)[1])) if w]
    return titles


def strip_item_heading(text: str, item: str) -> str:
    """Remove the leading ``ITEM 7A. QUANTITATIVE AND ...`` heading that every EDGAR-CORPUS item
    starts with: the item marker, then as many of the standard title's words as follow it."""
    match = _HEADING.match(text)
    if match is None:
        return text
    rest = _words(text[match.end() :])
    title, i, j = _item_titles()[item], 0, 0
    while i < len(rest) and j < len(title):
        word = _norm(rest[i])
        if word == "":  # bare punctuation between words
            i += 1
        elif word == title[j]:
            i, j = i + 1, j + 1
        else:
            break
    return " ".join(rest[i:])


def _filing_examples(filing: Mapping[str, Any], doc_id: str, strip_headings: bool) -> list[Example]:
    out = []
    for item in ITEM_PART:
        text = filing.get(f"section_{item}")
        if not isinstance(text, str):
            continue
        text = collapse_whitespace(text)
        if strip_headings:
            text = strip_item_heading(text, item)
        if len(text.split()) < MIN_ITEM_WORDS:
            continue
        out.append(
            Example(
                text=text[:MAX_ITEM_CHARS],
                path=edgar_item_path(item),
                doc_id=doc_id,
                section_id=item,
            )
        )
    return out


def _stream_edgar(year: int, max_filings: int) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    # The hub repo ships a loading script that `datasets` >= 4 refuses, so read the raw
    # JSONL file the script points at (columns: filename, cik, year, section_1 ... section_15).
    ds = load_dataset(
        "json",
        data_files=f"hf://datasets/eloukas/edgar-corpus/{year}/train.jsonl",
        split="train",
        streaming=True,
    )
    for i, row in enumerate(ds):
        if i >= max_filings:
            return
        yield row


def load_b1(
    n_test: int = 1000,
    seed: int = 0,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    year: int = 2020,
    max_filings: int | None = None,
    pool_per_leaf: int = POOL_PER_LEAF,
    strip_headings: bool = True,
) -> BenchData:
    """EDGAR-CORPUS 10-K items labelled Part -> Item. Filings are split into test and pool halves.

    ``rows`` are filing dicts with ``section_1`` ... ``section_15`` strings. Items under 50 words
    are dropped. The test set is ``n_test`` items sampled from the test half of the filings.

    Every item in the corpus starts with its own heading (``ITEM 7A. QUANTITATIVE AND ...``),
    which would hand the label to the model; ``strip_headings`` (default) removes it before the
    length filter, ``strip_headings=False`` keeps the text verbatim.
    """
    if rows is None:
        # about 14 usable items per filing; read enough for n_test items from half of them
        rows = _stream_edgar(year, max_filings or max(40, n_test // 2))
    filings = list(rows)
    rng = random.Random(seed)
    order = list(range(len(filings)))
    rng.shuffle(order)
    half = (len(order) + 1) // 2
    test_filings, pool_filings = order[:half], order[half:]

    def explode(indices: list[int]) -> list[Example]:
        out: list[Example] = []
        for i in indices:
            out.extend(
                _filing_examples(filings[i], str(filings[i].get("filename", i)), strip_headings)
            )
        return out

    test = _sample(explode(test_filings), n_test, rng)
    pool_items = explode(pool_filings)
    rng.shuffle(pool_items)
    return BenchData(
        name="b1",
        taxonomy=edgar_taxonomy(),
        test=Dataset(test),
        train_pool=Dataset(_cap_per_leaf(pool_items, pool_per_leaf)),
        subset_ids=None,
    )


# --- B2: DBPedia Classes ----------------------------------------------------------------


def _dbpedia_example(row: Mapping[str, Any]) -> tuple[Example, tuple[tuple[str, str], ...]]:
    labels = [split_camel(str(row[k])) for k in ("l1", "l2", "l3")]
    parts = tuple((slug(label), label) for label in labels)
    path = [".".join(p for p, _ in parts[: i + 1]) for i in range(len(parts))]
    return Example(text=str(row["text"]), path=path), parts


def load_b2(
    n_test: int = 1000,
    seed: int = 0,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    train_rows: Iterable[Mapping[str, Any]] | None = None,
    pool_per_leaf: int = POOL_PER_LEAF,
) -> BenchData:
    """DBPedia Classes (``text``, ``l1``, ``l2``, ``l3``) with CamelCase labels split into words.

    ``rows`` is the test split. ``train_rows`` feeds the few-shot pool and should already be in
    random order (the first ``pool_per_leaf`` per leaf are kept); without it the pool is empty.
    """
    if rows is None:
        rows, train_rows = _hf_dbpedia(seed)
    test_pairs = [_dbpedia_example(r) for r in rows]
    paths = {p for _, p in test_pairs}
    pool: list[Example] = []
    if train_rows is not None:
        counts: dict[str, int] = {}
        for row in train_rows:
            example, parts = _dbpedia_example(row)
            paths.add(parts)
            leaf = example.path[-1]
            if counts.get(leaf, 0) < pool_per_leaf:
                counts[leaf] = counts.get(leaf, 0) + 1
                pool.append(example)
    rng = random.Random(seed)
    test = _sample([e for e, _ in test_pairs], n_test, rng)
    return BenchData("b2", _path_taxonomy(paths), Dataset(test), Dataset(pool), None)


def _hf_dbpedia(seed: int) -> tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]]]:
    from datasets import load_dataset

    ds = load_dataset("DeveloperOats/DBPedia_Classes")
    return ds["test"], ds["train"].shuffle(seed=seed)


# --- B3: LEDGAR -------------------------------------------------------------------------


def load_b3(
    n_test: int = 1000,
    seed: int = 0,
    *,
    rows: Iterable[Mapping[str, Any]] | None = None,
    train_rows: Iterable[Mapping[str, Any]] | None = None,
    label_names: list[str] | None = None,
    pool_per_leaf: int = POOL_PER_LEAF,
    subset_size: int = B3_SUBSET_SIZE,
) -> BenchData:
    """LEDGAR (``text``, integer ``label``) as a flat taxonomy of the ``label_names``.

    ``subset_ids`` are the first ``subset_size`` indices of the (seeded, shuffled) test sample,
    so the subset is the same for any ``n_test`` of at least ``subset_size``.
    """
    if rows is None:
        rows, train_rows, label_names = _hf_ledgar(seed)
    if label_names is None:
        raise ValueError("label_names is required with injected rows")
    taxonomy = _path_taxonomy({((slug(n), n),) for n in label_names})
    ids = [slug(n) for n in label_names]

    def example(row: Mapping[str, Any]) -> Example:
        return Example(text=str(row["text"]), path=[ids[int(row["label"])]])

    test = _sample([example(r) for r in rows], n_test, random.Random(seed))
    pool = _cap_per_leaf((example(r) for r in train_rows or ()), pool_per_leaf)
    return BenchData(
        name="b3",
        taxonomy=taxonomy,
        test=Dataset(test),
        train_pool=Dataset(pool),
        subset_ids=list(range(min(subset_size, len(test)))),
    )


def _hf_ledgar(seed: int) -> tuple[Iterable[Mapping[str, Any]], Iterable[Mapping[str, Any]], list]:
    from datasets import load_dataset

    ds = load_dataset("coastalcph/lex_glue", "ledgar")
    return ds["test"], ds["train"].shuffle(seed=seed), ds["test"].features["label"].names


LOADERS = {"b1": load_b1, "b2": load_b2, "b3": load_b3}
