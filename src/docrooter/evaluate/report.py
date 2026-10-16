"""``Report``, ``Comparison`` and the ``evaluate`` / ``compare`` entry points."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table
from rich.text import Text

from docrooter.errors import BackendError
from docrooter.evaluate.metrics import (
    depth_accuracy,
    exact_path_accuracy,
    hierarchical_prf,
    is_stopped,
    leaf_prf,
    selective_accuracy,
    sibling_confusions,
    stop_rate,
)
from docrooter.taxonomy import Taxonomy

if TYPE_CHECKING:
    from docrooter.classify.classifier import Classifier
    from docrooter.train.dataset import Dataset

# Backend kinds that run a local model on a device; the others are CPU-only or remote.
_MODEL_KINDS = {"embedding", "nli", "mlm-choice", "trained"}


class Report(BaseModel):
    """Evaluation of a classifier on a labelled dataset.

    ``exact`` is exact path accuracy; ``hp``/``hr``/``hf1`` the micro hierarchical precision,
    recall and F1; ``macro_p``/``macro_r``/``macro_f1`` the flat multi-class scores of the leaf
    label, averaged over the gold leaf classes. ``depth_acc`` is keyed by 1-based depth and
    ``node_acc`` by internal node id (``Taxonomy.ROOT_ID`` for the top-level decision); both
    node metrics and ``confusions`` are teacher-forced, i.e. measured with the correct parent.
    ``stop_rate`` counts sections with no label or a path that stopped before a leaf.
    """

    model_config = ConfigDict(extra="forbid")

    n: int
    exact: float
    hp: float
    hr: float
    hf1: float
    macro_p: float
    macro_r: float
    macro_f1: float
    depth_acc: dict[int, float] = Field(default_factory=dict)
    node_acc: dict[str, float] = Field(default_factory=dict)
    stop_rate: float
    selective_acc: float
    confusions: dict[str, list[tuple[str, str, int]]] = Field(default_factory=dict)
    sections_per_sec: float
    device: str
    meta: dict[str, Any] = Field(default_factory=dict)

    def to_json(self, indent: int | None = 2) -> str:
        """The report as JSON; ``Report.model_validate_json`` reads it back."""
        return self.model_dump_json(indent=indent)

    def _summary_rows(self) -> list[tuple[str, str]]:
        return [
            ("Sections", str(self.n)),
            ("Exact path accuracy", f"{self.exact:.3f}"),
            ("hP", f"{self.hp:.3f}"),
            ("hR", f"{self.hr:.3f}"),
            ("hF1", f"{self.hf1:.3f}"),
            ("Leaf macro-P", f"{self.macro_p:.3f}"),
            ("Leaf macro-R", f"{self.macro_r:.3f}"),
            ("Leaf macro-F1", f"{self.macro_f1:.3f}"),
            ("Stop rate", f"{self.stop_rate:.3f}"),
            ("Selective accuracy", f"{self.selective_acc:.3f}"),
            ("Sections/sec", f"{self.sections_per_sec:.1f}"),
            ("Device", self.device),
        ]

    def _skipped(self) -> list[str]:
        return list(self.meta.get("skipped_nodes", []))

    def print(self, console: Console | None = None) -> None:
        """Print the report as rich tables."""
        console = console or Console(highlight=False)
        # Text, not str: node ids and labels may contain "[...]" that rich would read as markup.
        summary = Table("Metric", "Value", title="Evaluation")
        for name, value in self._summary_rows():
            summary.add_row(Text(name), Text(value))
        console.print(summary)
        if self.depth_acc:
            table = Table("Depth", "Accuracy", title="Accuracy by depth")
            for depth, acc in self.depth_acc.items():
                table.add_row(Text(str(depth)), Text(f"{acc:.3f}"))
            console.print(table)
        if self.node_acc:
            table = Table("Node", "Accuracy", title="Node decision accuracy")
            for node_id, acc in self.node_acc.items():
                table.add_row(Text(node_id), Text(f"{acc:.3f}"))
            console.print(table)
        if self.confusions:
            table = Table("Node", "Gold", "Predicted", "Count", title="Top sibling confusions")
            for node_id, rows in self.confusions.items():
                for gold, predicted, count in rows:
                    table.add_row(Text(node_id), Text(gold), Text(predicted), Text(str(count)))
            console.print(table)
        if self._skipped():
            console.print(Text(f"skipped nodes (no backend): {', '.join(self._skipped())}"))

    def to_markdown(self) -> str:
        """The report as Markdown tables."""
        lines = ["| Metric | Value |", "| --- | --- |"]
        lines += [f"| {name} | {value} |" for name, value in self._summary_rows()]
        if self.depth_acc:
            lines += ["", "| Depth | Accuracy |", "| --- | --- |"]
            lines += [f"| {d} | {acc:.3f} |" for d, acc in self.depth_acc.items()]
        if self.node_acc:
            lines += ["", "| Node | Accuracy |", "| --- | --- |"]
            lines += [f"| {_cell(n)} | {acc:.3f} |" for n, acc in self.node_acc.items()]
        if self.confusions:
            lines += ["", "| Node | Gold | Predicted | Count |", "| --- | --- | --- | --- |"]
            for node_id, rows in self.confusions.items():
                lines += [
                    f"| {_cell(node_id)} | {_cell(g)} | {_cell(p)} | {c} |" for g, p, c in rows
                ]
        if self._skipped():
            lines += ["", f"Skipped nodes (no backend): {', '.join(map(_cell, self._skipped()))}"]
        return "\n".join(lines) + "\n"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _uses_model(clf: Classifier) -> bool:
    """True when some backend the classifier may route to runs a local model on a device."""
    specs: list[Any] = [clf.backend, clf.fallback, *clf.overrides.values()]
    specs += [n.backend for n in clf.taxonomy.nodes.values()]
    for spec in specs:
        if isinstance(spec, str):
            if spec.partition(":")[0].strip() in _MODEL_KINDS:
                return True
        elif spec is not None and (hasattr(spec, "_device_in_use") or hasattr(spec, "_device")):
            return True
    return False


def _teacher_forced(
    clf: Classifier, dataset: Dataset
) -> tuple[dict[str, float], dict[str, list[tuple[str, str]]], list[str]]:
    """Node accuracy, ``(gold, predicted)`` child pairs and skipped nodes, given the right parent.

    One ``decide`` call per node, batching every text whose gold path passes through it. A node
    that no backend can decide is skipped.
    """
    by_node: dict[str, tuple[list[str], list[str]]] = {}
    for example in dataset:
        parents = [Taxonomy.ROOT_ID, *example.path[:-1]]
        for parent, child in zip(parents, example.path, strict=True):
            texts, golds = by_node.setdefault(parent, ([], []))
            texts.append(example.text)
            golds.append(child)
    order = [Taxonomy.ROOT_ID, *clf.taxonomy.nodes]
    accuracy: dict[str, float] = {}
    pairs: dict[str, list[tuple[str, str]]] = {}
    skipped: list[str] = []
    for node_id in (n for n in order if n in by_node):
        texts, golds = by_node[node_id]
        try:
            child_ids, probs, _ = clf.decide(texts, node_id)
        except BackendError:
            skipped.append(node_id)
            continue
        predicted = [child_ids[int(i)] for i in probs.argmax(axis=1)]
        pairs[node_id] = list(zip(golds, predicted, strict=True))
        accuracy[node_id] = sum(g == p for g, p in pairs[node_id]) / len(golds)
    return accuracy, pairs, skipped


def evaluate(clf: Classifier, dataset: Dataset, *, node_decisions: bool = True) -> Report:
    """Score ``clf`` on ``dataset`` (paths given as ids or labels).

    Throughput is measured on a second pass over the texts, after a one-text warm-up call, so
    model loading is not counted. With ``node_decisions`` the per-node accuracy and sibling
    confusions are computed too, with one ``Classifier.decide`` call per node.
    """
    resolved = dataset.resolve(clf.taxonomy)
    texts = [e.text for e in resolved]
    gold = [list(e.path) for e in resolved]
    if texts:
        clf.classify_texts(texts[:1])  # warm-up: loads models, builds caches
    start = time.perf_counter()
    results = clf.classify_texts(texts) if texts else []
    elapsed = time.perf_counter() - start
    pred = [list(r.best.node_ids) if r.best is not None else [] for r in results]
    stopped = [is_stopped(r) for r in results]

    hp, hr, hf1 = hierarchical_prf(pred, gold)
    macro_p, macro_r, macro_f1 = leaf_prf(pred, gold)
    node_acc: dict[str, float] = {}
    confusions: dict[str, list[tuple[str, str, int]]] = {}
    meta: dict[str, Any] = {"strategy": type(clf.strategy).__name__}
    if node_decisions and texts:
        node_acc, pairs, skipped = _teacher_forced(clf, resolved)
        confusions = sibling_confusions(pairs)
        meta["skipped_nodes"] = skipped
    device = "cpu"
    if _uses_model(clf):
        from docrooter.backends.loading import select_device

        device = select_device()
    return Report(
        n=len(texts),
        exact=exact_path_accuracy(pred, gold),
        hp=hp,
        hr=hr,
        hf1=hf1,
        macro_p=macro_p,
        macro_r=macro_r,
        macro_f1=macro_f1,
        depth_acc=depth_accuracy(pred, gold),
        node_acc=node_acc,
        stop_rate=stop_rate(results),
        selective_acc=selective_accuracy(pred, gold, stopped),
        confusions=confusions,
        sections_per_sec=len(texts) / elapsed if texts and elapsed > 0 else 0.0,
        device=device,
        meta=meta,
    )


class Comparison:
    """Reports of several named classifiers on one dataset, side by side."""

    def __init__(self, reports: dict[str, Report]):
        self.reports = dict(reports)

    def _rows(self) -> list[tuple[str, list[str]]]:
        fields: list[tuple[str, Any]] = [
            ("Exact path accuracy", lambda r: f"{r.exact:.3f}"),
            ("hP", lambda r: f"{r.hp:.3f}"),
            ("hR", lambda r: f"{r.hr:.3f}"),
            ("hF1", lambda r: f"{r.hf1:.3f}"),
            ("Leaf macro-F1", lambda r: f"{r.macro_f1:.3f}"),
            ("Stop rate", lambda r: f"{r.stop_rate:.3f}"),
            ("Selective accuracy", lambda r: f"{r.selective_acc:.3f}"),
            ("Sections/sec", lambda r: f"{r.sections_per_sec:.1f}"),
            ("Device", lambda r: r.device),
        ]
        reports = list(self.reports.values())
        return [(name, [get(r) for r in reports]) for name, get in fields]

    def print(self, console: Console | None = None) -> None:
        """Print one column per classifier."""
        console = console or Console(highlight=False)
        table = Table(title="Comparison")
        table.add_column("Metric")
        for name in self.reports:
            table.add_column(Text(name))
        for name, values in self._rows():
            table.add_row(Text(name), *(Text(v) for v in values))
        console.print(table)

    def to_markdown(self) -> str:
        """One column per classifier."""
        header = ["Metric", *(_cell(name) for name in self.reports)]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("---" for _ in header) + " |",
        ]
        lines += [f"| {name} | " + " | ".join(values) + " |" for name, values in self._rows()]
        return "\n".join(lines) + "\n"


def compare(classifiers: dict[str, Classifier], dataset: Dataset) -> Comparison:
    """Evaluate every classifier on ``dataset``; ``Comparison.print()`` shows them side by side."""
    return Comparison({name: evaluate(clf, dataset) for name, clf in classifiers.items()})
