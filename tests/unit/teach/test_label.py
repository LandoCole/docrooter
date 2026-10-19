"""The labelling pipeline (spec §7.3–§7.4, §4.0 rulings 5–6): shortlist, teacher, weak labels.

Fakes only: a table-driven embedder, a recording teacher and, where the prompt text matters,
the real :class:`LogprobTeacher` over a character tokenizer. No torch, mlx or network.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from docrooter.backends.base import Candidate, Scores, candidate_text
from docrooter.corpus import Corpus, CorpusElement, LabelKey, ManifestRecord
from docrooter.taxonomies import builtin_taxonomy
from docrooter.taxonomy import Taxonomy
from docrooter.teach import LogprobTeacher, TeachItem, table_text
from docrooter.teach import prompts as prompts_module
from docrooter.teach.label import (
    PRESCREEN_LABELER,
    TEACHER_LABELER,
    WEAK_LABELER,
    SampleSpec,
    in_function_sample,
    label_corpus,
    sample_elements,
)
from docrooter.teach.shortlist import (
    MAX_QUERY_CHARS,
    element_query,
    function_scores,
    percentile_scores,
    prescreen_functions,
    shortlist,
)
from docrooter.teach.weak import EDGAR_10K_ITEMS, weak_labels

FUNCTION_TAGS = [n.id for n in builtin_taxonomy("element_functions").leaves()]


# --- fakes -------------------------------------------------------------------------------------


class WordEmbedder:
    """Similarity = how many words of a candidate's id occur in the text, minus a small
    tie-break by candidate position. Records every call."""

    name = "embedding:fake-words"
    temperature = 0.05

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[str], str | None]] = []

    def score(
        self,
        texts: Sequence[str],
        candidates: Sequence[Candidate],
        *,
        instruction: str | None = None,
        multi_label: bool = False,
    ) -> Scores:
        self.calls.append((list(texts), [c.node_id for c in candidates], instruction))
        rows = []
        for text in texts:
            words = set(text.lower().replace(".", " ").split())
            row = []
            for j, c in enumerate(candidates):
                parts = c.node_id.split(".")[-1].split("_")
                row.append(0.1 * sum(p in words for p in parts) - 0.001 * j)
            rows.append(row)
        return Scores(
            np.array(rows, dtype=float).reshape(len(texts), len(candidates)), "similarity"
        )


class TableEmbedder:
    """Similarity from a per-text table ``{candidate id: score}`` (0 when absent)."""

    name = "embedding:fake-table"
    temperature = 0.05

    def __init__(self, table: dict[str, dict[str, float]]) -> None:
        self.table = table
        self.texts: list[str] = []

    def score(self, texts, candidates, *, instruction=None, multi_label=False) -> Scores:
        self.texts.extend(texts)
        values = [[self.table.get(t, {}).get(c.node_id, 0.0) for c in candidates] for t in texts]
        return Scores(np.array(values, dtype=float), "similarity")


class RecordingTeacher:
    """Gives the first option 0.7 and shares 0.3 among the rest; P(yes) = 0.6. Records calls.
    ``fail_after`` raises once that many items have been scored (a crash mid-run); ``tick``
    advances ``clock`` by that many seconds per call."""

    name = "fake:teacher@1#native"

    def __init__(self, *, fail_after: int | None = None, clock: Any = None, tick: float = 0.0):
        self.choose_calls: list[dict[str, Any]] = []
        self.yes_no_calls: list[tuple[str, list[TeachItem]]] = []
        self.items = 0
        self.fail_after = fail_after
        self.clock, self.tick = clock, tick

    def _count(self, n: int) -> None:
        if self.fail_after is not None and self.items + n > self.fail_after:
            raise RuntimeError("teacher died")
        self.items += n
        if self.clock is not None:
            self.clock.now += self.tick

    def choose(self, items, *, instruction, prefix_key, prefix_options=None):
        self._count(len(items))
        self.choose_calls.append(
            {
                "items": list(items),
                "instruction": instruction,
                "prefix_key": prefix_key,
                "prefix_options": list(prefix_options) if prefix_options is not None else None,
            }
        )
        out = []
        for item in items:
            tags = [t for t, _ in item.options]
            rest = 0.3 / (len(tags) - 1) if len(tags) > 1 else 0.0
            out.append(
                {t: (0.7 if i == 0 else rest) if len(tags) > 1 else 1.0 for i, t in enumerate(tags)}
            )
        return out

    def yes_no(self, items, *, question):
        self._count(len(items))
        self.yes_no_calls.append((question, list(items)))
        return [0.6] * len(items)

    def close(self) -> None:
        pass

    def texts(self) -> list[str]:
        seen = [i.text for call in self.choose_calls for i in call["items"]]
        return seen + [i.text for _, items in self.yes_no_calls for i in items]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CharTokenizer:
    """One token per character, ChatML-like template (as in test_teach)."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize, **kw) -> str:
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return out + "<|im_start|>assistant\n"


class FlatRunner:
    """Every next token equally likely; records prefixes and suffixes as text."""

    def __init__(self) -> None:
        self.prefixes: list[str] = []
        self.suffixes: list[str] = []

    def prefill(self, ids):
        self.prefixes.append("".join(map(chr, ids)))
        return len(self.prefixes) - 1

    def score_next(self, state, ids, candidates):
        self.suffixes.append("".join(map(chr, ids)))
        return [-math.log(1 << 17)] * len(candidates)

    def close(self) -> None:
        pass


# --- corpus helpers ----------------------------------------------------------------------------


def _sha(doc_id: str) -> str:
    return hashlib.sha256(doc_id.encode()).hexdigest()


ANNUAL_TEXTS = [
    ("heading", ["Item 1A. Risk Factors"], "Item 1A. Risk Factors"),
    ("paragraph", ["Item 1A. Risk Factors"], "Our business faces risk from supply disruption."),
    ("paragraph", ["Item 1A. Risk Factors"], "We expect revenue to grow, though risk remains."),
    ("heading", ["Properties"], "Properties"),
    ("paragraph", ["Properties"], "We lease properties and offices in twelve countries."),
    ("list_item", ["Properties"], "Headquarters, Chicago, leased."),
    ("heading", ["Notes to the Financial Statements"], "Notes to the Financial Statements"),
    ("paragraph", ["Notes to the Financial Statements"], "Revenue is recognised over time."),
    ("table", ["Notes to the Financial Statements"], "Revenue 2025 2024 Total 10 9"),
    ("caption", ["Notes to the Financial Statements"], "Table 3. Revenue by year"),
]


def _elements(doc_id: str, rows) -> list[CorpusElement]:
    out = []
    for i, (kind, path, text) in enumerate(rows):
        out.append(
            CorpusElement(
                doc_id=doc_id,
                element_id=f"e{i}",
                kind=kind,
                text=f"{text} ({doc_id})" if kind != "heading" and text else text,
                heading_path=list(path),
                section_title=path[-1] if path else None,
                page=1,
                table_markdown="| Revenue | 2025 |\n| --- | --- |\n| Total | 10 |"
                if kind == "table"
                else None,
            )
        )
    return out


def _add(
    corpus: Corpus,
    doc_id: str,
    *,
    split="train",
    doc_type="financial.annual_report",
    rows=ANNUAL_TEXTS,
) -> None:
    record = ManifestRecord(id=doc_id, sha256=_sha(doc_id), split=split, doc_type_hint=doc_type)
    corpus.add_document(record, _elements(doc_id, rows), [1])


PAPER_TEXTS = [
    ("heading", ["Abstract"], "Abstract"),
    ("paragraph", ["Abstract"], "We propose a method for parsing documents."),
    ("heading", ["Introduction"], "1 Introduction"),
    ("paragraph", ["Introduction"], "Documents are hard; we introduce a parser."),
    ("paragraph", ["Related Work"], "Prior work used rules [3]."),
]


@pytest.fixture
def corpus(tmp_path) -> Corpus:
    c = Corpus(tmp_path / "store")
    for i in range(3):
        _add(c, f"ar{i}")
    _add(c, "paper0", doc_type="research.research_paper", rows=PAPER_TEXTS)
    _add(c, "paper1", split="calib", doc_type="research.research_paper", rows=PAPER_TEXTS)
    _add(c, "gold0", split="gold")
    _add(c, "gold1", split="gold", doc_type="research.research_paper", rows=PAPER_TEXTS)
    return c


def _sampled_count(corpus: Corpus) -> int:
    return len(sample_elements(corpus, SampleSpec(target=1000, floor=0)))


def _spec(**kw) -> SampleSpec:
    return SampleSpec(**{"target": 1000, "floor": 0, **kw})


# --- shortlist ---------------------------------------------------------------------------------


def _element(text: str, path=("Item 1A. Risk Factors",), kind="paragraph") -> CorpusElement:
    return CorpusElement(
        doc_id="d",
        element_id=text[:8].replace(" ", "_") or "x",
        kind=kind,
        text=text,
        heading_path=list(path),
        section_title=path[-1] if path else None,
        page=1,
    )


def test_shortlist_includes_other_and_top_k():
    leaves = [n.id for n in builtin_taxonomy("element_topics").children("annual_report")]
    first, second = _element("first element"), _element("second element")
    # first: other ranks last, so it is appended after the top 5; second: other is in the top 5
    table = {
        element_query(first): {leaf: 1.0 - 0.05 * i for i, leaf in enumerate(leaves[:7])},
        element_query(second): {
            "annual_report.other": 0.9,
            "annual_report.exhibits": 0.8,
            "annual_report.properties": 0.7,
            "annual_report.risk_factors": 0.6,
            "annual_report.sustainability": 0.5,
            "annual_report.legal_proceedings": 0.4,
        },
    }
    backend = TableEmbedder(table)
    one, two = shortlist([first, second], "annual_report", backend, k=5)
    assert one == leaves[:5] + ["annual_report.other"]
    assert two == [
        "annual_report.other",
        "annual_report.exhibits",
        "annual_report.properties",
        "annual_report.risk_factors",
        "annual_report.sustainability",
    ]
    # the embedder sees the heading path and the text
    assert backend.texts[0] == element_query(first)
    assert element_query(first).startswith("Item 1A. Risk Factors\n") and "first element" in (
        element_query(first)
    )
    (three,) = shortlist([first], "annual_report", backend, k=2)
    assert three == leaves[:2] + ["annual_report.other"]


def _sample_elements(n: int) -> list[CorpusElement]:
    return [_element(f"element number {i}", path=(f"Section {i}",)) for i in range(n)]


def test_prescreen_functions_top_k_by_percentile():
    elements = _sample_elements(5)
    queries = [element_query(e) for e in elements]
    table = {
        q: {"forward_looking": 0.1 * i, "procedure": 0.5 - 0.1 * i} for i, q in enumerate(queries)
    }
    table[queries[2]]["marketing"] = 0.9
    tops = prescreen_functions(elements, TableEmbedder(table), k=3)
    # element 4 has the sample's highest forward_looking cosine, element 0 the highest procedure
    assert tops[4][0] == ("forward_looking", 1.0)
    assert tops[0][0] == ("procedure", 1.0)
    assert tops[2][0] == ("marketing", 1.0)
    for top in tops:
        scores = [score for _, score in top]
        assert len(top) == 3 and scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
    assert percentile_scores(np.array([[0.3], [0.1], [0.3], [0.2]]))[:, 0].tolist() == [
        pytest.approx(5 / 6),
        0.0,
        pytest.approx(5 / 6),
        pytest.approx(1 / 3),
    ]
    assert percentile_scores(np.array([[0.4, 0.1]])).tolist() == [[0.5, 0.5]]


def test_two_related_tags_both_score_high():
    """An element that is both a risk disclosure and forward-looking scores high on both: the
    tags do not share a probability mass (the old 12-way softmax put the second below 0.2)."""
    elements = _sample_elements(10)
    queries = [element_query(e) for e in elements]
    table = {
        q: {"risk_disclosure": 0.30 + 0.01 * i, "forward_looking": 0.31 - 0.01 * i}
        for i, q in enumerate(queries)
    }
    both = queries[5]
    table[both] = {"risk_disclosure": 0.80, "forward_looking": 0.72}
    scores = function_scores(elements, TableEmbedder(table))
    assert scores[5]["risk_disclosure"] == 1.0 and scores[5]["forward_looking"] == 1.0
    (top,) = [prescreen_functions(elements, TableEmbedder(table), k=3)[5]]
    assert {tag for tag, _ in top[:2]} == {"risk_disclosure", "forward_looking"}


def test_prescreen_scores_are_independent_of_other_tags():
    elements = _sample_elements(8)
    queries = [element_query(e) for e in elements]
    base = {q: {"citation": 0.05 * ((i * 3) % 8)} for i, q in enumerate(queries)}
    noisy = {
        q: {
            **base[q],
            **{
                tag: 0.9 - 0.1 * ((i + j) % 8)
                for j, tag in enumerate(FUNCTION_TAGS)
                if tag != "citation"
            },
        }
        for i, q in enumerate(queries)
    }
    quiet = function_scores(elements, TableEmbedder(base))
    loud = function_scores(elements, TableEmbedder(noisy))
    assert [row["citation"] for row in quiet] == [row["citation"] for row in loud]
    assert sorted(row["citation"] for row in quiet) == pytest.approx([i / 7 for i in range(8)])


# --- the family prefix (ruling 5) ----------------------------------------------------------------


def test_family_prefix_lists_all_leaves_and_suffix_restricts_letters(corpus):
    topics = builtin_taxonomy("element_topics")
    teacher = RecordingTeacher()
    label_corpus(corpus, teacher, WordEmbedder(), sample=_spec(), functions_share=0.0)
    assert teacher.choose_calls and not teacher.yes_no_calls
    families_seen = set()
    for call in teacher.choose_calls:
        family = call["prefix_key"].split(":", 1)[1]
        families_seen.add(family)
        leaves = topics.children(family)
        assert call["prefix_options"] == [(n.id, candidate_text(n)) for n in leaves]
        order = [n.id for n in leaves]
        for item in call["items"]:
            tags = [t for t, _ in item.options]
            assert f"{family}.other" in tags and len(tags) <= 6
            assert set(tags) <= set(order)
            assert tags == sorted(tags, key=order.index)  # family order, not the embedder's rank
    assert families_seen == {"annual_report", "research_paper"}

    # with the real scorer: the prefix letters every leaf, the suffix names only the shortlist
    runner = FlatRunner()
    real = LogprobTeacher("fake:char@0", loader=lambda: (CharTokenizer(), runner))
    store = Corpus(corpus.dir.parent / "store2")
    _add(store, "ar9")
    label_corpus(store, real, WordEmbedder(), sample=_spec(), functions_share=0.0)
    leaves = topics.children("annual_report")
    assert len(runner.prefixes) == 1
    assert (
        prompts_module.options_block([(n.id, candidate_text(n)) for n in leaves])
        in (runner.prefixes[0])
    )
    rows = [r for r in store.labels(TEACHER_LABELER).values() if r.facet == "topic"]
    assert len(rows) == len(runner.suffixes)
    named = []
    for suffix in runner.suffixes:
        letters = suffix.rsplit("Answer with one letter: ", 1)[1].split(".")[0]
        named.append(sorted(letters.replace(" or ", ", ").split(", ")))
    position = {n.id: prompts_module.LETTERS[i] for i, n in enumerate(leaves)}
    expected = sorted(sorted(position[t] for t in r.probs) for r in rows)
    assert sorted(named) == expected
    for row in rows:
        assert sum(row.probs.values()) == pytest.approx(1.0)
        assert row.provenance["teacher"] == "fake:char@0"


# --- functions: top 3 on half the sample (ruling 5) ---------------------------------------------


def test_function_questions_top3_on_half_sample(tmp_path):
    store = Corpus(tmp_path / "s")
    for i in range(8):
        _add(store, f"ar{i}")
    teacher = RecordingTeacher()
    report = label_corpus(store, teacher, WordEmbedder(), sample=_spec(), functions_share=0.5)
    picks = sample_elements(store, _spec())
    asked = {(p.element.doc_id, p.element.element_id) for p in picks if p.functions}
    assert asked == {
        (p.element.doc_id, p.element.element_id)
        for p in picks
        if in_function_sample(p.element.doc_id, p.element.element_id, 0.5)
    }
    assert 0.3 * len(picks) <= len(asked) <= 0.7 * len(picks)
    teacher_rows = store.labels(TEACHER_LABELER)
    prescreen = store.labels(PRESCREEN_LABELER)
    function_rows = {
        (k.doc_id, k.element_id): r for k, r in teacher_rows.items() if k.facet == "function"
    }
    assert set(function_rows) == asked
    for (doc_id, element_id), row in function_rows.items():
        scores = prescreen[
            next(k for k in prescreen if (k.doc_id, k.element_id) == (doc_id, element_id))
        ]
        assert set(scores.probs) == set(FUNCTION_TAGS)
        top3 = sorted(scores.probs, key=lambda t: (-scores.probs[t], FUNCTION_TAGS.index(t)))[:3]
        assert list(row.probs) == top3
        assert row.provenance["asked"] == top3
        assert all(p == pytest.approx(0.6) for p in row.probs.values())
    # every sampled element has prescreen scores (implicit negatives, ruling 6)
    assert len(prescreen) == len(picks)
    # grouped by question: one yes/no call per question in the single batch
    questions = [q for q, _ in teacher.yes_no_calls]
    assert len(questions) == len(set(questions))
    assert sum(len(items) for _, items in teacher.yes_no_calls) == 3 * len(asked)
    assert report.function_labelled == len(asked)
    assert report.teacher_items == len(picks) + 3 * len(asked)


# --- the wall-clock ceiling ----------------------------------------------------------------------


def test_ceiling_stops_and_reports(corpus):
    clock = FakeClock()
    teacher = RecordingTeacher(clock=clock, tick=3600.0)  # every teacher call takes an hour
    report = label_corpus(
        corpus,
        teacher,
        WordEmbedder(),
        sample=SampleSpec(target=1000, floor=20),
        ceiling_hours=1.5,
        functions_share=0.0,
        batch_size=4,
        clock=clock,
    )
    total = _sampled_count(corpus)
    assert total > 8
    assert report.stopped_by_ceiling
    assert report.sampled == total
    # the first batch ran (it went past the ceiling); the check before the second stopped it
    assert report.topic_labelled == 4
    assert len([k for k in corpus.labels(TEACHER_LABELER) if k.facet == "topic"]) == 4
    assert report.below_floor  # 4 < 20
    assert report.seconds >= 1.5 * 3600
    unlimited = label_corpus(
        corpus, RecordingTeacher(), WordEmbedder(), sample=_spec(), functions_share=0.0
    )
    assert not unlimited.stopped_by_ceiling and not unlimited.below_floor
    assert unlimited.skipped == 4 and unlimited.topic_labelled == total - 4


# --- resume (Review Focus 5) -------------------------------------------------------------------


def test_label_run_resumes(corpus):
    total = _sampled_count(corpus)
    first = RecordingTeacher(fail_after=9)  # dies in the second batch of 6 topic items
    with pytest.raises(RuntimeError, match="teacher died"):
        label_corpus(
            corpus, first, WordEmbedder(), sample=_spec(), batch_size=6, functions_share=0.0
        )
    done = {k for k in corpus.labels(TEACHER_LABELER) if k.facet == "topic"}
    assert len(done) == 6  # the finished batch was written; the broken one was not

    second = RecordingTeacher()
    report = label_corpus(
        corpus, second, WordEmbedder(), sample=_spec(), batch_size=6, functions_share=0.0
    )
    assert second.items == total - 6  # nothing finished is scored again
    assert report.skipped == 6 and report.topic_labelled == total - 6
    keys = {k for k in corpus.labels(TEACHER_LABELER) if k.facet == "topic"}
    assert len(keys) == total
    # every label line is whole, and no key was written twice
    lines = (corpus.dir / "labels" / f"{TEACHER_LABELER}.jsonl").read_text().splitlines()
    assert len(lines) == total

    # the weak and prescreen rows of the batch that died were written once and kept
    labels_dir = corpus.dir / "labels"
    prescreen_lines = (labels_dir / f"{PRESCREEN_LABELER}.jsonl").read_text().splitlines()
    weak_lines = (labels_dir / f"{WEAK_LABELER}.jsonl").read_text().splitlines()
    assert len(prescreen_lines) == total

    third = RecordingTeacher()
    again = label_corpus(corpus, third, WordEmbedder(), sample=_spec(), functions_share=0.0)
    assert third.items == 0 and again.skipped == total and again.topic_labelled == 0
    assert (labels_dir / f"{PRESCREEN_LABELER}.jsonl").read_text().splitlines() == prescreen_lines
    assert (labels_dir / f"{WEAK_LABELER}.jsonl").read_text().splitlines() == weak_lines


def test_function_resume_reuses_stored_cosines(corpus):
    total = _sampled_count(corpus)
    label_corpus(corpus, RecordingTeacher(), WordEmbedder(), sample=_spec(), functions_share=0.0)
    labels_dir = corpus.dir / "labels"
    prescreen_lines = (labels_dir / f"{PRESCREEN_LABELER}.jsonl").read_text().splitlines()
    embedder = WordEmbedder()
    teacher = RecordingTeacher()
    report = label_corpus(corpus, teacher, embedder, sample=_spec(), functions_share=1.0)
    assert report.function_labelled == total and report.topic_labelled == 0
    assert report.embedded == 0 and not embedder.calls  # cosines came from the prescreen rows
    assert (labels_dir / f"{PRESCREEN_LABELER}.jsonl").read_text().splitlines() == prescreen_lines
    prescreen = corpus.labels(PRESCREEN_LABELER)
    for key, row in corpus.labels(TEACHER_LABELER).items():
        if key.facet != "function":
            continue
        scores = prescreen[key].probs
        top3 = sorted(scores, key=lambda t: (-scores[t], FUNCTION_TAGS.index(t)))[:3]
        assert row.provenance["asked"] == top3


# --- the gold split ----------------------------------------------------------------------------


def test_never_labels_gold_split(corpus):
    teacher = RecordingTeacher()
    embedder = WordEmbedder()
    label_corpus(corpus, teacher, embedder, sample=_spec(), functions_share=1.0)
    # headings repeat across documents; the other texts name their document
    gold_texts = {e.text for e in corpus.iter_elements(split="gold") if "(gold" in e.text}
    assert gold_texts
    assert not (set(teacher.texts()) & gold_texts)
    assert not any("(gold" in t for t in teacher.texts())
    embedded = {t for texts, _, _ in embedder.calls for t in texts}
    assert embedded and not any("(gold" in t for t in embedded)
    for labeler in corpus.labelers():
        assert not any(k.doc_id.startswith("gold") for k in corpus.labels(labeler))
    picks = sample_elements(corpus, _spec())
    assert picks and not any(p.element.doc_id.startswith("gold") for p in picks)
    with pytest.raises(ValueError, match="gold"):
        SampleSpec(splits=("train", "gold"))


# --- weak labels -------------------------------------------------------------------------------


def _record(doc_type: str | None = "financial.annual_report") -> ManifestRecord:
    return ManifestRecord(id="d", sha256=_sha("d"), doc_type_hint=doc_type)


def test_weak_labels_from_headings():
    record = _record()
    risk = weak_labels(_element("x", path=("PART I", "Item 1A. Risk Factors")), record)
    assert risk["edgar_item"]["label"] == "annual_report.risk_factors"
    assert risk["edgar_item"]["item"] == "1A" and risk["edgar_item"]["source"] == "edgar_item"
    assert risk["section_norm"]["label"] == "annual_report.risk_factors"
    assert risk["section_norm"]["confidence"] == 1.0
    assert risk["section_norm"]["source"] == "heading_synonym"
    notes = weak_labels(_element("x", path=("Notes to the Financial Statements", "Leases")), record)
    assert notes["section_norm"]["label"] == "annual_report.notes_to_financial_statements"
    assert notes["section_norm"]["heading"] == "Notes to the Financial Statements"
    assert "edgar_item" not in notes
    assert weak_labels(_element("x", path=("Revenue by geography",)), record) == {}
    assert weak_labels(_element("x", path=()), record) == {}
    # a partial match below 0.5 is not recorded
    assert "section_norm" not in weak_labels(
        _element("x", path=("Business travel policy",)), _record("legal.contract")
    )
    # EDGAR items only for 10-K-style families
    paper = weak_labels(
        _element("x", path=("Item 7. Results",)), _record("research.research_paper")
    )
    assert "edgar_item" not in paper
    # every "Item N." heading synonym in the taxonomy maps to its own leaf
    for leaf in builtin_taxonomy("element_topics").children("annual_report"):
        for example in leaf.examples:
            if example.startswith("Item "):
                item = example.split()[1].rstrip(".")
                assert EDGAR_10K_ITEMS[item] == leaf.id.split(".", 1)[1], example


def test_weak_labels_recorded_not_mixed(corpus):
    teacher = RecordingTeacher()
    label_corpus(corpus, teacher, WordEmbedder(), sample=_spec(), functions_share=0.5)
    weak = corpus.labels(WEAK_LABELER)
    assert weak
    facets = {k.facet for k in weak}
    assert facets <= {"section_norm", "edgar_item"} and "edgar_item" in facets
    for row in weak.values():
        assert row.provenance["source"] in {"heading_synonym", "edgar_item"}
        assert len(row.probs) == 1
    risk = [r for k, r in weak.items() if k.facet == "edgar_item"]
    assert all(set(r.probs) == {"annual_report.risk_factors"} for r in risk)
    # the teacher's rows hold only the teacher's distributions, over its shortlist
    teacher_rows = corpus.labels(TEACHER_LABELER)
    assert {k.facet for k in teacher_rows} == {"topic", "function"}
    for key, row in teacher_rows.items():
        assert "source" not in row.provenance
        if key.facet == "topic":
            shortlisted = [t for t, _ in row.provenance["shortlist"]]
            assert set(row.probs) == set(shortlisted)
            assert max(row.probs.values()) == pytest.approx(0.7)  # as the teacher said
    # the weak labeller is separate from the teacher and the prescreen
    assert len({TEACHER_LABELER, PRESCREEN_LABELER, WEAK_LABELER}) == 3


# --- Review Focus 2: prompt and taxonomy in the cache key --------------------------------------


def _edited(taxonomy: Taxonomy, node_id: str) -> Taxonomy:
    data = taxonomy.to_dict()
    for node in data["nodes"]:
        if node["id"] == node_id:
            node["description"] += " Edited."
    return Taxonomy.from_dict(data)


def test_cache_key_includes_prompt_and_taxonomy_hash(corpus, monkeypatch):
    total = _sampled_count(corpus)
    label_corpus(corpus, RecordingTeacher(), WordEmbedder(), sample=_spec(), functions_share=1.0)
    same = RecordingTeacher()
    label_corpus(corpus, same, WordEmbedder(), sample=_spec(), functions_share=1.0)
    assert same.items == 0

    topics = _edited(builtin_taxonomy("element_topics"), "annual_report.properties")
    retopic = RecordingTeacher()
    report = label_corpus(
        corpus, retopic, WordEmbedder(), sample=_spec(), functions_share=1.0, topics=topics
    )
    assert report.topic_labelled == total and not retopic.yes_no_calls

    functions = _edited(builtin_taxonomy("element_functions"), "marketing")
    refunction = RecordingTeacher()
    report = label_corpus(
        corpus,
        refunction,
        WordEmbedder(),
        sample=_spec(),
        functions_share=1.0,
        topics=topics,
        functions=functions,
    )
    assert report.function_labelled == total and not refunction.choose_calls

    monkeypatch.setattr(
        prompts_module,
        "CHOOSE_FAMILY_SYSTEM",
        prompts_module.CHOOSE_FAMILY_SYSTEM + "\nBe careful.",
    )
    reprompt = RecordingTeacher()
    report = label_corpus(
        corpus,
        reprompt,
        WordEmbedder(),
        sample=_spec(),
        functions_share=1.0,
        topics=topics,
        functions=functions,
    )
    assert report.topic_labelled == total and report.function_labelled == total
    # the newest row wins, and its provenance names the prompt it was made with
    row = next(r for k, r in corpus.labels(TEACHER_LABELER).items() if k.facet == "topic")
    assert row.provenance["taxonomy_hash"] == topics.content_hash()
    assert row.provenance["label_key"] and row.provenance["prompt_hash"]


# --- Review Focus 3: long elements and tables ---------------------------------------------------


def test_truncation_and_table_inputs(tmp_path):
    store = Corpus(tmp_path / "s")
    long_text = "risk " * 20000
    rows = ANNUAL_TEXTS[6:] + [("paragraph", ["Item 1A. Risk Factors"], long_text)]
    _add(store, "t0", rows=rows)
    teacher = RecordingTeacher()
    embedder = WordEmbedder()
    label_corpus(store, teacher, embedder, sample=_spec(), functions_share=0.0)
    items = [i for call in teacher.choose_calls for i in call["items"]]
    table = next(e for e in store.iter_elements() if e.kind == "table")
    caption = next(e for e in store.iter_elements() if e.kind == "caption")
    table_item = next(i for i in items if "| Revenue |" in i.text)
    assert table_item.text == table_text(caption.text, table.table_markdown or "")
    assert "Element: table" in table_item.context
    assert "Section: Notes to the Financial Statements" in table_item.context
    # the embedder query is cut; the teacher gets the text and cuts it to its token limit
    assert all(len(t) <= MAX_QUERY_CHARS for texts, _, _ in embedder.calls for t in texts)
    # with the real scorer, the long element is cut to MAX_TEXT_TOKENS and nothing crashes
    runner = FlatRunner()
    real = LogprobTeacher("fake:char@0", loader=lambda: (CharTokenizer(), runner))
    store2 = Corpus(tmp_path / "s2")
    _add(store2, "t1", rows=rows)
    label_corpus(store2, real, WordEmbedder(), sample=_spec(), functions_share=1.0)
    limit = prompts_module.MAX_TEXT_TOKENS + prompts_module.MAX_CONTEXT_TOKENS + 200
    assert runner.suffixes and all(len(s) <= limit for s in runner.suffixes)
    assert len(store2.labels(TEACHER_LABELER)) == 2 * _sampled_count(store2)


# --- the sample (spec §7.4) ---------------------------------------------------------------------


def test_sample_is_capped_per_document_and_stratified(tmp_path):
    store = Corpus(tmp_path / "s")
    many = [("paragraph", ["Body"], f"Paragraph {i}.") for i in range(80)]
    many += [("heading", ["Body"], f"Heading {i}") for i in range(5)]
    many += [("table", ["Body"], f"Cell {i}") for i in range(2)]
    many += [("figure", ["Body"], "")]  # nothing to label
    for i in range(3):
        _add(store, f"ar{i}", rows=many)
    _add(store, "rp0", doc_type="research.research_paper", rows=many)
    picks = sample_elements(store, SampleSpec(per_doc=60, target=1000, floor=0))
    per_doc: dict[str, int] = {}
    for p in picks:
        per_doc[p.element.doc_id] = per_doc.get(p.element.doc_id, 0) + 1
    assert per_doc == {"ar0": 60, "ar1": 60, "ar2": 60, "rp0": 60}
    assert all(p.element.text.strip() for p in picks)
    # a short target keeps every kind and both families
    few = sample_elements(store, SampleSpec(per_doc=60, target=24, floor=0))
    assert len(few) == 24
    assert {p.element.kind for p in few} == {"paragraph", "heading", "table"}
    assert {p.family for p in few} == {"annual_report", "research_paper"}
    counts: dict[str, int] = {}
    for p in few:
        counts[p.family] = counts.get(p.family, 0) + 1
    assert counts["research_paper"] >= 10  # families take turns, not documents
    # the order is stable
    assert [p.element.element_id for p in few] == [
        p.element.element_id for p in sample_elements(store, SampleSpec(target=24, floor=0))
    ]
    with pytest.raises(ValueError):
        SampleSpec(target=10, floor=20)
    with pytest.raises(ValueError):
        SampleSpec(per_doc=0)


def test_label_keys_are_distinct_per_facet(corpus):
    label_corpus(corpus, RecordingTeacher(), WordEmbedder(), sample=_spec(), functions_share=1.0)
    rows = corpus.labels(TEACHER_LABELER)
    some = next(iter(rows))
    topic = rows[LabelKey(some.doc_id, some.element_id, some.text_sha1, "topic")]
    function = rows[LabelKey(some.doc_id, some.element_id, some.text_sha1, "function")]
    assert topic.provenance["label_key"] != function.provenance["label_key"]
    for row in (topic, function):
        prov = row.provenance
        assert prov["teacher"] == RecordingTeacher.name
        assert prov["engine"] == "fake" and prov["revision"] == "1" and prov["dtype"] == "native"
        assert prov["embedder"].startswith("embedding:fake-words")
        assert prov["timestamp"] and prov["taxonomy_hash"] and prov["prompt_hash"]


class _ClosingEmbedder(WordEmbedder):
    closed = False

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model = model

    def close(self) -> None:
        type(self).closed = True


def test_cli_teach_label(corpus, monkeypatch):
    import json as json_module

    from typer.testing import CliRunner

    from docrooter.backends import embedding as embedding_module
    from docrooter.cli import app
    from docrooter.teach import mlx as mlx_module

    monkeypatch.setattr(_ClosingEmbedder, "closed", False)
    specs: list[str] = []

    def fake_get_teacher(spec: str) -> RecordingTeacher:
        specs.append(spec)
        return RecordingTeacher()

    monkeypatch.setattr("docrooter.teach.get_teacher", fake_get_teacher)
    monkeypatch.setattr(embedding_module, "EmbeddingBackend", _ClosingEmbedder)
    result = CliRunner().invoke(
        app, ["teach", "label", str(corpus.dir), "--target", "7", "--functions-share", "0"]
    )
    assert result.exit_code == 0, result.output
    report = json_module.loads(result.stdout)
    assert report["sampled"] == 7 and report["topic_labelled"] == 7
    assert report["floor"] == 7  # the floor is capped at the target
    assert specs == [mlx_module.DEFAULT_TEACHER] and _ClosingEmbedder.closed
    missing = CliRunner().invoke(app, ["teach", "label", str(corpus.dir / "nowhere")])
    assert missing.exit_code == 1 and "no corpus" in missing.output


def test_embedder_released_before_the_teacher_runs(corpus):
    log: list[str] = []

    class LoggingEmbedder(WordEmbedder):
        def score(self, texts, candidates, **kw):
            log.append("embed")
            return super().score(texts, candidates, **kw)

        def close(self) -> None:
            log.append("close")

    class LoggingTeacher(RecordingTeacher):
        def choose(self, items, **kw):
            log.append("teacher")
            return super().choose(items, **kw)

        def yes_no(self, items, **kw):
            log.append("teacher")
            return super().yes_no(items, **kw)

    label_corpus(
        corpus,
        LoggingTeacher(),
        LoggingEmbedder(),
        sample=_spec(),
        batch_size=4,
        functions_share=0.5,
    )
    assert log.count("close") == 1 and "teacher" in log
    assert log.index("close") > max(i for i, e in enumerate(log) if e == "embed")
    assert log.index("close") < log.index("teacher")
    log.clear()
    label_corpus(
        corpus, LoggingTeacher(), LoggingEmbedder(), sample=_spec(), release_embedder=False
    )
    assert "close" not in log


def test_cli_closes_the_teacher_when_the_embedder_fails(corpus, monkeypatch):
    from typer.testing import CliRunner

    from docrooter.backends import embedding as embedding_module
    from docrooter.cli import app

    closed: list[bool] = []

    class ClosingTeacher(RecordingTeacher):
        def close(self) -> None:
            closed.append(True)

    def broken_embedder(model: str) -> None:
        raise ValueError("no such embedder")

    monkeypatch.setattr("docrooter.teach.get_teacher", lambda spec: ClosingTeacher())
    monkeypatch.setattr(embedding_module, "EmbeddingBackend", broken_embedder)
    result = CliRunner().invoke(app, ["teach", "label", str(corpus.dir)])
    assert result.exit_code == 1 and "no such embedder" in result.output
    assert closed == [True]
