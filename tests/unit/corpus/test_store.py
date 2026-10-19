"""The corpus store (spec §6): the manifest, per-document element files, label files and stats,
with atomic, resumable writes (Review Focus 5)."""

import json

import pytest
from pydantic import ValidationError

from docrooter.corpus import (
    Corpus,
    CorpusElement,
    LabelKey,
    LabelRow,
    ManifestRecord,
    read_manifest,
    text_sha1,
)
from docrooter.corpus import store as store_module
from docrooter.errors import DocrooterError


def _sha(n):
    return f"{n:064x}"


def _record(doc_id="a", sha=1, split="train", **extra):
    return ManifestRecord(
        id=doc_id,
        url=f"https://example.org/{doc_id}.pdf",
        source="edgar_ars",
        licence="public-domain",
        sha256=_sha(sha),
        bytes=1000,
        pages=3,
        fetched_at="2026-10-19T00:00:00Z",
        doc_type_hint="financial.annual_report",
        family_hint="annual_report",
        split=split,
        **extra,
    )


def _elements(doc_id, specs):
    """``specs``: ``(kind, text)`` pairs; ids e0, e1, … and a two-level heading path."""
    return [
        CorpusElement(
            doc_id=doc_id,
            element_id=f"e{k}",
            kind=kind,
            text=text,
            heading_path=["Report", "Risks"],
            section_title="Risks",
            page=1 + k // 2,
        )
        for k, (kind, text) in enumerate(specs)
    ]


BODY = [("heading", "Risks"), ("paragraph", "Rates may rise."), ("table", "a b 1 2")]


def test_text_sha1_and_element_validation():
    (element,) = _elements("a", [("paragraph", "Rates may rise.")])
    assert element.text_sha1 == text_sha1("Rates may rise.")
    assert len(element.text_sha1) == 40
    with pytest.raises(ValidationError, match="text_sha1"):
        CorpusElement(doc_id="a", element_id="e0", kind="paragraph", text="x", text_sha1="0" * 40,
                      heading_path=[], section_title=None, page=None)  # fmt: skip
    with pytest.raises(ValidationError):
        CorpusElement(doc_id="a", element_id="e0", kind="nonsense", text="x", heading_path=[],
                      section_title=None, page=None)  # fmt: skip


def test_add_is_idempotent_by_sha256(tmp_path):
    corpus = Corpus(tmp_path / "c")
    assert corpus.add_document(_record("a", 1), _elements("a", BODY), [1, 2]) is True
    doc_file = tmp_path / "c" / "docs" / "a.json"
    before = doc_file.read_bytes()

    # The same document again, and the same bytes under another id (a re-hosted copy): no-ops.
    assert corpus.add_document(_record("a", 1), _elements("a", BODY[:1]), [1]) is False
    assert corpus.add_document(_record("copy", 1), _elements("copy", BODY), [1]) is False
    assert doc_file.read_bytes() == before
    assert not (tmp_path / "c" / "docs" / "copy.json").exists()
    assert [r.id for r in corpus.manifest()] == ["a"]
    assert len((tmp_path / "c" / "manifest.jsonl").read_text().splitlines()) == 1
    assert corpus.has("a") and not corpus.has("copy")
    assert corpus.has_sha256(_sha(1)) and not corpus.has_sha256(_sha(2))

    # A fresh store over the same directory sees the same thing.
    again = Corpus(tmp_path / "c")
    assert again.add_document(_record("a", 1), _elements("a", BODY), [1, 2]) is False
    assert [e.element_id for e in again.iter_elements()] == ["e0", "e1", "e2"]

    # The same id with other bytes is a conflict, not an update.
    with pytest.raises(ValueError, match="already in the corpus"):
        again.add_document(_record("a", 2), _elements("a", BODY), [1])
    # Ids differing only in case would share a file on a case-insensitive file system.
    with pytest.raises(ValueError, match="case"):
        again.add_document(_record("A", 3), _elements("A", BODY), [1])


def test_add_checks_its_arguments(tmp_path):
    corpus = Corpus(tmp_path)
    with pytest.raises(ValueError, match="doc_id"):
        corpus.add_document(_record("a", 1), _elements("b", BODY), [1])
    with pytest.raises(ValueError, match="element id"):
        corpus.add_document(_record("a", 1), _elements("a", BODY) * 2, [1])
    with pytest.raises(ValueError, match="pages"):
        corpus.add_document(_record("a", 1), _elements("a", BODY), [0])
    assert corpus.manifest() == [] and not (tmp_path / "docs" / "a.json").exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".hidden", "a b", "x" * 200, "c:\\x"])
def test_unsafe_ids_and_labelers_are_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        _record(bad, 1)
    with pytest.raises(ValueError, match="labeler"):
        Corpus(tmp_path).write_labels(bad, [])
    with pytest.raises(ValueError, match="labeler"):
        Corpus(tmp_path).labels(bad)


def test_resume_after_partial_write(tmp_path, monkeypatch):
    """Review Focus 5: a crash between the doc file and its manifest line leaves no orphan
    visible to ``iter_elements``; adding the document again repairs it."""
    directory = tmp_path / "c"
    corpus = Corpus(directory)
    corpus.add_document(_record("a", 1), _elements("a", BODY), [1])

    real_append = store_module._append_lines

    def crash(path, lines):
        if path.name == "manifest.jsonl":
            raise KeyboardInterrupt("killed between the doc file and the manifest line")
        return real_append(path, lines)

    monkeypatch.setattr(store_module, "_append_lines", crash)
    with pytest.raises(KeyboardInterrupt):
        corpus.add_document(_record("b", 2), _elements("b", BODY), [1, 2])
    assert (directory / "docs" / "b.json").exists()  # the doc file was written ...
    monkeypatch.setattr(store_module, "_append_lines", real_append)

    resumed = Corpus(directory)  # ... but a restarted run does not see it
    assert not resumed.has("b") and not resumed.has_sha256(_sha(2))
    assert [r.id for r in resumed.manifest()] == ["a"]
    assert {e.doc_id for e in resumed.iter_elements()} == {"a"}
    assert resumed.stats()["documents"] == 1

    # A torn manifest line (the process died mid-append) and a stray temp file are invisible.
    with (directory / "manifest.jsonl").open("ab") as f:
        f.write(b'{"id": "b", "sha256": "00')
    (directory / "docs" / ".b.json.123.tmp").write_text("{partial")
    resumed = Corpus(directory)
    assert [r.id for r in resumed.manifest()] == ["a"]
    assert {e.doc_id for e in resumed.iter_elements()} == {"a"}

    # Re-adding repairs: the doc file is rewritten, the torn tail dropped, the line appended.
    assert resumed.add_document(_record("b", 2), _elements("b", BODY[:2]), [1]) is True
    lines = (directory / "manifest.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["a", "b"]
    final = Corpus(directory)
    assert [(e.doc_id, e.element_id) for e in final.iter_elements()] == [
        ("a", "e0"), ("a", "e1"), ("a", "e2"), ("b", "e0"), ("b", "e1"),
    ]  # fmt: skip
    assert json.loads((directory / "docs" / "b.json").read_text())["pages_read"] == [1]


def test_doc_file_is_replaced_atomically(tmp_path, monkeypatch):
    """A crash while writing a doc file never leaves a half-written ``<id>.json``."""
    corpus = Corpus(tmp_path)

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(store_module.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        corpus.add_document(_record("a", 1), _elements("a", BODY), [1])
    assert not (tmp_path / "docs" / "a.json").exists()
    assert list((tmp_path / "docs").iterdir()) == []  # the temp file is cleaned up
    assert Corpus(tmp_path).manifest() == []


def test_iter_elements_filters_split_and_kind(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.add_document(_record("t", 1, "train"), _elements("t", BODY), [1, 2])
    corpus.add_document(_record("c", 2, "calib"), _elements("c", BODY), [1])
    corpus.add_document(_record("g", 3, "gold"), _elements("g", BODY), [1])

    def ids(**kwargs):
        return [(e.doc_id, e.element_id) for e in corpus.iter_elements(**kwargs)]

    assert len(ids()) == 9
    assert ids(split="gold") == [("g", "e0"), ("g", "e1"), ("g", "e2")]
    assert ids(split=["train", "calib"], kinds=["paragraph"]) == [("t", "e1"), ("c", "e1")]
    assert ids(kinds={"table", "heading"}) == [
        ("t", "e0"), ("t", "e2"), ("c", "e0"), ("c", "e2"), ("g", "e0"), ("g", "e2"),
    ]  # fmt: skip
    assert ids(split="gold", kinds=[]) == []
    element = next(corpus.iter_elements(split="calib", kinds=["table"]))
    assert element == CorpusElement(
        doc_id="c", element_id="e2", kind="table", text="a b 1 2", text_sha1=text_sha1("a b 1 2"),
        heading_path=["Report", "Risks"], section_title="Risks", page=2,
    )  # fmt: skip
    with pytest.raises(ValueError, match="split"):
        list(corpus.iter_elements(split="test"))
    with pytest.raises(ValueError, match="kind"):
        list(corpus.iter_elements(kinds=["para"]))


def test_a_doc_file_that_does_not_match_its_record_is_an_error(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.add_document(_record("a", 1), _elements("a", BODY), [1])
    path = tmp_path / "docs" / "a.json"
    data = json.loads(path.read_text())
    data["sha256"] = _sha(9)
    path.write_text(json.dumps(data))
    with pytest.raises(DocrooterError, match="a.json"):
        list(Corpus(tmp_path).iter_elements())


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.pop("elements"),
        lambda d: d.pop("pages_read"),
        lambda d: d["elements"][0].pop("kind"),
        lambda d: d["elements"][0].update(kind="nonsense"),
        lambda d: d["elements"].append("not an element"),
    ],
)
def test_a_malformed_doc_file_is_a_docrooter_error_naming_it(tmp_path, damage):
    corpus = Corpus(tmp_path)
    corpus.add_document(_record("a", 1), _elements("a", BODY), [1])
    path = tmp_path / "docs" / "a.json"
    data = json.loads(path.read_text())
    damage(data)
    path.write_text(json.dumps(data))
    with pytest.raises(DocrooterError, match="a.json"):
        list(Corpus(tmp_path).iter_elements())
    with pytest.raises(DocrooterError, match="a.json"):
        Corpus(tmp_path).stats()


def test_failures_and_attempt_notes(tmp_path):
    corpus = Corpus(tmp_path)
    assert corpus.failures() == {} and corpus.pending_attempt() is None
    corpus.start_attempt("a")
    assert Corpus(tmp_path).pending_attempt() == "a"  # durable: another store object sees it
    corpus.finish_attempt()
    corpus.finish_attempt()  # clearing twice is fine
    assert corpus.pending_attempt() is None
    corpus.record_failure("a", "DocrooterError: bad")
    corpus.record_failure("b", "crashed previous run")
    corpus.record_failure("a", "MemoryError")
    failures = Corpus(tmp_path).failures()
    assert list(failures) == ["a", "b"] and failures["a"]["reason"] == "MemoryError"
    assert failures["b"]["timestamp"].endswith("+00:00")
    with pytest.raises(ValueError, match="document id"):
        corpus.start_attempt("../x")


def test_labels_keyed_by_doc_element_textsha(tmp_path):
    corpus = Corpus(tmp_path)
    corpus.add_document(_record("a", 1), _elements("a", BODY), [1])
    elements = list(corpus.iter_elements())
    provenance = {"engine": "mlx", "model": "m", "revision": "r", "prompt_hash": "p",
                  "taxonomy_hash": "t", "timestamp": "2026-10-19T00:00:00Z"}  # fmt: skip

    def row(element, facet="topic", probs=None, sha=None):
        return LabelRow(
            doc_id=element.doc_id,
            element_id=element.element_id,
            text_sha1=sha or element.text_sha1,
            facet=facet,
            probs=probs or {"annual_report.risk": 0.9, "annual_report.other": 0.1},
            provenance=provenance,
        )

    assert corpus.labels("teacher") == {}
    assert corpus.write_labels("teacher", [row(e) for e in elements]) == 3
    corpus.write_labels(
        "teacher",
        [
            row(elements[1], probs={"annual_report.risk": 0.2, "annual_report.other": 0.8}),
            row(elements[1], facet="function", probs={"risk_disclosure": 0.7}),
            row(elements[2], sha="f" * 40),  # the element's text changed: another key
        ],
    )
    labels = Corpus(tmp_path).labels("teacher")
    key = LabelKey("a", "e1", elements[1].text_sha1, "topic")
    assert set(labels) == {
        LabelKey("a", "e0", elements[0].text_sha1, "topic"),
        key,
        LabelKey("a", "e1", elements[1].text_sha1, "function"),
        LabelKey("a", "e2", elements[2].text_sha1, "topic"),
        LabelKey("a", "e2", "f" * 40, "topic"),
    }
    assert labels[key].probs == {"annual_report.risk": 0.2, "annual_report.other": 0.8}  # last
    assert labels[key].key == key and labels[key].provenance == provenance
    assert Corpus(tmp_path).labels("other") == {}

    # A torn final line (a crash mid-write) is ignored, and the next write drops it.
    path = tmp_path / "labels" / "teacher.jsonl"
    with path.open("ab") as f:
        f.write(b'{"doc_id": "a", "elem')
    assert len(Corpus(tmp_path).labels("teacher")) == 5
    corpus.write_labels("teacher", [row(elements[0], facet="function", probs={"definition": 0.1})])
    assert len(Corpus(tmp_path).labels("teacher")) == 6
    assert all(json.loads(line) for line in path.read_text().splitlines())

    with pytest.raises(ValidationError):
        row(elements[0], probs={"x": 1.5})
    with pytest.raises(ValidationError):
        row(elements[0], probs={"x": float("nan")})


def test_stats_are_computed_and_written(tmp_path, monkeypatch):
    corpus = Corpus(tmp_path)
    corpus.add_document(_record("t", 1, "train"), _elements("t", BODY), [1, 2])
    corpus.add_document(_record("g", 2, "gold", title="Kept extra field"), _elements("g", BODY[:1]),
                        [1])  # fmt: skip
    corpus.write_labels(
        "teacher",
        [LabelRow(doc_id="t", element_id="e0", text_sha1=text_sha1("Risks"), facet="topic",
                  probs={"x": 1.0}, provenance={})],
    )  # fmt: skip

    def no_full_load(self, labeler):
        raise AssertionError("stats() must stream label files")

    monkeypatch.setattr(Corpus, "labels", no_full_load)
    stats = corpus.stats()
    monkeypatch.undo()
    assert stats == json.loads((tmp_path / "stats.json").read_text())
    assert stats["documents"] == 2 and stats["elements"] == 4 and stats["pages_read"] == 3
    assert stats["by_split"] == {
        "calib": {"documents": 0, "elements": 0},
        "gold": {"documents": 1, "elements": 1},
        "train": {"documents": 1, "elements": 3},
    }
    assert stats["by_kind"] == {"heading": 2, "paragraph": 1, "table": 1}
    assert stats["by_source"] == {"edgar_ars": 2}
    assert stats["labels"] == {"teacher": 1} and stats["failed"] == 0
    assert corpus.manifest()[1].model_dump()["title"] == "Kept extra field"  # extras survive
    assert Corpus(tmp_path / "empty").stats()["documents"] == 0


def test_read_manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text(_record("a", 1).model_dump_json() + "\n\n" + _record("b", 2).model_dump_json())
    assert [r.id for r in read_manifest(path)] == ["a", "b"]
    path.write_text(_record("a", 1).model_dump_json() + '\n{"id": "b"}\n')
    with pytest.raises(DocrooterError, match="line 2"):
        read_manifest(path)
    path.write_text('{"id": "a", "sha256": "nothex"}\n')
    with pytest.raises(DocrooterError, match="line 1"):
        read_manifest(path)
