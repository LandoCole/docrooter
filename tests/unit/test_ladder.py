import pytest

import docrooter
from docrooter import Classifier, Document, DocumentView, Flat, SectionView, Taxonomy, TaxonomyNode
from docrooter.backends import KeywordBackend
from docrooter.classify.documents import SECTION_VIEW
from docrooter.classify.normalize import NORMALIZE_VIEW, normalize_sections
from docrooter.ladder import (
    FULL_CAP,
    FULL_RESERVE_SHARE,
    FULL_RESERVE_TOKENS,
    LEVELS,
    document_view,
    full_view_budget,
    next_level,
    section_view,
)
from docrooter.views import estimate_tokens


def report() -> Document:
    body = (
        "Revenue grew strongly in the fiscal year. Operating margin improved on lower costs. "
        "The board proposes a higher dividend. Shareholders approved the plan. "
    )
    return Document.from_sections(
        [
            {"title": "Letter to shareholders", "level": 1, "text": body * 3},
            {"title": "Financial review", "level": 1, "text": body * 6},
            {"title": "Income statement", "level": 2, "text": "Revenue, costs and profit."},
            {"title": "Balance sheet", "level": 2, "text": ""},
            {"title": "Outlook", "level": 1, "text": body},
        ],
        source="reports/acme-2024.pdf",
    )


def taxonomy() -> Taxonomy:
    return Taxonomy(
        [
            TaxonomyNode(id="annual_report", label="Annual report", description="revenue"),
            TaxonomyNode(id="contract", label="Contract", description="agreement parties"),
        ]
    )


class Recording(KeywordBackend):
    """A keyword backend that keeps every text it was asked to score."""

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    def score(self, texts, candidates, *, instruction=None, multi_label=False):
        self.texts.extend(texts)
        return super().score(texts, candidates, instruction=instruction, multi_label=multi_label)


def test_levels_are_monotone_in_tokens():
    doc = report()
    ids = [s.id for s in doc.walk()]
    per_level = [section_view(level).render_all(doc, ids) for level in LEVELS]
    for lower, higher in zip(per_level, per_level[1:], strict=False):
        for sid in ids:
            assert higher[sid].n_tokens >= lower[sid].n_tokens
    totals = [sum(r.n_tokens for r in rendered.values()) for rendered in per_level]
    assert totals == sorted(set(totals))  # each rung adds context on this document

    doc_tokens = [document_view(level).render(doc).n_tokens for level in LEVELS]
    assert doc_tokens == sorted(set(doc_tokens))


def test_level_and_view_together_raise():
    doc = report()
    with pytest.raises(ValueError, match="level"):
        docrooter.classify_document(
            doc, taxonomy(), backend=KeywordBackend(), level="terms", view=DocumentView()
        )
    clf = Classifier(taxonomy(), KeywordBackend(), strategy=Flat())
    with pytest.raises(ValueError, match="level"):
        clf.classify_documents([doc], level="terms", view=DocumentView())
    with pytest.raises(ValueError, match="level"):
        normalize_sections(
            doc, taxonomy="generic", backend=KeywordBackend(), level="terms", view=SectionView()
        )
    with pytest.raises(ValueError, match="level"):
        section_view("auto")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        document_view("everything")  # type: ignore[arg-type]


def test_level_none_keeps_old_defaults():
    doc = report()
    old = docrooter.classify_document(doc, taxonomy(), backend=KeywordBackend())
    assert old.view_text == DocumentView().render(doc).text
    same = docrooter.classify_document(doc, taxonomy(), backend=KeywordBackend(), level=None)
    assert same.view_text == old.view_text and same.best == old.best

    backend = Recording()
    docrooter.classify_document(doc, taxonomy(), backend=backend, mode="sections")
    leaves = [s.id for s in doc.leaves()]
    assert backend.texts == [v.text for v in SECTION_VIEW.render_all(doc, leaves).values()]

    backend = Recording()
    normalize_sections(doc, taxonomy="generic", backend=backend)
    wanted = [s.id for s in doc.walk() if s.level in (1, 2)]
    assert backend.texts == [v.text for v in NORMALIZE_VIEW.render_all(doc, wanted).values()]

    # A level replaces those defaults with its presets.
    for level in LEVELS:
        result = docrooter.classify_document(doc, taxonomy(), backend=KeywordBackend(), level=level)
        assert result.view_text == document_view(level).render(doc).text

        backend = Recording()
        docrooter.classify_document(doc, taxonomy(), backend=backend, mode="sections", level=level)
        views = section_view(level).render_all(doc, leaves)
        assert backend.texts == [v.text for v in views.values()]

        backend = Recording()
        normalize_sections(doc, taxonomy="generic", backend=backend, level=level)
        views = section_view(level).render_all(doc, wanted)
        assert backend.texts == [v.text for v in views.values()]


def test_full_level_caps_and_flags_truncation():
    words = [f"w{i}" for i in range(2000)]
    big = Document.from_sections(
        [
            {"title": f"Part {n}", "level": 1, "text": f"start{n} " + " ".join(words)}
            for n in range(100)  # about 200,000 words: far beyond the cap
        ],
        source="big.pdf",
    )
    seen: list[int] = []

    def count(text: str) -> int:
        seen.append(len(text.split()))
        return estimate_tokens(text)

    budget = full_view_budget(FULL_CAP)
    full = document_view("full").render(big, count)
    assert full.truncated and full.n_tokens <= budget < FULL_CAP
    assert budget - full.n_tokens < 10  # cut at the budget, not well short of it
    assert "start0" in full.text and "start99" not in full.text  # head first
    assert max(seen) <= 4 * budget + 1000  # the whole document is never counted

    small = document_view("full", full_cap=4000).render(big)
    assert small.truncated and small.n_tokens <= full_view_budget(4000)

    under = document_view("full").render(report())
    assert not under.truncated and "Outlook" in under.text and "dividend" in under.text

    long_section = section_view("full").render_all(big, [big.leaves()[0].id])
    rendered = next(iter(long_section.values()))
    assert rendered.truncated and rendered.n_tokens <= 2048

    # A budget smaller than one field: the heading path alone overflows the headings rung.
    deep = Document.from_sections(
        [{"title": " ".join(["Very long heading"] * 20), "level": 1, "text": "x"}], source="d.md"
    )
    tiny = next(iter(section_view("headings").render_all(deep, [deep.leaves()[0].id]).values()))
    assert tiny.truncated and 0 < tiny.n_tokens <= 32

    result = docrooter.classify_document(big, taxonomy(), backend=KeywordBackend(), level="full")
    assert result.truncated and estimate_tokens(result.view_text) <= budget
    assert not docrooter.classify_document(report(), taxonomy(), backend=KeywordBackend()).truncated


def test_next_level_order():
    assert LEVELS == ("headings", "terms", "lead", "full")
    assert [next_level(level) for level in LEVELS] == ["terms", "lead", "full", None]
    with pytest.raises(ValueError, match="level"):
        next_level("auto")  # type: ignore[arg-type]


def test_full_reserve_is_pinned():
    assert (FULL_CAP, FULL_RESERVE_TOKENS, FULL_RESERVE_SHARE) == (16000, 1024, 0.15)
    assert full_view_budget(16000) == 16000 - 1024 - 2400 == document_view("full").max_tokens
    assert document_view("full", full_cap=8000).max_tokens == 8000 - 1024 - 1200
    assert document_view("lead", full_cap=8000) == document_view("lead")  # only full uses it
    with pytest.raises(ValueError, match="full_cap"):
        document_view("full", full_cap=1200)  # nothing left after the reserve


def test_full_starts_with_the_lead():
    cover = " ".join(f"c{i}" for i in range(30000))
    doc = Document.from_sections(
        [
            {"title": "Cover", "level": 1, "text": cover},
            {"title": "Introduction", "level": 1, "text": "Intro words. " * 100},
            {"title": "Body", "level": 1, "text": "body words " * 50},
        ],
        source="x.pdf",
    )
    lines = {
        ln.split(": ", 1)[0]: ln.split(": ", 1)[1]
        for ln in document_view("full").render(doc).text.splitlines()
    }
    lead = {
        ln.split(": ", 1)[0]: ln.split(": ", 1)[1]
        for ln in document_view("lead").render(doc).text.splitlines()
    }
    assert lines["Text"].startswith(lead["Lead"])  # the lead first, though a cover precedes it
    # then the rest in reading order, each word once
    rest = lines["Text"][len(lead["Lead"]) :].split()
    assert rest[:2] == ["c0", "c1"] and "Intro" not in rest[:30000]

    small = report()
    full_text = document_view("full").render(small).text.split("Text: ", 1)[1].split()
    body = [w for s in small.walk() for w in s.text.split()]
    assert sorted(full_text) == sorted(body)  # every body word exactly once

    ids = [s.id for s in small.walk() if s.text.strip()]
    leads = section_view("lead").render_all(small, ids)
    fulls = section_view("full").render_all(small, ids)
    for sid in ids:
        section_lead = leads[sid].text.split("Lead: ", 1)[1]
        assert section_lead in " ".join(fulls[sid].text.split())


def test_terms_rung_adds_key_phrases_over_headings():
    long_toc = Document.from_sections(
        [
            {
                "title": f"Chapter {n} on the regulatory capital adequacy framework review",
                "level": 1,
                "text": f"Topic{n} liquidity coverage ratio stress testing scenario {n}.",
            }
            for n in range(40)
        ],
        source="filing.pdf",
    )
    headings = document_view("headings").render(long_toc)
    terms = document_view("terms").render(long_toc)
    assert headings.truncated  # the contents alone fill the headings rung
    assert "Key phrases:" in terms.text and "Key phrases:" not in headings.text
    assert terms.n_tokens >= headings.n_tokens + 40
    contents = lambda text: text.split("Contents: ", 1)[1]  # noqa: E731
    assert contents(terms.text).startswith(contents(headings.text))  # nothing lost


def test_sections_mode_renders_no_document_view():
    words = " ".join(f"w{i}" for i in range(3000))
    big = Document.from_sections(
        [{"title": "Part", "level": 1, "text": words}, {"title": "Two", "level": 1, "text": "x"}],
        source="big.pdf",
    )
    rendered: list[int] = []
    original = DocumentView.render

    def spy(self, doc, count_tokens=None):
        rendered.append(self.max_tokens)
        return original(self, doc, count_tokens)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(DocumentView, "render", spy)
        res = docrooter.classify_document(
            big, taxonomy(), backend=KeywordBackend(), mode="sections", level="full"
        )
    assert rendered == [] and res.view_text == ""
    assert res.truncated  # the classified section view was cut at 2048
    lead = docrooter.classify_document(
        big, taxonomy(), backend=KeywordBackend(), mode="sections", level="lead"
    )
    assert lead.truncated  # section views are cut at 192
    view = docrooter.classify_document(big, taxonomy(), backend=KeywordBackend(), level="lead")
    assert view.view_text and not view.truncated


def test_normalize_level_applies_to_sections_only():
    doc = report()
    wanted = [s.id for s in doc.walk() if s.level in (1, 2)]
    sections = [v.text for v in section_view("full").render_all(doc, wanted).values()]

    backend = Recording()
    normalize_sections(doc, backend=backend, level="full")
    assert backend.texts[0] == DocumentView().render(doc).text  # the doc-type default
    assert backend.texts[1:] == sections

    backend = Recording()
    normalize_sections(doc, backend=backend, level="full", doc_level="headings")
    assert backend.texts[0] == document_view("headings").render(doc).text
    assert backend.texts[1:] == sections


def test_unknown_level_raises_before_a_model_loads():
    doc = report()
    missing = "embedding:no-such/model-anywhere"
    with pytest.raises(ValueError, match="level"):
        docrooter.classify_document(doc, taxonomy(), backend=missing, level="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        normalize_sections(doc, backend=missing, level="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="level"):
        normalize_sections(doc, backend=missing, doc_level="bogus")  # type: ignore[arg-type]
