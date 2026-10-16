import pytest

from docrooter import Document, DocumentView
from docrooter.views import estimate_tokens


@pytest.fixture
def doc():
    d = Document.from_sections(
        [
            {"title": "Intro", "level": 1, "text": "Intro body about the yearly results."},
            {"title": "Scope", "level": 2, "text": "Scope body."},
            {"title": "Detail", "level": 3, "text": "Deep detail."},
            {"title": "Results", "level": 1, "text": "Revenue grew strongly."},
        ],
        source="reports/acme-2024.pdf",
    )
    return d


def lines_by_label(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        label, _, rest = line.partition(": ")
        out[label] = rest
    return out


def test_render_order_and_labels(doc):
    v = DocumentView().render(doc)
    lines = v.text.splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        "Title",
        "Stats",
        "Contents",
        "Lead",
        "Key phrases",
    ]
    assert lines[0] == "Title: acme-2024"
    assert lines[1] == "Stats: 4 sections"
    assert lines[2] == "Contents: 1 Intro; 1.1 Scope; 2 Results"
    assert lines[3].startswith("Lead: Intro body about the yearly results. Scope body.")
    assert not v.truncated and v.n_tokens == estimate_tokens(v.text)


def test_title_and_pages_from_document(doc):
    titled = doc.model_copy(update={"title": "Acme Annual Report", "meta": {"n_pages": 12}})
    fields = lines_by_label(DocumentView().render(titled).text)
    assert fields["Title"] == "Acme Annual Report"
    assert fields["Stats"] == "12 pages, 4 sections"


def test_contents_depth_entries_and_existing_numbering():
    d = Document.from_sections(
        [
            {"title": "1. Introduction", "level": 1, "text": "a"},
            {"title": "1.1 Scope", "level": 2, "text": "b"},
            {"title": "2. Method", "level": 1, "text": "c"},
            {"title": "3. Results", "level": 1, "text": "d"},
        ]
    )
    contents = lines_by_label(DocumentView(toc_depth=1).render(d).text)["Contents"]
    assert contents == "1 Introduction; 2 Method; 3 Results"
    capped = lines_by_label(DocumentView(toc_entries=2).render(d).text)["Contents"]
    assert capped == "1 Introduction; 1.1 Scope"


def test_lead_skips_cover_notice_and_toc_sections():
    d = Document.from_sections(
        [
            {"title": "Cover", "level": 1, "text": "cover words"},
            {"title": "Important Notice", "level": 1, "text": "notice words"},
            {"title": "Table of Contents", "level": 1, "text": "toc words 1 2 3"},
            {"title": "Copyright", "level": 1, "text": "copyright words"},
            {"title": "Disclaimer", "level": 1, "text": "disclaimer words"},
            {"title": "Forward-Looking Statements", "level": 1, "text": "forward words"},
            {"title": "1. Overview", "level": 1, "text": "The real opening text."},
        ]
    )
    lead = lines_by_label(DocumentView(fields=("lead",)).render(d).text)["Lead"]
    assert lead == "The real opening text."


def test_lead_word_limit():
    body = " ".join(f"w{i}" for i in range(50))
    d = Document.from_sections([{"title": "A", "level": 1, "text": body}])
    lead = lines_by_label(DocumentView(fields=("lead",), lead_words=10).render(d).text)["Lead"]
    assert lead.split() == [f"w{i}" for i in range(10)]


def test_keyphrases_pool_sections():
    d = Document.from_sections(
        [
            {"title": "A", "level": 1, "text": "turbine blade"},
            {"title": "B", "level": 1, "text": "turbine blade"},
            {"title": "C", "level": 1, "text": "grid storage"},
            {"title": "D", "level": 1, "text": "hydro dam"},
        ]
    )
    phrases = lines_by_label(DocumentView(fields=("keyphrases",), keyphrases=3).render(d).text)
    # "turbine blade" scores lower than "grid storage" in each section, but wins summed over two.
    assert phrases["Key phrases"].split(", ") == ["turbine blade", "grid storage", "hydro dam"]


def test_budget_respected_and_last_field_truncated():
    body = " ".join(f"word{i}" for i in range(400))
    d = Document.from_sections([{"title": "Long", "level": 1, "text": body}], source="x.md")
    v = DocumentView(max_tokens=40).render(d)
    assert v.truncated and v.n_tokens <= 40
    lines = v.text.splitlines()
    assert lines[-1].startswith("Lead: word0") and len(lines) == 4


def test_custom_counter(doc):
    v = DocumentView(max_tokens=12).render(doc, count_tokens=lambda t: len(t.split()))
    assert v.truncated and v.n_tokens == len(v.text.split()) <= 12


def test_view_never_empty():
    v = DocumentView(fields=("lead",), max_tokens=1).render(Document(source="empty.pdf"))
    assert v.text == "empty" and v.truncated
    v2 = DocumentView(max_tokens=1).render(Document(source="annual report 2024.pdf"))
    assert v2.text.strip() and v2.truncated


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown"):
        DocumentView(fields=("title", "bogus"))


def test_contents_keeps_years_in_headings():
    d = Document.from_sections(
        [
            {"title": "2024 Annual Report", "level": 1, "text": "a"},
            {"title": "2023. Review", "level": 1, "text": "b"},
            {"title": "12 Notes", "level": 1, "text": "c"},
        ]
    )
    contents = lines_by_label(DocumentView(fields=("toc",)).render(d).text)["Contents"]
    assert contents == "1 2024 Annual Report; 2 Review; 3 Notes"
