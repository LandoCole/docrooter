import pytest
from pydantic import ValidationError

from docrooter import Document, Section
from docrooter import TagResult as TopLevelTagResult
from docrooter.elements import Element, TagResult


def _element(**extra) -> Element:
    return Element(
        id="e0", kind="paragraph", text="hi", boxes=[], order=0, section_id="s1", **extra
    )


def test_v04_element_json_unchanged_without_tags():
    element = _element()
    assert element.tags is None
    assert "tags" not in element.model_dump()
    assert '"tags"' not in element.model_dump_json()
    doc = Document(
        source="x",
        sections={"s1": Section(id="s1", title="T", level=1, text="hi", element_ids=["e0"])},
        root_ids=["s1"],
        elements=[element],
    )
    assert "tags" not in doc.model_dump()["elements"][0]
    assert '"tags"' not in doc.model_dump_json()


def test_tags_round_trip():
    tags = {
        "topic": TagResult(
            labels=["revenue"],
            probs={"revenue": 0.7, "other": 0.3},
            confidence=0.7,
            source="bundle:element-tagger-v1",
            family_probs={"annual_report": 0.9, "generic": 0.1},
        ),
        "function": TagResult(
            labels=["definition", "claim"],
            probs={"definition": 0.6, "claim": 0.55},
            confidence=0.6,
            source="teacher:mlx",
        ),
    }
    element = _element(tags=tags)
    dumped = element.model_dump()
    assert dumped["tags"]["topic"]["family_probs"] == {"annual_report": 0.9, "generic": 0.1}
    assert "family_probs" not in dumped["tags"]["function"]
    restored = Element.model_validate_json(element.model_dump_json())
    assert restored == element
    assert restored.tags is not None
    assert restored.tags["function"].family_probs is None
    assert restored.tags["topic"].family_probs == {"annual_report": 0.9, "generic": 0.1}


def test_tagresult_exported_at_top_level():
    assert TopLevelTagResult is TagResult


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_probs_and_confidence_validation(bad):
    ok = {"labels": ["a"], "probs": {"a": 0.5}, "confidence": 0.5, "source": "s"}
    TagResult(**ok)
    TagResult(**{**ok, "probs": {"a": 0.0, "b": 1.0}, "confidence": 1.0})
    TagResult(**{**ok, "confidence": 0.0})
    with pytest.raises(ValidationError):
        TagResult(**{**ok, "confidence": bad})
    with pytest.raises(ValidationError):
        TagResult(**{**ok, "probs": {"a": bad}})
    with pytest.raises(ValidationError):
        TagResult(**{**ok, "family_probs": {"f": bad}})
