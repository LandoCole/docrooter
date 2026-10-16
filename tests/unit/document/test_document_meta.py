from docrooter.document import Document


def test_meta_defaults_to_independent_empty_dict():
    a, b = Document(source="a"), Document(source="b")
    assert a.meta == {}
    a.meta["k"] = 1
    assert b.meta == {}


def test_meta_round_trips_through_model_dump():
    doc = Document.from_sections([{"title": "A", "level": 1, "text": "x"}])
    doc.meta.update({"engine": "pymupdf", "pages": 3, "nested": {"ok": [1, 2]}})
    restored = Document.model_validate(doc.model_dump())
    assert restored.meta == doc.meta
    assert Document.model_validate_json(doc.model_dump_json()).meta == doc.meta


def test_v01_payload_without_meta_still_validates():
    payload = Document.from_sections([{"title": "A", "level": 1}]).model_dump()
    del payload["meta"]
    assert Document.model_validate(payload).meta == {}
