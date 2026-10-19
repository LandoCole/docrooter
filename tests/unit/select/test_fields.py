"""Fields: the named needs behind ``select()`` (spec §4), their built-in sets and validation."""

import json

import pytest

from docrooter.errors import FieldError
from docrooter.select import Field, builtin_fields, load_fields, validate_fields
from docrooter.taxonomies import builtin_taxonomy

FAMILIES = ["annual_report", "fund_document", "contract"]


@pytest.fixture(scope="module")
def topics():
    return builtin_taxonomy("element_topics")


@pytest.fixture(scope="module")
def functions():
    return builtin_taxonomy("element_functions")


@pytest.mark.parametrize("family", FAMILIES)
def test_builtin_field_sets_valid_against_taxonomies(family, topics, functions):
    fields = builtin_fields(family)
    validate_fields(fields, topics, functions)  # raises on any problem
    assert 8 <= len(fields) <= 15
    assert len({f.id for f in fields}) == len(fields)
    leaves = {n.id for n in topics.leaves()}
    for field in fields:
        assert field.family == family
        assert field.id.startswith(f"{family}."), field.id
        for topic in field.topics:
            assert topic in leaves and topic.startswith(f"{family}."), (field.id, topic)
            assert topic != f"{family}.other", field.id  # "other" is not a need
        for function in field.functions:
            assert function in functions, (field.id, function)


@pytest.mark.parametrize("family", FAMILIES)
def test_each_field_has_description_and_query(family):
    for field in builtin_fields(family):
        assert field.label.strip(), field.id
        assert field.description.strip().endswith("."), field.id
        assert field.query is not None and field.query.strip().endswith("?"), field.id


def test_builtin_sets_carry_the_cross_cutting_function_pairs():
    """The pairs the plan names: fees with financial_data, risk with risk_disclosure, and
    obligations with obligation."""
    by_id = {f.id: f for family in FAMILIES for f in builtin_fields(family)}
    assert "financial_data" in by_id["fund_document.fees_and_expenses"].functions
    assert "risk_disclosure" in by_id["annual_report.risk_factors"].functions
    assert "risk_disclosure" in by_id["fund_document.principal_risks"].functions
    assert "obligation" in by_id["contract.obligations"].functions
    assert len(by_id["contract.obligations"].topics) > 1  # cross-cutting: spans clause topics


def test_builtin_fields_unknown_family():
    with pytest.raises(FieldError, match="annual_report"):  # the message lists what exists
        builtin_fields("research_paper")
    with pytest.raises(FieldError):
        builtin_fields("../taxonomies/doc_types")


def test_builtin_fields_returns_fresh_objects():
    first = builtin_fields("contract")
    first[0].topics.append("contract.other")
    assert "contract.other" not in builtin_fields("contract")[0].topics


def _custom() -> list[Field]:
    return [
        Field(
            id="fund_document.cost_of_ownership",
            label="Cost of ownership",
            description="What it costs to hold the fund for a year.",
            topics=["fund_document.fees_and_expenses"],
            functions=["financial_data"],
            query="How much does it cost to own this fund each year?",
        ),
        Field(
            id="fund_document.objective",
            label="Objective",
            description="The fund's goal.",
            topics=["fund_document.investment_objective"],
        ),
    ]


def test_custom_fields_json_roundtrip(tmp_path, topics, functions):
    fields = _custom()
    as_object = tmp_path / "fields.json"
    as_object.write_text(json.dumps({"fields": [f.model_dump() for f in fields]}))
    as_list = tmp_path / "list.json"
    as_list.write_text(json.dumps([f.model_dump(exclude_defaults=True) for f in fields]))
    assert load_fields(as_object) == fields
    assert load_fields(str(as_list)) == fields
    validate_fields(load_fields(as_object), topics, functions)
    assert load_fields(as_object)[1].functions == [] and load_fields(as_object)[1].query is None


def test_load_fields_rejects_malformed_files(tmp_path):
    typo = tmp_path / "typo.json"
    typo.write_text(json.dumps([{"id": "x", "label": "X", "description": "d.", "topic": ["a"]}]))
    with pytest.raises(FieldError, match="typo.json"):
        load_fields(typo)
    no_topics = tmp_path / "empty.json"
    no_topics.write_text(json.dumps([{"id": "x", "label": "X", "description": "d.", "topics": []}]))
    with pytest.raises(FieldError):
        load_fields(no_topics)
    not_json = tmp_path / "bad.json"
    not_json.write_text("{")
    with pytest.raises(FieldError):
        load_fields(not_json)


def _field(**overrides) -> Field:
    values = {
        "id": "annual_report.test",
        "label": "Test",
        "description": "A test field.",
        "topics": ["annual_report.risk_factors"],
    }
    return Field(**{**values, **overrides})


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ([_field(topics=["annual_report.nonexistent"])], "unknown topic"),
        ([_field(topics=["annual_report"])], "not a leaf"),
        (
            [_field(topics=["annual_report.risk_factors", "fund_document.principal_risks"])],
            "family",
        ),
        ([_field(), _field(id="x", topics=["contract.term"])], "family"),
        ([_field(id="annual_report.x", topics=["contract.term"])], "family"),
        ([_field(functions=["not_a_function"])], "unknown function"),
        ([_field(), _field()], "duplicate"),
    ],
)
def test_unknown_topic_rejected(fields, problem, topics, functions):
    with pytest.raises(FieldError, match=problem) as excinfo:
        validate_fields(fields, topics, functions)
    assert excinfo.value.problems


def test_validate_fields_reports_every_problem(topics, functions):
    bad = [_field(topics=["annual_report.nope"], functions=["nope"])]
    with pytest.raises(FieldError) as excinfo:
        validate_fields(bad, topics, functions)
    assert len(excinfo.value.problems) == 2
