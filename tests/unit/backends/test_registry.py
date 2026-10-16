import pytest

from docrooter.backends import (
    KeywordBackend,
    parse_spec,
    register_backend_kind,
    registry,
    resolve_backend,
)
from docrooter.errors import BackendError


def test_parse_spec():
    assert parse_spec("embedding:dmis-lab/UltraBERT-Embed-150M@abc") == (
        "embedding",
        "dmis-lab/UltraBERT-Embed-150M",
        "abc",
    )
    assert parse_spec("keyword") == ("keyword", "", None)
    assert parse_spec("nli:some/model") == ("nli", "some/model", None)
    assert parse_spec("trained:./models/x") == ("trained", "./models/x", None)


def test_unknown_kind_lists_known():
    with pytest.raises(BackendError, match="embedding"):
        resolve_backend("bogus:x")


def test_resolve_keyword():
    assert isinstance(resolve_backend("keyword"), KeywordBackend)
    assert resolve_backend("keyword", temperature=0.5).temperature == 0.5


def test_instances_pass_through():
    b = KeywordBackend()
    assert resolve_backend(b) is b


def test_register_custom_kind_passes_model_and_revision(monkeypatch):
    monkeypatch.setattr(registry, "_KINDS", dict(registry._KINDS))
    calls = []

    def factory(*args, **kw):
        calls.append((args, kw))
        return KeywordBackend(name="custom")

    register_backend_kind("custom", factory)
    assert resolve_backend("custom:m@r", extra=1).name == "custom"
    assert resolve_backend("custom:m").name == "custom"
    assert resolve_backend("custom").name == "custom"
    assert calls == [(("m",), {"extra": 1, "revision": "r"}), (("m",), {}), ((), {})]


def test_unavailable_builtin_module_is_backend_error(monkeypatch):
    monkeypatch.setattr(
        registry, "_KINDS", {**registry._KINDS, "ghost": "docrooter.backends.nope:Ghost"}
    )
    with pytest.raises(BackendError, match="ghost"):
        resolve_backend("ghost")


def test_bare_specs_use_each_kind_default_model(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    # construction loads nothing, so no model extra is needed
    assert resolve_backend("embedding").name == "embedding:dmis-lab/UltraBERT-Embed-150M"
    assert resolve_backend("nli").name == "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert resolve_backend("mlm-choice").name == "mlm-choice:dmis-lab/UltraBERT-380M"
    assert resolve_backend("jev").name == "jev:jev-1.13.0"


def test_embedding_backend_has_a_default_model():
    from docrooter.backends.embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingBackend

    assert DEFAULT_EMBEDDING_MODEL == "dmis-lab/UltraBERT-Embed-150M"
    assert EmbeddingBackend().model_id == DEFAULT_EMBEDDING_MODEL


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("openai", "openai needs a model: openai:<model>"),
        ("trained", "trained needs a path: trained:<bundle dir>"),
    ],
)
def test_bare_specs_that_need_a_model_are_backend_errors(spec, message):
    with pytest.raises(BackendError) as info:
        resolve_backend(spec)
    assert str(info.value) == message
