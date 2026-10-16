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
