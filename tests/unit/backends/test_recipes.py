import sys
import types

import pytest

from docrooter.backends.loading import attn_implementation, select_device, trust_decision
from docrooter.backends.recipes import Recipe, get_recipe, register_recipe
from docrooter.errors import BackendError

ULTRA = "dmis-lab/UltraBERT-Embed-150M"
ULTRA_REV = "367d5343b5a0164d57b810cedf5de4b8983d5984"


def test_trust_requires_pin_or_explicit():
    unknown = get_recipe("some/unknown-model")
    assert trust_decision("some/unknown-model", unknown, None) is False
    assert trust_decision("some/unknown-model", unknown, True) is True

    unpinned = Recipe(trust_remote_code=True)
    with pytest.raises(BackendError, match="trust_remote_code=True"):
        trust_decision("some/remote-model", unpinned, None)
    with pytest.raises(BackendError, match="trust_remote_code=True"):
        trust_decision("some/remote-model", unpinned, False)
    assert trust_decision("some/remote-model", unpinned, True) is True

    assert trust_decision(ULTRA, get_recipe(ULTRA), None) is True


def test_builtin_recipes():
    ultra = get_recipe(ULTRA)
    assert ultra.revision == ULTRA_REV and ultra.trust_remote_code
    assert (ultra.max_seq_len, ultra.finetune_lr) == (1024, 5e-5)
    big = get_recipe("dmis-lab/UltraBERT-380M")
    assert big.revision == "65bbbfeefb03bc93b9965d1d511708dabcee8511"
    assert (big.trust_remote_code, big.max_seq_len, big.finetune_lr) == (True, 1024, 3e-5)
    bge = get_recipe("BAAI/bge-small-en-v1.5")
    assert bge.query_prompt == "Represent this sentence for searching relevant passages: "
    assert get_recipe("MoritzLaurer/deberta-v3-base-zeroshot-v2.0").temperature == 1.0
    default = get_recipe("nobody/knows")
    assert default == Recipe() and default.query_prompt == "{instruction}\n"


def test_ultrabert_prompts():
    recipe = get_recipe(ULTRA)
    assert recipe.query_prompt.format(instruction="X") == "Instruct: X\nQuery: "
    assert recipe.label_prompt == "Label: "


def test_get_recipe_overrides_and_register():
    assert get_recipe(ULTRA, batch_size=4, revision=None).batch_size == 4
    assert get_recipe(ULTRA, revision=None).revision == ULTRA_REV  # None means "no override"
    assert get_recipe(ULTRA, revision="abc").revision == "abc"
    with pytest.raises(ValueError):
        get_recipe(ULTRA, no_such_field=1)
    register_recipe("my/model", Recipe(temperature=0.5))
    assert get_recipe("my/model").temperature == 0.5
    # overrides never mutate the registered recipe
    assert get_recipe("my/model", temperature=0.9).temperature == 0.9
    assert get_recipe("my/model").temperature == 0.5


def _fake_torch(monkeypatch, *, cuda=False, mps=False):
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_device_order(monkeypatch):
    monkeypatch.delenv("DOCROOTER_DEVICE", raising=False)
    _fake_torch(monkeypatch, cuda=True, mps=True)
    assert select_device() == "cuda"
    _fake_torch(monkeypatch, cuda=False, mps=True)
    assert select_device() == "mps"
    _fake_torch(monkeypatch, cuda=False, mps=False)
    assert select_device() == "cpu"


def test_device_preference_and_env(monkeypatch):
    monkeypatch.delenv("DOCROOTER_DEVICE", raising=False)
    _fake_torch(monkeypatch, cuda=True, mps=True)
    assert select_device("cpu") == "cpu"
    monkeypatch.setenv("DOCROOTER_DEVICE", "mps")
    assert select_device() == "mps"
    assert select_device("cpu") == "cpu"  # explicit argument beats the environment


def test_device_without_torch(monkeypatch):
    monkeypatch.delenv("DOCROOTER_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)  # makes ``import torch`` raise ImportError
    assert select_device() == "cpu"


def test_attn_implementation(monkeypatch):
    assert attn_implementation("cpu") == "sdpa"
    assert attn_implementation("mps") == "sdpa"
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    assert attn_implementation("cuda") == "sdpa"
    monkeypatch.setitem(sys.modules, "flash_attn", types.ModuleType("flash_attn"))
    assert attn_implementation("cuda") == "flash_attention_2"


def test_batch_size_defaults_by_device():
    recipe = get_recipe(ULTRA)
    assert recipe.batch_size is None
    assert recipe.batch_size_for("cpu") == 16 and recipe.batch_size_for("mps") == 16
    assert recipe.batch_size_for("cuda") == 32 and recipe.batch_size_for("cuda:1") == 32
    pinned = get_recipe(ULTRA, batch_size=8)
    assert pinned.batch_size_for("cuda") == 8 and pinned.batch_size_for("cpu") == 8
