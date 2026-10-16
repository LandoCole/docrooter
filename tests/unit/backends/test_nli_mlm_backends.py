"""NLI and MLM-choice backends against fake tokenizers and models (no torch)."""

import contextlib
import sys
import types

import numpy as np
import pytest

from docrooter.backends.base import Candidate
from docrooter.backends.mlm_choice import LETTERS, MLMChoiceBackend
from docrooter.backends.nli import NLIBackend
from docrooter.backends.registry import resolve_backend
from docrooter.errors import BackendError

ULTRA_380 = "dmis-lab/UltraBERT-380M"


class Batch(dict):
    """Stands in for a BatchEncoding: item access plus ``.to``."""

    def to(self, device):
        self.device = device
        return self


class Out:
    def __init__(self, logits):
        self.logits = logits


class Arr(np.ndarray):
    """ndarray with the ``.float().cpu()`` chain real tensors offer."""

    def float(self):
        return self

    def cpu(self):
        return self


def arr(x):
    return np.asarray(x, dtype=float).view(Arr)


# --- NLI -----------------------------------------------------------------------------------


class FakeNLITokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, premises, hypotheses, **kwargs):
        self.calls.append((list(premises), list(hypotheses), kwargs))
        return Batch(premises=list(premises), hypotheses=list(hypotheses))

    def encode(self, text, add_special_tokens=True):
        return text.split()


class FakeNLIModel:
    """Entailment logit = 5 when the hypothesis mentions a word of the premise, else 0."""

    def __init__(self, label2id):
        self.config = types.SimpleNamespace(label2id=label2id)
        self.batches = []
        self.moved_to = None
        self.evaluated = False

    def to(self, device):
        self.moved_to = device

    def eval(self):
        self.evaluated = True

    def __call__(self, premises, hypotheses):
        self.batches.append(len(premises))
        rows = []
        for p, h in zip(premises, hypotheses, strict=True):
            hit = 5.0 if set(p.lower().split()) & set(h.lower().strip(".").split()) else 0.0
            row = [0.0] * len(self.config.label2id)
            for key, index in self.config.label2id.items():
                if key.lower() == "entailment":
                    row[index] = hit
            rows.append(row)
        return Out(arr(rows))


@pytest.fixture
def fake_transformers(monkeypatch):
    state = types.SimpleNamespace(
        label2id={"entailment": 0, "not_entailment": 1},
        tokenizer=FakeNLITokenizer(),
        models=[],
        model_kwargs=[],
        tokenizer_kwargs=[],
    )

    def make_model(model_id, **kwargs):
        state.model_kwargs.append((model_id, kwargs))
        model = state.model_factory(state.label2id) if hasattr(state, "model_factory") else None
        state.models.append(model)
        return model

    state.model_factory = FakeNLIModel
    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda model_id, **kw: (
                state.tokenizer_kwargs.append((model_id, kw)) or state.tokenizer
            )
        ),
        AutoModelForSequenceClassification=types.SimpleNamespace(from_pretrained=make_model),
    )
    torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")
    return state


TAX = Candidate("tax", "Tax")
LIT = Candidate("lit", "Litigation")


def test_nli_construction_does_not_load(fake_transformers):
    b = NLIBackend()
    assert fake_transformers.models == []
    assert b.name == "nli:MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert b.max_candidates is None and b.supports("x") and b.temperature == 1.0
    with pytest.raises(BackendError, match="placeholder"):
        NLIBackend(hypothesis_template="no slot")


def test_nli_softmax_over_candidates(fake_transformers):
    b = NLIBackend(batch_size=2)
    s = b.score(["tax filing", "litigation hold"], [TAX, LIT])
    assert s.kind == "softmax" and s.values.shape == (2, 2)
    assert np.allclose(s.values.sum(axis=1), 1)
    assert s.values[0, 0] > s.values[0, 1] and s.values[1, 1] > s.values[1, 0]
    (model,) = fake_transformers.models
    assert model.batches == [2, 2]  # 4 pairs in batches of 2
    assert model.moved_to == "cpu" and model.evaluated
    premises, hypotheses, kwargs = fake_transformers.tokenizer.calls[0]
    assert premises == ["tax filing", "tax filing"]
    assert hypotheses == ["This text is about Tax.", "This text is about Litigation."]
    assert kwargs["truncation"] == "only_first" and kwargs["max_length"] == 512


def test_nli_independent_when_multi_label(fake_transformers):
    s = NLIBackend().score(["tax filing"], [TAX, LIT], multi_label=True)
    assert s.kind == "independent"
    # entail 5 vs other 0 -> sigmoid(5); entail 0 vs other 0 -> 0.5
    assert np.allclose(s.values[0], [1 / (1 + np.exp(-5)), 0.5])


def test_nli_uses_candidate_text_and_template(fake_transformers):
    b = NLIBackend(hypothesis_template="Topic: {}")
    b.score(["x"], [Candidate("t", "Tax: income taxes")])
    assert fake_transformers.tokenizer.calls[0][1] == ["Topic: Tax: income taxes"]


def test_nli_load_arguments(fake_transformers):
    NLIBackend(device="mps").score(["x"], [TAX])
    ((model_id, kwargs),) = fake_transformers.model_kwargs
    assert model_id == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
    assert kwargs == {"revision": None, "dtype": "float32"}  # DeBERTa rejects attn_implementation
    assert "trust_remote_code" not in kwargs
    assert fake_transformers.models[0].moved_to == "mps"


def test_nli_three_way_labels(fake_transformers):
    fake_transformers.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
    s = NLIBackend().score(["tax"], [TAX], multi_label=True)
    assert s.values[0, 0] > 0.9  # entail 5 vs contradiction 0


def test_nli_labels_without_contradiction_pool_the_rest(fake_transformers):
    fake_transformers.label2id = {"ENTAILMENT": 1, "neutral": 0, "other": 2}
    fake_transformers.model_factory = lambda l2i: FakeNLIModel({"entailment": 1, "x": 0, "y": 2})
    s = NLIBackend().score(["tax", "none"], [TAX], multi_label=True)
    # no entailment hit: entail 0 vs logsumexp([0, 0]) = ln 2 -> 1/3
    assert np.allclose(s.values[1, 0], 1 / 3)


def test_nli_requires_entailment_label(fake_transformers):
    fake_transformers.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    with pytest.raises(BackendError, match="entailment"):
        NLIBackend().score(["x"], [TAX])
    fake_transformers.label2id = {"not_entailment": 0, "other": 1}
    with pytest.raises(BackendError, match="entailment"):
        NLIBackend().score(["x"], [TAX])


def test_nli_empty_and_tokenizer(fake_transformers):
    b = NLIBackend()
    assert b.score([], [TAX]).values.shape == (0, 1)
    assert b.score(["t"], []).values.shape == (1, 0)
    assert fake_transformers.models == []
    assert b.tokenizer()("one two three") == 3


def test_registry_resolves_nli(fake_transformers):
    b = resolve_backend("nli:some/model@abc")
    assert isinstance(b, NLIBackend) and b.recipe.revision == "abc"


# --- MLM -----------------------------------------------------------------------------------


class FakeMLMTokenizer:
    """Whitespace tokenizer; one id per distinct word, ``[MASK]`` is id 1, pad is 0."""

    mask_token = "<mask>"
    mask_token_id = 1

    def __init__(self, single_token=lambda form: True):
        self.vocab = {"<mask>": 1}
        self.single_token = single_token
        self.calls = []

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab) + 10)

    def encode(self, text, add_special_tokens=True):
        words = text.split()
        if len(text.strip()) == 1:  # a lone marker: one token (keeping any space) or two
            if self.single_token(text):
                return [self._id(text)]
            return [self._id(text), self._id(text + "'")]
        ids = [self._id(w) for w in words]
        return [2, *ids, 3] if add_special_tokens else ids

    def decode(self, ids):
        inverse = {v: k for k, v in self.vocab.items()}
        return " ".join(inverse[i] for i in ids)

    def __call__(self, prompts, **kwargs):
        self.calls.append((list(prompts), kwargs))
        rows = [self.encode(p) for p in prompts]
        width = max(map(len, rows))
        return Batch(input_ids=np.array([r + [0] * (width - len(r)) for r in rows]))


class FakeMLM:
    """Puts logit 3 on the letter whose option text contains the prompt's key word."""

    def __init__(self, tokenizer, max_len=None):
        self.tok = tokenizer
        self.config = types.SimpleNamespace()
        self.moved_to = None
        self.batches = []

    def to(self, device):
        self.moved_to = device

    def eval(self):
        pass

    def __call__(self, input_ids):
        self.batches.append(np.asarray(input_ids).shape)
        ids = np.asarray(input_ids)
        vocab = 5000
        logits = np.zeros((ids.shape[0], ids.shape[1], vocab))
        inverse = {v: k for k, v in self.tok.vocab.items()}
        for b, row in enumerate(ids):
            words = [inverse.get(int(i), "") for i in row]
            text = words[words.index("Text:") + 1] if "Text:" in words else ""
            options = {}
            for i, w in enumerate(words):
                if w in {f"{L}." for L in LETTERS} and i + 1 < len(words):
                    options[w[0]] = words[i + 1]
            mask_at = int(np.nonzero(row == 1)[0][0])
            for letter, option in options.items():
                if option.lower() == text.lower():
                    logits[
                        b, mask_at, self.tok.vocab.get(" " + letter) or self.tok.vocab[letter]
                    ] = 3.0
            # decoy: huge logit on a non-marker token that must be ignored
            logits[b, mask_at, 4999] = 100.0
        return Out(arr(logits))


@pytest.fixture
def fake_mlm(monkeypatch):
    state = types.SimpleNamespace(tokenizer=FakeMLMTokenizer(), model_kwargs=[], tok_kwargs=[])
    model_holder = {}

    def make_tok(model_id, **kw):
        state.tok_kwargs.append((model_id, kw))
        return state.tokenizer

    def make_model(model_id, **kw):
        state.model_kwargs.append((model_id, kw))
        model_holder["m"] = FakeMLM(state.tokenizer)
        state.model = model_holder["m"]
        return state.model

    transformers = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=make_tok),
        AutoModelForMaskedLM=types.SimpleNamespace(from_pretrained=make_model),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    )
    monkeypatch.setenv("DOCROOTER_DEVICE", "cpu")
    return state


def test_mlm_attributes_and_trust(fake_mlm):
    b = MLMChoiceBackend()
    assert b.max_candidates == 26 and b.name == f"mlm-choice:{ULTRA_380}"
    assert b.trust_remote_code is True and b.recipe.revision.startswith("65bbbfee")
    assert b.temperature == 1.0
    assert fake_mlm.model_kwargs == []  # lazy
    with pytest.raises(BackendError, match="trust_remote_code=True"):
        MLMChoiceBackend(revision="main")
    assert MLMChoiceBackend(revision="main", trust_remote_code=True).trust_remote_code
    assert MLMChoiceBackend("some/mlm").trust_remote_code is False


def test_mlm_load_arguments(fake_mlm):
    MLMChoiceBackend(device="mps").score(["Tax"], [Candidate("a", "Tax")])
    ((_, kwargs),) = fake_mlm.model_kwargs
    assert kwargs == {
        "revision": "65bbbfeefb03bc93b9965d1d511708dabcee8511",
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "dtype": "float32",
    }
    ((_, tok_kwargs),) = fake_mlm.tok_kwargs
    assert tok_kwargs["trust_remote_code"] is True
    assert fake_mlm.model.moved_to == "mps"


def test_mlm_scores_restricted_to_marker_logits(fake_mlm):
    cands = [Candidate("a", "Alpha"), Candidate("b", "Beta"), Candidate("c", "Gamma")]
    s = MLMChoiceBackend().score(["Beta", "Gamma"], cands)
    assert s.kind == "softmax" and s.values.shape == (2, 3)
    assert np.allclose(s.values.sum(axis=1), 1)
    assert s.values[0].argmax() == 1 and s.values[1].argmax() == 2
    e = np.exp(3.0)
    assert np.allclose(s.values[0], np.array([1, e, 1]) / (e + 2))  # the 100-logit decoy is out


def test_mlm_prompt_uses_mask_token_and_instruction(fake_mlm):
    MLMChoiceBackend().score(["Alpha"], [Candidate("a", "Alpha")], instruction="Pick one")
    prompts, kwargs = fake_mlm.tokenizer.calls[0]
    assert prompts == ["Pick one\nText: Alpha\nOptions:\nA. Alpha\nAnswer: <mask>"]
    assert kwargs["max_length"] == 1024 and kwargs["padding"] is True
    MLMChoiceBackend().score(["Alpha"], [Candidate("a", "Alpha")])
    assert fake_mlm.tokenizer.calls[1][0][0].startswith("Classify this document section\nText:")


def test_mlm_multi_label_is_still_softmax(fake_mlm):
    s = MLMChoiceBackend().score(
        ["Alpha"], [Candidate("a", "Alpha"), Candidate("b", "B")], multi_label=True
    )
    assert s.kind == "softmax" and np.isclose(s.values.sum(), 1)


def test_mlm_marker_prefers_space_form(fake_mlm):
    b = MLMChoiceBackend()
    b.score(["Alpha"], [Candidate("a", "Alpha")])
    assert (
        b._marker_ids["A"] == fake_mlm.tokenizer.vocab[" A"]
        if " A" in fake_mlm.tokenizer.vocab
        else True
    )
    # when " A" is two tokens, fall back to "A"
    fake_mlm.tokenizer.single_token = lambda form: not form.startswith(" ")
    fresh = MLMChoiceBackend()
    fresh.score(["Alpha"], [Candidate("a", "Alpha")])
    assert fresh._marker_ids["A"] == fake_mlm.tokenizer.vocab["A"]


def test_mlm_marker_without_single_token_raises(fake_mlm):
    fake_mlm.tokenizer.single_token = lambda form: False
    with pytest.raises(BackendError, match="marker 'A'"):
        MLMChoiceBackend().score(["Alpha"], [Candidate("a", "Alpha")])


def test_mlm_too_many_candidates(fake_mlm):
    cands = [Candidate(str(i), f"opt{i}") for i in range(27)]
    with pytest.raises(BackendError, match="at most 26"):
        MLMChoiceBackend().score(["t"], cands)
    assert fake_mlm.model_kwargs == []
    s = MLMChoiceBackend().score(["opt0"], cands[:26])
    assert s.values.shape == (1, 26)


def test_mlm_truncates_text_to_fit(fake_mlm):
    b = MLMChoiceBackend()
    b.recipe = b.recipe.model_copy(update={"max_seq_len": 30})
    long_text = " ".join(f"w{i}" for i in range(200))
    b.score([long_text], [Candidate("a", "Alpha"), Candidate("b", "Beta")])
    (prompt,), kwargs = fake_mlm.tokenizer.calls[0]
    assert len(fake_mlm.tokenizer.encode(prompt)) <= 30
    assert prompt.endswith("Answer: <mask>") and "w0 w1" in prompt and "w199" not in prompt


def test_mlm_options_that_do_not_fit_raise(fake_mlm):
    b = MLMChoiceBackend()
    b.recipe = b.recipe.model_copy(update={"max_seq_len": 8})
    with pytest.raises(BackendError, match="options alone"):
        b.score(["t"], [Candidate("a", "Alpha Alpha Alpha"), Candidate("b", "Beta")])


def test_mlm_batches_and_restores_order(fake_mlm):
    b = MLMChoiceBackend()
    b.recipe = b.recipe.model_copy(update={"batch_size": 2})
    cands = [Candidate("a", "Alpha"), Candidate("b", "Beta")]
    s = b.score(["Beta", "Alpha", "Beta"], cands)
    assert [int(v.argmax()) for v in s.values] == [1, 0, 1]
    assert [n for n, _ in fake_mlm.model.batches] == [2, 1]


def test_mlm_empty_and_tokenizer(fake_mlm):
    b = MLMChoiceBackend()
    assert b.score([], [Candidate("a", "x")]).values.shape == (0, 1)
    assert b.score(["t"], []).values.shape == (1, 0)
    assert fake_mlm.model_kwargs == []
    assert b.tokenizer()("one two three") == 3


def test_registry_resolves_mlm_choice(fake_mlm):
    assert isinstance(resolve_backend("mlm-choice"), MLMChoiceBackend)


def test_nli_temperature_is_one_for_any_recipe(fake_transformers):
    """Entailment probabilities are used as they are; the generic recipe's 0.05 is for
    embedding similarities and would sharpen them to p**20."""
    assert NLIBackend("facebook/bart-large-mnli").temperature == 1.0
    assert resolve_backend("nli:facebook/bart-large-mnli").temperature == 1.0


def test_nli_default_batch_is_16_off_cuda_and_close_reloads(fake_transformers):
    b = NLIBackend()
    texts = [f"tax filing {i}" for i in range(9)]
    b.score(texts, [TAX, LIT])  # 18 pairs
    assert fake_transformers.models[0].batches == [16, 2]
    b.close()
    assert b._model is None and b._tokenizer is None
    b.score(["tax filing"], [TAX, LIT])
    assert len(fake_transformers.models) == 2  # loaded again after close


def test_mlm_close_reloads(fake_mlm):
    b = MLMChoiceBackend()
    cands = [Candidate("a", "Alpha"), Candidate("b", "Beta")]
    b.score(["Alpha"], cands)
    b.close()
    assert b._model is None and b._tokenizer is None and b._marker_ids == {}
    b.score(["Beta"], cands)
    assert len(fake_mlm.model_kwargs) == 2


def test_nli_mps_oom_moves_the_loaded_model_to_cpu(fake_transformers, monkeypatch):
    oom = "MPS backend out of memory (MPS allocated: 7.9 GB, max allowed: 8.0 GB)"
    real_call = FakeNLIModel.__call__

    def call(self, premises, hypotheses):
        if self.moved_to == "mps":
            raise RuntimeError(oom)
        return real_call(self, premises, hypotheses)

    monkeypatch.setattr(FakeNLIModel, "__call__", call)
    b = NLIBackend(device="mps")
    s = b.score(["tax filing"], [TAX, LIT])
    assert s.values[0, 0] > s.values[0, 1]
    (model,) = fake_transformers.models  # moved, not reloaded
    assert model.moved_to == "cpu" and b._device_in_use == "cpu"
