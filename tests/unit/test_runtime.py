"""``docrooter.configure`` and the runtime settings applied at model load, with a fake torch."""

import logging
import os
import sys
import types

import pytest

import docrooter
from docrooter import runtime
from docrooter.backends import loading

GB = 1024**3
ENV_VARS = (
    "DOCROOTER_DEVICE",
    "DOCROOTER_THREADS",
    "DOCROOTER_MEMORY_BUDGET_GB",
    "DOCROOTER_MPS_MEMORY_FRACTION",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
)


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    runtime._reset()
    yield
    runtime._reset()


class FakeTorch(types.SimpleNamespace):
    """Records ``set_num_threads``, the MPS fraction and the cache flushes."""

    def __init__(self, *, mps=True, cuda=False, has_fraction=True, recommended=12 * GB):
        calls: list = []
        mps_ns = types.SimpleNamespace(
            empty_cache=lambda: calls.append(("mps_empty",)),
            recommended_max_memory=lambda: recommended,
        )
        if has_fraction:
            mps_ns.set_per_process_memory_fraction = lambda f: calls.append(("fraction", f))
        super().__init__(
            calls=calls,
            set_num_threads=lambda n: calls.append(("threads", n)),
            mps=mps_ns,
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
            cuda=types.SimpleNamespace(
                is_available=lambda: cuda,
                is_initialized=lambda: cuda,
                empty_cache=lambda: calls.append(("cuda_empty",)),
            ),
        )


def test_configure_is_exported():
    assert docrooter.configure is runtime.configure
    assert "configure" in docrooter.__all__


def test_defaults_environment_and_configure(monkeypatch):
    assert runtime.settings() == runtime.RuntimeSettings(memory_budget_gb=8.0)
    monkeypatch.setenv("DOCROOTER_THREADS", "3")
    monkeypatch.setenv("DOCROOTER_MEMORY_BUDGET_GB", "6")
    monkeypatch.setenv("DOCROOTER_MPS_MEMORY_FRACTION", "0.25")
    monkeypatch.setenv("DOCROOTER_DEVICE", "mps")
    assert runtime.settings() == runtime.RuntimeSettings("mps", 3, 6.0, 0.25)
    docrooter.configure(threads=2, memory_budget_gb=4, mps_memory_fraction=0.5, device="cpu")
    assert runtime.settings() == runtime.RuntimeSettings("cpu", 2, 4.0, 0.5)
    docrooter.configure(threads=6)  # None leaves the others alone
    assert runtime.settings() == runtime.RuntimeSettings("cpu", 6, 4.0, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [{"threads": 0}, {"threads": 1.5}, {"mps_memory_fraction": 0}, {"mps_memory_fraction": 1.5},
     {"memory_budget_gb": 0}, {"memory_budget_gb": -2}, {"device": " "}],
)  # fmt: skip
def test_configure_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        docrooter.configure(**kwargs)


@pytest.mark.parametrize(
    ("name", "value"),
    [("DOCROOTER_THREADS", "four"), ("DOCROOTER_MEMORY_BUDGET_GB", "lots"),
     ("DOCROOTER_MEMORY_BUDGET_GB", "0"), ("DOCROOTER_MPS_MEMORY_FRACTION", "2")],
)  # fmt: skip
def test_bad_environment_values_raise(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        runtime.settings()


def test_mps_fraction_from_the_budget():
    settings = runtime.RuntimeSettings(memory_budget_gb=8.0)
    assert settings.metal_budget_gb == 6.5  # 1.5 GiB of host headroom
    assert settings.mps_fraction(16 * GB) == 6.5 / 16
    assert settings.mps_fraction(4 * GB) == 1.0  # never above 1
    assert runtime.RuntimeSettings(memory_budget_gb=2.0).metal_budget_gb == 1.0  # half, at least
    assert runtime.RuntimeSettings(mps_memory_fraction=0.3).mps_fraction(16 * GB) == 0.3


def test_default_batch_size_follows_device_and_budget():
    assert runtime.default_batch_size("cuda:0", 1.0) == 32
    assert runtime.default_batch_size("cpu", 8.0) == 16
    assert runtime.default_batch_size("mps", 32.0) == 16
    assert runtime.default_batch_size("cpu", 4.0) == 8
    assert runtime.default_batch_size("cpu", 0.5) == 4  # floor
    docrooter.configure(memory_budget_gb=4)
    assert runtime.default_batch_size("mps") == 8


def test_configure_device_is_used_by_select_device(monkeypatch):
    monkeypatch.setenv("DOCROOTER_DEVICE", "from-env")
    assert loading.select_device() == "from-env"
    docrooter.configure(device="from-configure")
    assert loading.select_device() == "from-configure"
    assert loading.select_device("explicit") == "explicit"


def test_prepare_sets_watermarks_from_the_budget_without_torch(monkeypatch):
    monkeypatch.setattr(runtime, "estimated_recommended_bytes", lambda: 16 * GB)
    assert loading.prepare_model_load("cpu") == "cpu"
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.4062"  # 6.5 of 16 GiB
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.325"
    assert "torch" not in sys.modules
    loading.after_model_load("cpu")  # no torch loaded: nothing to apply
    assert "torch" not in sys.modules


def test_exact_fraction_applied_once_torch_is_loaded(monkeypatch):
    monkeypatch.setattr(runtime, "estimated_recommended_bytes", lambda: 16 * GB)
    loading.prepare_model_load("mps")  # sets the estimated watermarks
    torch = FakeTorch(recommended=10 * GB)
    monkeypatch.setitem(sys.modules, "torch", torch)
    docrooter.configure(threads=4)
    loading.after_model_load("mps")
    assert torch.calls == [("threads", 4), ("fraction", 0.65)]  # 6.5 of 10 GiB
    loading.after_model_load("mps")  # unchanged settings are not applied again
    assert len(torch.calls) == 2
    docrooter.configure(memory_budget_gb=5)  # our own watermarks still count as ours
    loading.after_model_load("mps")
    assert torch.calls[-1] == ("fraction", 0.35) and len(torch.calls) == 3  # 3.5 of 10


def test_user_watermarks_are_kept_and_no_fraction_applied(monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
    loading.prepare_model_load("mps")
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.7"
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ
    torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    loading.after_model_load("mps")
    assert torch.calls == []


def test_watermarks_inherited_from_a_docrooter_parent_count_as_ours(monkeypatch):
    for name, value in runtime.watermark_env(runtime.settings()).items():
        monkeypatch.setenv(name, value)  # what a parent process with the same budget set
    torch = FakeTorch(recommended=16 * GB)
    monkeypatch.setitem(sys.modules, "torch", torch)
    loading.after_model_load("mps")
    assert torch.calls == [("fraction", 6.5 / 16)]


@pytest.mark.parametrize(
    ("device", "mps", "has_fraction"), [("cpu", True, True), ("mps", False, True),
                                        ("mps", True, False)],
)  # fmt: skip
def test_fraction_skipped_off_mps_or_without_the_api(monkeypatch, device, mps, has_fraction):
    torch = FakeTorch(mps=mps, has_fraction=has_fraction)
    monkeypatch.setitem(sys.modules, "torch", torch)
    loading.after_model_load(device)
    assert torch.calls == []


def test_release_memory_flushes_only_an_imported_torch(monkeypatch):
    loading.release_memory()  # no torch: just gc
    assert "torch" not in sys.modules
    torch = FakeTorch(mps=True, cuda=True)
    monkeypatch.setitem(sys.modules, "torch", torch)
    loading.release_memory()
    assert torch.calls == [("mps_empty",), ("cuda_empty",)]


OOM = "MPS backend out of memory (MPS allocated: 7.9 GB, max allowed: 8.0 GB)"


def test_is_mps_oom():
    assert loading.is_mps_oom(RuntimeError(OOM), "mps")
    assert not loading.is_mps_oom(RuntimeError(OOM), "cpu")
    assert not loading.is_mps_oom(RuntimeError("shape mismatch"), "mps")


class FlakyBackend:
    """Fails with an MPS OOM while on MPS; records the devices it ran on."""

    name = "flaky"

    def __init__(self, error=OOM):
        self._device, self._device_in_use = None, "mps"
        self._model = types.SimpleNamespace(device="mps")
        self._model.to = lambda device: setattr(self._model, "device", device)
        self.error, self.runs = error, []

    def call(self):
        self.runs.append(self._device_in_use)
        if self._device_in_use == "mps":
            raise RuntimeError(self.error)
        return "ok"


def test_run_with_cpu_fallback_retries_once_on_cpu(caplog):
    backend = FlakyBackend()
    with caplog.at_level(logging.WARNING):
        assert loading.run_with_cpu_fallback(backend, "scoring", backend.call) == "ok"
    model = backend._model
    assert backend.runs == ["mps", "cpu"]
    assert backend._device == "cpu" and backend._model is model  # moved, not reloaded
    assert model.device == "cpu"
    assert "falling back to the CPU" in caplog.text and "flaky" in caplog.text


def test_run_with_cpu_fallback_reraises_other_errors():
    backend = FlakyBackend(error="something else")
    with pytest.raises(RuntimeError, match="something else"):
        loading.run_with_cpu_fallback(backend, "scoring", backend.call)
    assert backend.runs == ["mps"] and backend._device is None


def test_move_backend_to_cpu_without_a_model_defers_to_the_next_load():
    backend = types.SimpleNamespace(name="b", _model=None, _device="mps", _device_in_use="mps")
    loading.move_backend_to_cpu(backend, "loading")
    assert backend._model is None and backend._device == backend._device_in_use == "cpu"
