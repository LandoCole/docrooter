"""``benchmarks._guard``: env caps and the one-model-process-at-a-time check."""

import argparse

import pytest

from benchmarks import _guard

ROOT = "/Users/me/Doc Rooter/.venv/bin/python"


GB = 1024**3
CAP_VARS = (
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "DOCROOTER_THREADS",
    "DOCROOTER_MEMORY_BUDGET_GB",
    "DOCROOTER_MPS_MEMORY_FRACTION",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
)


@pytest.fixture
def env(monkeypatch):
    """A clean environment and runtime, with a 16 GiB recommended Metal maximum."""
    from docrooter import runtime

    for name in CAP_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runtime, "estimated_recommended_bytes", lambda: 16 * GB)
    runtime._reset()
    yield monkeypatch
    runtime._reset()


def test_apply_env_caps_keeps_existing_values(env):
    import os

    env.setenv("OMP_NUM_THREADS", "2")
    applied = _guard.apply_env_caps()
    assert os.environ["OMP_NUM_THREADS"] == "2" and "OMP_NUM_THREADS" not in applied
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["DOCROOTER_THREADS"] == "4"
    assert os.environ["DOCROOTER_MEMORY_BUDGET_GB"] == "8"
    assert applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.4062"  # 8 - 1.5 GiB of 16
    assert applied["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.325"
    assert _guard.apply_env_caps() == {}


def test_mps_caps_follow_the_budget_fraction_and_configure(env):
    import os

    from docrooter import configure

    env.setenv("DOCROOTER_MEMORY_BUDGET_GB", "4")
    _guard.apply_env_caps()
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.1562"  # 2.5 of 16 GiB
    for name in _guard.MPS_CAP_VARS:
        env.delenv(name)
    env.setenv("DOCROOTER_MPS_MEMORY_FRACTION", "0.3")  # an explicit fraction wins
    _guard.apply_env_caps()
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.3"
    for name in _guard.MPS_CAP_VARS:
        env.delenv(name)
    env.delenv("DOCROOTER_MPS_MEMORY_FRACTION")
    configure(memory_budget_gb=2)  # an earlier configure() counts too
    _guard.apply_env_caps()
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0625"  # 1 (half) of 16


def test_user_watermarks_are_kept(env):
    import os

    env.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")
    _guard.apply_env_caps()
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.9"
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ


def test_configure_after_the_guard_still_applies_its_fraction(env):
    import sys
    import types

    from docrooter import configure
    from docrooter.backends import loading

    _guard.apply_env_caps()  # watermarks for 8 GiB
    configure(memory_budget_gb=4)  # then the program lowers the budget
    calls = []
    torch = types.SimpleNamespace(
        set_num_threads=lambda n: calls.append(("threads", n)),
        mps=types.SimpleNamespace(
            set_per_process_memory_fraction=lambda f: calls.append(("fraction", f)),
            recommended_max_memory=lambda: 16 * GB,
        ),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)),
    )
    env.setitem(sys.modules, "torch", torch)
    loading.after_model_load("mps")
    assert ("fraction", 2.5 / 16) in calls  # not treated as user-set watermarks


TABLE = [
    (1, 0, "launchd", "/sbin/launchd"),
    (100, 1, "zsh", "zsh -c PYTHONPATH=src uv run python -m benchmarks.run --suite few-shot"),
    (101, 100, "uv", "uv run python -m benchmarks.run --suite few-shot"),
    (102, 101, "python3.12", f"{ROOT} -m benchmarks.run --suite few-shot"),  # this process
    (200, 1, "python3.12", f"{ROOT} -m benchmarks.pdf_bench --systems x"),  # another one
    (300, 1, "zsh", "zsh wait.sh benchmarks.normalize_bench"),  # a waiting shell
    (301, 300, "pgrep", "pgrep -f benchmarks\\."),
    (400, 1, "python3.12", f"{ROOT} -m benchmarks.toc"),  # no models
    (500, 1, "Python", "/usr/bin/Python /x/bin/docrooter serve --port 8000"),
    (600, 1, "python3.12", f"{ROOT} -m pytest tests/unit/test_bench_pdf.py"),
    (700, 1, "python3.12", f"{ROOT} benchmarks/memory_check.py --case train"),
    (800, 1, "python3.12", f"{ROOT} -m benchmarks.profile_pipeline"),
]


def test_other_model_processes_skip_self_ancestors_wrappers_and_non_model_runs():
    found = _guard.other_model_processes(TABLE, pid=102, ppid=101)
    assert [pid for pid, _ in found] == [200, 500, 700, 800]


def test_teacher_labelling_is_a_model_process():
    table = [
        (10, 1, "Python", "/usr/bin/Python /x/.venv/bin/docrooter teach label store --target 9"),
        (11, 1, "docrooter", "docrooter teach label store"),
        (12, 1, "zsh", "zsh -c docrooter teach label store"),  # a wrapping shell
        (13, 1, "python3.12", f"{ROOT} -m pytest tests/unit/teach/test_label.py"),
    ]
    found = _guard.other_model_processes(table, pid=99, ppid=98)
    assert [pid for pid, _ in found] == [10, 11]


def test_parent_is_excluded_even_when_ps_missed_it():
    table = [(5, 1, "python3.12", f"{ROOT} -m benchmarks.run")]
    assert _guard.other_model_processes(table, pid=9, ppid=5) == []


def test_single_model_process_refuses_unless_allowed(monkeypatch, capsys):
    monkeypatch.setattr(_guard, "apply_env_caps", lambda environ=None: {})
    monkeypatch.setattr(
        _guard, "other_model_processes", lambda: [(200, "python -m benchmarks.pdf_bench")]
    )
    with pytest.raises(SystemExit) as exc:
        _guard.single_model_process("benchmarks.run")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "another model process is running" in err and "200" in err
    _guard.single_model_process("benchmarks.run", allow_concurrent=True)  # no exit
    monkeypatch.setattr(_guard, "other_model_processes", lambda: [])
    _guard.single_model_process("benchmarks.run")


def test_guard_argument():
    parser = argparse.ArgumentParser()
    _guard.add_guard_argument(parser)
    assert parser.parse_args([]).allow_concurrent is False
    assert parser.parse_args(["--allow-concurrent"]).allow_concurrent is True


def test_process_table_reads_ps():
    rows = _guard._process_table()
    assert rows and all(isinstance(pid, int) and isinstance(cmd, str) for pid, _, _, cmd in rows)
