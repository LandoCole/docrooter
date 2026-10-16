"""``benchmarks._guard``: env caps and the one-model-process-at-a-time check."""

import argparse

import pytest

from benchmarks import _guard

ROOT = "/Users/me/Doc Rooter/.venv/bin/python"


def test_apply_env_caps_keeps_existing_values(monkeypatch):
    from docrooter import runtime

    monkeypatch.setattr(runtime, "estimated_recommended_bytes", lambda: 16 * 1024**3)
    env = {"OMP_NUM_THREADS": "2"}
    applied = _guard.apply_env_caps(env)
    assert env["OMP_NUM_THREADS"] == "2" and "OMP_NUM_THREADS" not in applied
    assert env["TOKENIZERS_PARALLELISM"] == "false" and env["DOCROOTER_THREADS"] == "4"
    assert env["DOCROOTER_MEMORY_BUDGET_GB"] == "8"
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.5"  # 8 GiB of an estimated 16
    assert env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.4"
    assert _guard.apply_env_caps(env) == {}


def test_mps_caps_follow_the_budget_and_respect_the_user(monkeypatch):
    from docrooter import runtime

    monkeypatch.setattr(runtime, "estimated_recommended_bytes", lambda: 16 * 1024**3)
    env = {"DOCROOTER_MEMORY_BUDGET_GB": "4"}
    _guard.apply_env_caps(env)
    assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.25"
    mine = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.9"}
    _guard.apply_env_caps(mine)
    assert mine["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.9"
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in mine


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
]


def test_other_model_processes_skip_self_ancestors_wrappers_and_non_model_runs():
    found = _guard.other_model_processes(TABLE, pid=102, ppid=101)
    assert [pid for pid, _ in found] == [200, 500, 700]


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
