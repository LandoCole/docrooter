"""Memory guard for benchmark entry points: conservative env caps, one model process at a time.

On a 16 GB machine two model-loading benchmarks running together can exhaust memory, so a
model benchmark refuses to start while another one (or ``docrooter serve``) is running, unless
``--allow-concurrent`` is passed. Every entry point also gets conservative defaults: 4 torch
threads, no tokenizer parallelism and an 8 GiB memory budget, from which the MPS watermark
caps are derived (see :mod:`docrooter.runtime`); values already in the environment win.

The caps are environment variables, so they must be set before torch is imported: call
:func:`apply_env_caps` (or :func:`single_model_process`) at the top of ``main``, before any
model loads. The docrooter core never imports torch on its own.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable

from docrooter import runtime

ENV_CAPS: dict[str, str] = {
    "OMP_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "DOCROOTER_THREADS": "4",
    "DOCROOTER_MEMORY_BUDGET_GB": "8",
}
# Also set (from the budget) by apply_env_caps; listed so results can record them.
MPS_CAP_VARS = (runtime.MPS_HIGH_WATERMARK_ENV, runtime.MPS_LOW_WATERMARK_ENV)
# Processes that load models: the model benchmarks (``python -m benchmarks.<name>`` or
# ``benchmarks/<name>.py``) and the playground server.
_MODEL_PROCESS = re.compile(
    r"benchmarks[./](run|pdf_bench|normalize_bench|memory_check|profile_pipeline|elements_bench|ocr_bench)\b"
    r"|\bdocrooter serve\b"
)
# Only an interpreter (or the docrooter script) running one counts; a shell or ``uv run``
# wrapping it, a waiting ``pgrep`` loop or a ``grep`` that merely names it does not.
_INTERPRETER = re.compile(r"^(python[\d.]*|docrooter)$", re.IGNORECASE)


def apply_env_caps() -> dict[str, str]:
    """Set every :data:`ENV_CAPS` variable that is not already set, then the MPS watermark
    variables from the effective runtime settings (the budget, or an explicit
    ``DOCROOTER_MPS_MEMORY_FRACTION``, and any earlier ``configure()``) unless the user set
    them. Returns what it set.

    The watermarks are written through :func:`docrooter.runtime.apply_before_torch`, so the
    runtime knows they are its own: a later ``configure(memory_budget_gb=...)`` still applies
    its exact MPS fraction.
    """
    applied = {}
    for name, value in ENV_CAPS.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[name] = value
    before = {name: os.environ.get(name) for name in MPS_CAP_VARS}
    runtime.apply_before_torch(runtime.settings())
    applied.update(
        {name: os.environ[name] for name in MPS_CAP_VARS if os.environ.get(name) != before[name]}
    )
    return applied


def add_guard_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--allow-concurrent`` to a benchmark's argument parser."""
    parser.add_argument(
        "--allow-concurrent",
        action="store_true",
        help="start even if another model benchmark is running (risks running out of memory)",
    )


Process = tuple[int, int, str, str]  # pid, ppid, program (ps ucomm), command line


def _process_table() -> list[Process]:
    """Every process, from ``ps``; empty if ``ps`` fails."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,ucomm=,command="],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2], parts[3]))
    return rows


def _ancestors(pid: int, parents: dict[int, int]) -> set[int]:
    """``pid`` and every process above it (its launcher shells, ``uv run`` and so on)."""
    chain = set()
    while pid > 1 and pid not in chain:
        chain.add(pid)
        pid = parents.get(pid, 0)
    return chain


def other_model_processes(
    table: Iterable[Process] | None = None,
    *,
    pid: int | None = None,
    ppid: int | None = None,
) -> list[tuple[int, str]]:
    """Other running model processes as ``(pid, command)``.

    Only Python interpreters (or the ``docrooter`` script) running a model benchmark or
    ``docrooter serve`` count: shells, ``uv run`` wrappers and ``pgrep`` loops that merely name
    one do not. This process (``pid``, default ``os.getpid()``), its parent (``ppid``, default
    ``os.getppid()``) and all their ancestors never count.
    """
    rows = list(_process_table() if table is None else table)
    pid = os.getpid() if pid is None else pid
    ppid = os.getppid() if ppid is None else ppid
    parents = {p: pp for p, pp, _, _ in rows}
    parents[pid] = ppid
    mine = _ancestors(pid, parents)
    return [
        (p, command)
        for p, _, program, command in rows
        if p not in mine and _INTERPRETER.match(program) and _MODEL_PROCESS.search(command)
    ]


def single_model_process(name: str, *, allow_concurrent: bool = False) -> None:
    """Set the env caps, then exit with status 2 if another model process is running.

    ``name`` is this benchmark's name, used in the message. With ``allow_concurrent`` the check
    is skipped (the caps still apply).
    """
    apply_env_caps()
    if allow_concurrent:
        return
    others = other_model_processes()
    if others:
        listing = "\n".join(f"  {p}: {command[:160]}" for p, command in others)
        print(
            f"{name}: another model process is running; wait for it to finish, or pass "
            f"--allow-concurrent to start anyway:\n{listing}",
            file=sys.stderr,
        )
        raise SystemExit(2)
