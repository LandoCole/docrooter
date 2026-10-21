import os
import sys

import pytest

# Plain CLI output whatever the host: Typer forces a Rich terminal (ANSI colours, 80-column
# boxes) when GITHUB_ACTIONS or FORCE_COLOR is set, which breaks the CLI tests' substring
# checks on CI. Set before any test module imports typer.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ["NO_COLOR"] = "1"
os.environ.setdefault("COLUMNS", "200")
os.environ.pop("FORCE_COLOR", None)

# Unit tests must never pull in torch, onnxruntime, mlx or an OCR engine. Snapshot at import
# time so a package that something else imported before the unit tests ran does not count against
# them.
_FORBIDDEN = ("torch", "onnxruntime", "rapidocr", "ocrmac", "mlx", "mlx_lm")
_AT_START = {name for name in _FORBIDDEN if name in sys.modules}
_reported: set[str] = set()


@pytest.fixture(autouse=True)
def _no_heavy_imports():
    yield
    # pymupdf4llm (the `pdf` extra) depends on pymupdf-layout, which imports onnxruntime, and it
    # loads rapidocr when that is installed: neither is docrooter's doing, so a run that has
    # imported pymupdf4llm is not held to those two. torch and ocrmac stay forbidden.
    excused = {"onnxruntime", "rapidocr"} if "pymupdf4llm" in sys.modules else set()
    for name in _FORBIDDEN:
        if name in _AT_START or name in _reported or name in excused:
            continue
        if name in sys.modules:
            _reported.add(name)  # name the first offender only; later tests would just inherit it
            pytest.fail(f"this unit test imported {name} (unit tests must not)", pytrace=False)
