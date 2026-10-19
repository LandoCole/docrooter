"""Collecting the integration tests imports no model library.

CI runs bare ``pytest`` over the whole ``tests`` tree, so every module under
``tests/integration`` is imported at collection time, before the unit tests run. A module-level
``pytest.importorskip("mlx_lm")`` there pulled in transformers and torch and broke the unit tests
that assert torch is not imported. Heavy imports belong inside fixtures and test functions.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SCRIPT = """
import importlib.util, sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted((root / "tests" / "integration").glob("*.py")):
    spec = importlib.util.spec_from_file_location(f"_integration_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # import only: no test runs
heavy = sorted(m for m in ("torch", "mlx", "mlx_lm", "transformers") if m in sys.modules)
print("HEAVY:" + ",".join(heavy))
"""


def test_integration_modules_import_no_model_library():
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(ROOT)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PATH": "/usr/bin:/bin"},
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "HEAVY:", result.stdout
