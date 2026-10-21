"""Test-wide environment: plain CLI output whatever the host.

Typer forces a Rich terminal (ANSI colours, 80-column boxes) when ``GITHUB_ACTIONS`` or
``FORCE_COLOR`` is set, which breaks the CLI tests' substring checks on CI. These must be set
before typer is imported, so they live here, not in a fixture.
"""

import os

os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ["NO_COLOR"] = "1"
os.environ.setdefault("COLUMNS", "200")
os.environ.pop("FORCE_COLOR", None)
