"""Generate the CLI reference: run ``docrooter <command> --help`` for every command.

The docs build calls :func:`render` (see ``docs/hooks.py``), so the reference always matches
the code. Run it by hand to see the page::

    PYTHONPATH=src python docs/gen_cli.py            # Markdown on stdout
    PYTHONPATH=src python docs/gen_cli.py -o cli.md  # or into a file

Each command runs in its own process, with plain (non-rich) help at a fixed width, so the
output does not depend on the terminal. Defaults that depend on the machine (``doctype
--workers``) show the value of the machine that built the docs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 80  # fits the content column without horizontal scrolling
# Runs the Typer app under the name users type, without relying on the console script.
_RUNNER = f"from docrooter.cli import app; app(prog_name='docrooter', max_content_width={WIDTH})"

INTRO = """\
# CLI reference

Every `docrooter` command with its options, generated from `docrooter <command> --help` when
these docs were built (by `docs/gen_cli.py`). Each command also prints this with `--help`.

- Errors are printed as `error: ...` on stderr, with exit status 1.
- `-b/--backend` takes a backend spec such as `keyword`, `embedding:BAAI/bge-small-en-v1.5`,
  `jev` or `openai:gpt-4o-mini`; see [Backends](../concepts/backends.md).
- Commands that need an extra (PDF engines, models, the playground) say which one to install
  when it is missing.
"""


def commands() -> list[list[str]]:
    """Every command path, in the order ``docrooter --help`` lists them, groups before their
    subcommands. Walks the Click tree iteratively."""
    import typer

    from docrooter.cli import app

    root = typer.main.get_command(app)
    found: list[list[str]] = []
    stack: list[tuple[list[str], Any]] = [([], root)]
    while stack:
        path, command = stack.pop()
        found.append(path)
        children = list(getattr(command, "commands", {}).items())  # a group's subcommands
        if children:
            stack.extend((path + [name], child) for name, child in reversed(children))
    return found


def help_text(path: list[str]) -> str:
    """``docrooter <path> --help`` as plain text."""
    env = dict(os.environ)
    env.update({"TYPER_USE_RICH": "0", "COLUMNS": str(WIDTH), "NO_COLOR": "1"})
    src = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    done = subprocess.run(
        [sys.executable, "-c", _RUNNER, *path, "--help"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=120,
    )
    if done.returncode != 0:
        raise RuntimeError(f"docrooter {' '.join(path)} --help failed:\n{done.stderr}")
    return done.stdout.rstrip()


def render() -> str:
    """The whole CLI reference page as Markdown."""
    parts = [INTRO]
    for path in commands():
        title = " ".join(["docrooter", *path])
        level = "#" * min(2 + max(len(path) - 1, 0), 4)
        parts.append(f"{level} `{title}`\n\n```text\n{help_text(path)}\n```\n")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-o", "--out", type=Path, help="write here instead of stdout")
    args = parser.parse_args(argv)
    text = render()
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
