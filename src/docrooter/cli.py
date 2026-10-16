"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, help="Document TOC extraction and section classification.")


@app.callback()
def main() -> None:
    """Document TOC extraction and section classification."""


@app.command()
def toc(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Print the TOC as a JSON list.")] = False,
) -> None:
    """Print a document's table of contents."""
    from docrooter.document.load import load

    entries = load(file).toc()
    if as_json:
        typer.echo(json.dumps([e.model_dump() for e in entries], indent=2))
        return
    for entry in entries:
        typer.echo("  " * entry.depth + entry.title)
