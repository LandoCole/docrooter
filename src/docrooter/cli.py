"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from docrooter.errors import TaxonomyError
from docrooter.taxonomy import Taxonomy, TaxonomyIssue, validate_nodes

app = typer.Typer(no_args_is_help=True, help="Document TOC extraction and section classification.")
taxonomy_app = typer.Typer(no_args_is_help=True, help="Work with taxonomy files.")
app.add_typer(taxonomy_app, name="taxonomy")


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


@taxonomy_app.callback()
def taxonomy_main() -> None:
    """Work with taxonomy files."""


def _print_issues(console: Console, issues: list[TaxonomyIssue]) -> None:
    table = Table("Severity", "Code", "Location", "Message")
    for issue in issues:
        style = "red" if issue.severity == "error" else "yellow"
        # Text, not str: labels and messages may contain "[...]" that rich would read as markup.
        table.add_row(
            Text(issue.severity, style=style),
            Text(issue.code),
            Text(issue.location or "-"),
            Text(issue.message),
        )
    console.print(table)


@taxonomy_app.command("validate")
def taxonomy_validate(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Taxonomy file.")
    ],
    sep: Annotated[str, typer.Option(help="Path separator for 'path' CSV files.")] = ">",
) -> None:
    """Check a JSON or CSV taxonomy file; exit 1 if it has errors."""
    from docrooter.taxonomy.io import read_nodes

    console = Console(highlight=False)
    try:
        parsed = read_nodes(file, sep=sep)
    except TaxonomyError as exc:
        # Parse failures carry their own issues; show them like validation errors.
        _print_issues(console, exc.issues)
        raise typer.Exit(1) from exc
    issues = validate_nodes(parsed.nodes, locations=parsed.locations)
    if any(i.severity == "error" for i in issues):
        _print_issues(console, issues)
        raise typer.Exit(1)
    taxonomy = Taxonomy(parsed.nodes, locations=parsed.locations, root=parsed.root)
    console.print(f"valid: {len(taxonomy)} nodes, depth {taxonomy.depth()}")
    if issues:
        _print_issues(console, issues)
