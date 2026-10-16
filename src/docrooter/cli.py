"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from docrooter.classify.classifier import DEFAULT_BACKEND
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.errors import DocrooterError, TaxonomyError
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


def _parse_strategy(value: str) -> Greedy | Beam | Flat:
    """``greedy``, ``beam:N`` or ``flat`` to a strategy; anything else is a usage error."""
    name, _, arg = value.strip().lower().partition(":")
    if name == "greedy" and not arg:
        return Greedy()
    if name == "flat" and not arg:
        return Flat()
    if name == "beam" and arg.isdigit() and int(arg) >= 1:
        return Beam(int(arg))
    raise typer.BadParameter(
        f"invalid strategy {value!r}; use 'greedy', 'beam:N' (N >= 1) or 'flat'",
        param_hint="--strategy",
    )


@app.command()
def classify(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    taxonomy: Annotated[
        Path,
        typer.Option(
            "--taxonomy",
            "-t",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Taxonomy file (JSON or CSV).",
        ),
    ],
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="Backend spec, e.g. keyword or embedding:MODEL.")
    ] = DEFAULT_BACKEND,
    strategy: Annotated[str, typer.Option(help="Traversal: greedy, beam:N or flat.")] = "beam:3",
    instruction: Annotated[
        str | None, typer.Option(help="Instruction passed to backends that accept one.")
    ] = None,
    threshold: Annotated[float, typer.Option(help="Minimum probability to follow a label.")] = 0.35,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", dir_okay=False, help="Write JSON here and print a table."),
    ] = None,
) -> None:
    """Classify a document's sections against a taxonomy; JSON to stdout unless --out is given."""
    from docrooter.classify.classifier import Classifier
    from docrooter.document.load import load

    chosen = _parse_strategy(strategy)
    console = Console(highlight=False)
    try:
        doc = load(file)
        classifier = Classifier(
            Taxonomy.load(taxonomy),
            backend,
            strategy=chosen,
            instruction=instruction,
            threshold=threshold,
        )
        result = classifier.classify(doc)
    except (DocrooterError, ImportError) as exc:
        Console(stderr=True, highlight=False).print(Text(f"error: {exc}", style="red"))
        raise typer.Exit(1) from exc

    if out is None:
        typer.echo(result.to_json())
        return
    out.write_text(result.to_json(), encoding="utf-8")
    table = Table("Section", "Path", "Score")
    for section in result.sections.values():
        best = section.best
        # Text, not str: titles and labels may contain "[...]" that rich would read as markup.
        if best is not None and best.node_ids:
            path, score = " > ".join(best.labels), f"{best.score:.2f}"
        else:  # no label was chosen; an empty path's score of 1.0 would only mislead
            path = "-"
            score = (best.stopped_reason if best else None) or "-"
        table.add_row(Text(section.title), Text(path), Text(score))
    console.print(table)
    console.print(Text(f"wrote {out}"))


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
