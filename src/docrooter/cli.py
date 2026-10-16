"""Command-line interface."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import pydantic
import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from docrooter._extras import require
from docrooter.classify.classifier import DEFAULT_BACKEND
from docrooter.classify.strategies import Beam, Flat, Greedy
from docrooter.errors import DocrooterError, TaxonomyError
from docrooter.taxonomy import Taxonomy, TaxonomyIssue, validate_nodes

app = typer.Typer(no_args_is_help=True, help="Document TOC extraction and section classification.")
taxonomy_app = typer.Typer(no_args_is_help=True, help="Work with taxonomy files.")
app.add_typer(taxonomy_app, name="taxonomy")
models_app = typer.Typer(no_args_is_help=True, help="Download and inspect model weights.")
app.add_typer(models_app, name="models")


# What a command reports as "error: ..." with exit status 1; anything else is a bug and shows
# its traceback.
_EXPECTED_ERRORS = (DocrooterError, ImportError, ValueError, OSError, pydantic.ValidationError)


@contextmanager
def _errors_exit_1() -> Iterator[None]:
    """Print an expected error as ``error: <message>`` on stderr and exit with status 1."""
    try:
        yield
    except _EXPECTED_ERRORS as exc:
        # Text, not markup: messages quote user text that may contain "[...]". soft_wrap keeps
        # a copy-pasteable "pip install 'docrooter[...]'" on one line.
        Console(stderr=True, highlight=False).print(
            Text(f"error: {exc}", style="red"), soft_wrap=True
        )
        raise typer.Exit(1) from exc


@app.callback()
def main() -> None:
    """Document TOC extraction and section classification."""


@app.command()
def toc(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    as_json: Annotated[bool, typer.Option("--json", help="Print the TOC as a JSON list.")] = False,
    engine: Annotated[
        str,
        typer.Option(
            help="PDF engine: auto, pymupdf, pdfplumber, both or pymupdf4llm (ignored otherwise)."
        ),
    ] = "auto",
) -> None:
    """Print a document's table of contents; PDFs also get a one-line summary on stderr."""
    from docrooter.document.load import load
    from docrooter.pdf.load import ENGINES

    if engine not in ENGINES:
        raise typer.BadParameter(
            f"invalid engine {engine!r}; use one of {', '.join(ENGINES)}", param_hint="--engine"
        )
    is_pdf = file.suffix.lower() == ".pdf"
    with _errors_exit_1():
        started = time.perf_counter()
        doc = load(file, engine=engine) if is_pdf else load(file)
        seconds = time.perf_counter() - started
        entries = doc.toc()
    if as_json:
        typer.echo(json.dumps([e.model_dump() for e in entries], indent=2))
        return
    for entry in entries:
        typer.echo("  " * entry.depth + entry.title)
    if is_pdf:
        typer.echo(
            f"{doc.meta.get('n_pages', '?')} pages, {len(entries)} sections, {seconds:.2f}s, "
            f"engine {doc.meta.get('engine', engine)}",
            err=True,
        )


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
    with _errors_exit_1():
        doc = load(file)
        classifier = Classifier(
            Taxonomy.load(taxonomy),
            backend,
            strategy=chosen,
            instruction=instruction,
            threshold=threshold,
        )
        result = classifier.classify(doc)
        if out is not None:
            out.write_text(result.to_json(), encoding="utf-8")

    if out is None:
        typer.echo(result.to_json())
        return
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


@app.command("eval")
def eval_(
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
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Labelled examples (JSONL or CSV).",
        ),
    ],
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="Backend spec, e.g. keyword or embedding:MODEL.")
    ] = DEFAULT_BACKEND,
    strategy: Annotated[str, typer.Option(help="Traversal: greedy, beam:N or flat.")] = "beam:3",
    threshold: Annotated[float, typer.Option(help="Minimum probability to follow a label.")] = 0.35,
    fmt: Annotated[
        str, typer.Option("--format", help="Output format: table, md or json.")
    ] = "table",
) -> None:
    """Evaluate a backend on a labelled dataset: accuracy, hierarchical F1, speed."""
    from docrooter.classify.classifier import Classifier
    from docrooter.evaluate import evaluate
    from docrooter.train.dataset import Dataset

    if fmt not in ("table", "md", "json"):
        raise typer.BadParameter(
            f"invalid format {fmt!r}; use 'table', 'md' or 'json'", param_hint="--format"
        )
    chosen = _parse_strategy(strategy)
    with _errors_exit_1():
        classifier = Classifier(
            Taxonomy.load(taxonomy), backend, strategy=chosen, threshold=threshold
        )
        report = evaluate(classifier, Dataset.load(dataset))

    if fmt == "json":
        typer.echo(report.to_json())
    elif fmt == "md":
        typer.echo(report.to_markdown())
    else:
        report.print()


@app.command("train")
def train_(
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
    dataset: Annotated[
        Path,
        typer.Option(
            "--dataset",
            "-d",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Labelled examples (JSONL or CSV).",
        ),
    ],
    out: Annotated[
        Path, typer.Option("--out", "-o", file_okay=False, help="Directory to write the bundle to.")
    ],
    base: Annotated[str, typer.Option("--base", help="Base embedding model.")] = (
        "dmis-lab/UltraBERT-Embed-150M"
    ),
    finetune_body: Annotated[
        bool,
        typer.Option(
            "--finetune-body/--no-finetune-body",
            help="Fine-tune the shared body before fitting the heads.",
        ),
    ] = True,
    epochs: Annotated[
        int | None, typer.Option(min=1, help="Fine-tuning epochs (default: the model recipe's).")
    ] = None,
    lr: Annotated[
        float | None, typer.Option(min=0, help="Learning rate (default: the model recipe's).")
    ] = None,
    min_examples: Annotated[
        int, typer.Option(min=1, help="Examples each child needs for its node to get a head.")
    ] = 4,
) -> None:
    """Train per-node heads (and fine-tune the shared body) and save a model bundle."""
    from docrooter.train import trainer
    from docrooter.train.dataset import Dataset

    console = Console(highlight=False)
    with _errors_exit_1():
        model = trainer.train(
            Taxonomy.load(taxonomy),
            Dataset.load(dataset),
            base_model=base,
            finetune_body=finetune_body,
            epochs=epochs,
            lr=lr,
            min_examples_per_child=min_examples,
        )
        model.save(out)

    manifest = model.manifest
    table = Table("Node with head", "Children", "Examples")
    for node_id, info in manifest["heads"].items():
        # Text, not str: ids may contain "[...]" that rich would read as markup.
        table.add_row(Text(node_id), Text(", ".join(info["classes"])), Text(str(info["n"])))
    console.print(table)
    metrics = manifest["train_metrics"]
    console.print(
        Text(
            f"{len(manifest['heads'])} head(s), {metrics['triplets']} triplet(s), "
            f"{metrics['seconds']:.1f}s; wrote {out}"
        )
    )


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
    with _errors_exit_1():
        try:
            parsed = read_nodes(file, sep=sep)
        except TaxonomyError as exc:
            if not exc.issues:
                raise
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


@models_app.callback()
def models_main() -> None:
    """Download and inspect model weights."""


@models_app.command("pull")
def models_pull(
    model_id: Annotated[
        str, typer.Argument(help="Hugging Face repo id, e.g. BAAI/bge-small-en-v1.5.")
    ],
    revision: Annotated[
        str | None, typer.Option(help="Commit, tag or branch; defaults to the recipe's pin.")
    ] = None,
) -> None:
    """Download a model into the Hugging Face cache and print where it landed."""
    from docrooter.backends.recipes import get_recipe

    with _errors_exit_1():
        hub = require("huggingface_hub", "models")
        rev = revision or get_recipe(model_id).revision
        path = hub.snapshot_download(repo_id=model_id, revision=rev)
    typer.echo(str(path))


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    raise AssertionError("unreachable")


@models_app.command("list")
def models_list() -> None:
    """List models in the Hugging Face cache."""
    console = Console(highlight=False)
    with _errors_exit_1():
        hub = require("huggingface_hub", "models")
        repos = [
            r for r in hub.scan_cache_dir().repos if getattr(r, "repo_type", "model") == "model"
        ]
    if not repos:
        console.print("no cached models")
        return
    table = Table("Model", "Revisions", "Size")
    for repo in sorted(repos, key=lambda r: r.repo_id):
        revisions = ", ".join(sorted(r.commit_hash[:8] for r in repo.revisions)) or "-"
        table.add_row(
            Text(repo.repo_id, no_wrap=True), Text(revisions), Text(_human_size(repo.size_on_disk))
        )
    console.print(table)
