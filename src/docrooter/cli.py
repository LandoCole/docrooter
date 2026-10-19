"""Command-line interface."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

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

if TYPE_CHECKING:
    from docrooter.classify.classifier import Classifier
    from docrooter.classify.results import DocTypeResult

app = typer.Typer(no_args_is_help=True, help="Document TOC extraction and section classification.")
taxonomy_app = typer.Typer(no_args_is_help=True, help="Work with taxonomy files.")
app.add_typer(taxonomy_app, name="taxonomy")
models_app = typer.Typer(no_args_is_help=True, help="Download and inspect model weights.")
app.add_typer(models_app, name="models")
config_app = typer.Typer(no_args_is_help=True, help="Inspect the rooter configuration.")
app.add_typer(config_app, name="config")
corpus_app = typer.Typer(no_args_is_help=True, help="Build and inspect corpus stores.")
app.add_typer(corpus_app, name="corpus")


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


def _print_version(value: bool) -> None:
    if value:
        import docrooter

        typer.echo(f"docrooter {docrooter.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the docrooter version and exit.",
        ),
    ] = False,
) -> None:
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


# --- doctype and normalize ----------------------------------------------------------------

_DOC_SUFFIXES = (".pdf", ".md", ".txt")
_SKIPPED_DIRS = frozenset({".venv", "node_modules", "__pycache__"})  # and every hidden directory
_DEFAULT_DOCTYPE_TAXONOMY = "builtin:doc_types"
_BUILTIN_PREFIX = "builtin:"


def _default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _load_taxonomy_spec(spec: str) -> Taxonomy:
    """A taxonomy from ``builtin:<name>`` or a file path."""
    if spec.startswith(_BUILTIN_PREFIX):
        from docrooter.taxonomies import builtin_taxonomy

        try:
            return builtin_taxonomy(spec[len(_BUILTIN_PREFIX) :])
        except KeyError as exc:
            raise ValueError(exc.args[0]) from exc
    return Taxonomy.load(Path(spec))


def _expand_paths(paths: list[Path]) -> list[Path]:
    """Files as given; directories expanded recursively to ``.pdf``, ``.md`` and ``.txt``
    files in sorted order, skipping hidden directories, ``node_modules`` and ``__pycache__``
    (and so ``.venv``). Duplicates are dropped."""
    out: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            found: list[Path] = []
            for folder, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _SKIPPED_DIRS]
                found += [
                    Path(folder) / name
                    for name in files
                    if Path(name).suffix.lower() in _DOC_SUFFIXES
                    and (Path(folder) / name).is_file()
                ]
            out.update(dict.fromkeys(sorted(found)))
        else:
            out[path] = None
    return list(out)


def _parse_one(
    path: str, engine: str, max_pages: int | None = None
) -> tuple[dict | None, float, str | None]:
    """Load one file: ``(Document.model_dump(), seconds, None)`` or ``(None, seconds, error)``.

    Module level and picklable results, so a process pool can run it. Never raises: a bad file
    must not stop the batch. The error message does not repeat the path (rows show it).
    ``max_pages`` caps the pages read from a PDF.
    """
    from docrooter.document.load import load

    started = time.perf_counter()
    try:
        if path.lower().endswith(".pdf"):
            doc = load(Path(path), engine=engine, max_pages=max_pages)
        else:
            doc = load(Path(path))
        return doc.model_dump(), time.perf_counter() - started, None
    except Exception as exc:
        message = _without_path(str(exc) or type(exc).__name__, path)
        return None, time.perf_counter() - started, message


def _without_path(message: str, path: str) -> str:
    """``message`` without the file's path: ``"cannot open PDF <path>: Failed to open file
    '<path>'"`` becomes ``"cannot open PDF: Failed to open the file"``."""
    message = message.replace(f" {path}:", ":")
    return message.replace(f"file '{path}'", "the file").replace(f"'{path}'", "the file")


_Parsed = tuple[dict | None, float, str | None]


def _parse_chunks(
    paths: list[Path],
    engine: str,
    workers: int,
    chunk_size: int,
    max_pages: int | None = None,
) -> Generator[list[_Parsed], None, None]:
    """Parse the files ``chunk_size`` at a time, yielding each chunk's results in order, so
    only one chunk of parsed documents is held at once.

    With several ``workers`` and files, one process pool serves every chunk. If a worker dies
    (``BrokenProcessPool``), the files without a result, and every later chunk, are parsed in
    this process, one by one.
    """
    names = [str(p) for p in paths]
    chunks = [names[i : i + chunk_size] for i in range(0, len(names), chunk_size)]
    if workers <= 1 or len(names) <= 1:
        for chunk in chunks:
            yield _parse_remaining(chunk, engine, max_pages=max_pages)
        return
    import concurrent.futures
    from concurrent.futures import BrokenExecutor

    broken = False
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(names))) as pool:
        for n, chunk in enumerate(chunks):
            parsed: list[_Parsed | None] = [None] * len(chunk)
            if not broken:
                try:
                    futures = [pool.submit(_parse_one, name, engine, max_pages) for name in chunk]
                    for i, future in enumerate(futures):
                        parsed[i] = future.result()
                except BrokenExecutor:
                    broken = True
                    left = sum(r is None for r in parsed) + sum(map(len, chunks[n + 1 :]))
                    Console(stderr=True, highlight=False).print(
                        Text(
                            f"warning: a parser process died; parsing the remaining {left} "
                            "file(s) in this process",
                            style="yellow",
                        ),
                        soft_wrap=True,
                    )
            todo = [i for i, result in enumerate(parsed) if result is None]
            redone = _parse_remaining([chunk[i] for i in todo], engine, max_pages=max_pages)
            for i, result in zip(todo, redone, strict=True):
                parsed[i] = result
            yield [result for result in parsed if result is not None]


def _parse_remaining(
    names: list[str],
    engine: str,
    parse: Callable[..., _Parsed] = _parse_one,
    *,
    max_pages: int | None = None,
) -> list[_Parsed]:
    """Parse ``names`` in this process; a file whose parser raises anyway becomes an error."""
    out: list[_Parsed] = []
    for name in names:
        started = time.perf_counter()
        try:
            out.append(parse(name, engine, max_pages))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            message = _without_path(str(exc) or type(exc).__name__, name)
            out.append((None, time.perf_counter() - started, message))
    return out


@dataclass
class _DocRow:
    """What ``doctype`` reports for one file."""

    source: str
    parse_s: float
    error: str | None = None
    result: DocTypeResult | None = None

    @property
    def node_ids(self) -> list[str]:
        return self.result.best.node_ids if self.result else []

    def record(self) -> dict[str, Any]:
        """The JSONL record: the leaf id as ``type``, its label, and the ids from the top."""
        result = self.result
        best = result.best if result else None
        return {
            "source": self.source,
            "type": best.node_ids[-1] if best and best.node_ids else None,
            "label": best.labels[-1] if best and best.labels else None,
            "path": list(best.node_ids) if best else [],
            "confidence": result.confidence if result else None,
            "leaf_probs": result.leaf_probs if result else {},
            "timings": {
                "parse": self.parse_s,
                "classify": result.timings.get("classify", 0.0) if result else None,
            },
            "error": self.error,
        }

    def cells(self) -> tuple[Text, str, str, str]:
        """Type, confidence, parse seconds and classify seconds, as table cells."""
        parse = f"{self.parse_s:.2f}"
        if self.error is not None or self.result is None:
            return Text(f"error: {self.error}", style="red"), "-", parse, "-"
        best, classify = self.result.best, f"{self.result.timings.get('classify', 0.0):.3f}"
        confidence = f"{self.result.confidence:.2f}"
        if not best.node_ids:  # nothing assigned: say why instead of leaving it blank
            return (
                Text(f"({best.stopped_reason or 'none'})", style="yellow"),
                confidence,
                parse,
                classify,
            )
        return Text(" > ".join(best.labels)), confidence, parse, classify


def _classify_chunk(
    classifier: Classifier, files: list[Path], parsed: list[_Parsed], mode: str, threshold: float
) -> list[_DocRow]:
    """One row per file of a chunk, classifying the files that parsed."""
    from docrooter.document import Document

    indexes = [i for i, (data, _, _) in enumerate(parsed) if data is not None]
    results: dict[int, DocTypeResult] = {}
    if indexes:
        docs = [Document.model_validate(parsed[i][0]) for i in indexes]
        classified = classifier.classify_documents(docs, mode=mode, threshold=threshold)
        results = dict(zip(indexes, classified, strict=True))
    return [
        _DocRow(str(path), parse_s, error, results.get(i))
        for i, (path, (_, parse_s, error)) in enumerate(zip(files, parsed, strict=True))
    ]


@app.command()
def doctype(
    paths: Annotated[
        list[Path],
        typer.Argument(
            exists=True, readable=True, help="Documents or directories (searched recursively)."
        ),
    ],
    taxonomy: Annotated[
        str,
        typer.Option("--taxonomy", "-t", help="Taxonomy file (JSON or CSV) or builtin:<name>."),
    ] = _DEFAULT_DOCTYPE_TAXONOMY,
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="Backend spec, e.g. keyword or embedding:MODEL.")
    ] = DEFAULT_BACKEND,
    mode: Annotated[
        str, typer.Option(help="What is classified: view, sections or hybrid.")
    ] = "view",
    engine: Annotated[
        str, typer.Option(help="PDF engine: auto, pymupdf, pdfplumber, both or pymupdf4llm.")
    ] = "auto",
    keep: Annotated[
        list[str] | None,
        typer.Option(
            "--keep",
            help="Only list files whose type, or an ancestor of it, has this id (repeatable).",
        ),
    ] = None,
    paths_only: Annotated[
        bool, typer.Option("--paths-only", help="Print only the (kept) file paths, one per line.")
    ] = False,
    jsonl: Annotated[
        Path | None,
        typer.Option("--jsonl", dir_okay=False, help="Also write one JSON record per file here."),
    ] = None,
    workers: Annotated[
        int, typer.Option(min=1, help="Parser processes (default: min(4, CPU count)).")
    ] = _default_workers(),
    threshold: Annotated[
        float, typer.Option(help="Minimum confidence to assign a type (0 always labels).")
    ] = 0.0,
    chunk_size: Annotated[
        int,
        typer.Option(
            "--chunk-size", min=1, help="Files parsed and classified together (bounds memory)."
        ),
    ] = 32,
    max_pages: Annotated[
        int | None,
        typer.Option("--max-pages", min=1, help="Read at most this many pages of each PDF."),
    ] = None,
) -> None:
    """Classify each document's type; a table by default, JSONL with --jsonl.

    Files are handled --chunk-size at a time (parsed, classified, written, then dropped), so
    memory does not grow with the number of files.
    """
    from docrooter.classify.classifier import Classifier
    from docrooter.classify.documents import MODES
    from docrooter.pdf.load import ENGINES

    if mode not in MODES:
        raise typer.BadParameter(
            f"invalid mode {mode!r}; use one of {', '.join(MODES)}", param_hint="--mode"
        )
    if engine not in ENGINES:
        raise typer.BadParameter(
            f"invalid engine {engine!r}; use one of {', '.join(ENGINES)}", param_hint="--engine"
        )
    with _errors_exit_1():
        tax = _load_taxonomy_spec(taxonomy)
        keep_ids = list(dict.fromkeys(keep or []))
        unknown = [k for k in keep_ids if k not in tax]
        if unknown:
            raise typer.BadParameter(
                f"unknown id(s) {', '.join(map(repr, unknown))}; valid ids: {', '.join(tax.nodes)}",
                param_hint="--keep",
            )
        files = _expand_paths(paths)
        if not files:
            raise ValueError("no .pdf, .md or .txt files found")

        wanted = set(keep_ids)
        warn = Console(stderr=True, highlight=False)
        # (path, parsed without error, table cells): all a row needs once its chunk is gone
        shown: list[tuple[str, bool, tuple[Text, str, str, str]]] = []
        out = jsonl.open("w", encoding="utf-8") if jsonl is not None else None
        try:
            with (
                Classifier(tax, backend) as classifier,
                closing(_parse_chunks(files, engine, workers, chunk_size, max_pages)) as chunks,
            ):
                for start, parsed in zip(range(0, len(files), chunk_size), chunks, strict=True):
                    batch = files[start : start + chunk_size]
                    rows = _classify_chunk(classifier, batch, parsed, mode, threshold)
                    del parsed  # the documents: only the rows' results outlive the chunk
                    for row in rows:
                        if keep_ids:
                            if row.error is not None:
                                warn.print(
                                    Text(f"warning: {row.source}: {row.error}", style="yellow"),
                                    soft_wrap=True,
                                )
                            if not wanted & set(row.node_ids):
                                continue
                        if out is not None:
                            out.write(json.dumps(row.record(), ensure_ascii=False) + "\n")
                        shown.append((row.source, row.error is None, row.cells()))
                    if out is not None:
                        out.flush()
        finally:
            if out is not None:
                out.close()

    if paths_only:
        for source, ok, _ in shown:
            if ok:
                typer.echo(source)
        return
    table = Table("File", "Type", "Confidence", "Parse s", "Classify s")
    for source, _, cells in shown:
        # Text, not str: paths and labels may contain "[...]" that rich would read as markup.
        table.add_row(Text(source), *cells)
    Console(highlight=False).print(table)


def _parse_levels(value: str) -> list[int]:
    try:
        levels = [int(part) for part in value.split(",") if part.strip()]
    except ValueError:
        levels = []
    if not levels or any(level < 1 for level in levels):
        raise typer.BadParameter(
            f"invalid levels {value!r}; use comma-separated positive integers, e.g. 1,2",
            param_hint="--levels",
        )
    return levels


@app.command()
def normalize(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    doc_type: Annotated[
        str | None,
        typer.Option(help="Document-type id choosing the section taxonomy (default: classify it)."),
    ] = None,
    backend: Annotated[
        str, typer.Option("--backend", "-b", help="Backend spec, e.g. keyword or embedding:MODEL.")
    ] = DEFAULT_BACKEND,
    levels: Annotated[str, typer.Option(help="Heading levels to normalise, e.g. 1,2.")] = "1,2",
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the result as JSON instead of a table.")
    ] = False,
) -> None:
    """Map each section title to a canonical section label of the document's type."""
    from docrooter.classify.normalize import DOC_TYPES, normalize_sections
    from docrooter.document.load import load
    from docrooter.taxonomies import builtin_taxonomy

    chosen_levels = _parse_levels(levels)
    with _errors_exit_1():
        if doc_type is not None and doc_type not in (known := builtin_taxonomy(DOC_TYPES)):
            raise typer.BadParameter(
                f"unknown document type {doc_type!r}; valid ids: {', '.join(known.nodes)}",
                param_hint="--doc-type",
            )
        doc = load(file)
        result = normalize_sections(doc, doc_type=doc_type, backend=backend, levels=chosen_levels)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    console = Console(highlight=False)
    confidence = (
        f" ({result.doc_type_confidence:.2f})" if result.doc_type_confidence is not None else ""
    )
    console.print(
        Text(f"doc type: {result.doc_type or '-'}{confidence}; taxonomy: {result.taxonomy}")
    )
    table = Table("Section", "Canonical", "Score")
    for norm in result.sections.values():
        if norm.canonical is None:
            table.add_row(Text(norm.title), Text("-"), "-")
        else:
            table.add_row(Text(norm.title), Text(norm.label or norm.canonical), f"{norm.score:.2f}")
    console.print(table)


# --- elements, process and config ------------------------------------------------------------

_PROFILE_HELP = "Rooter profile: fast, balanced or accurate (default: balanced, or the file's)."
_CONFIG_HELP = "A TOML or JSON router configuration (see 'docrooter config show')."
_TEXT_COLUMN_CHARS = 80


def _check_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    from docrooter.rooter.profiles import PROFILES

    if profile not in PROFILES:
        raise typer.BadParameter(
            f"invalid profile {profile!r}; use one of {', '.join(PROFILES)}", param_hint="--profile"
        )
    return profile


def _parse_kinds(value: str | None) -> frozenset[str] | None:
    """The ``--kinds`` list as a set of element kinds; anything unknown is a usage error."""
    if value is None:
        return None
    from typing import get_args

    from docrooter.elements.models import ElementKind

    known = get_args(ElementKind)
    kinds = [k.strip().lower() for k in value.split(",") if k.strip()]
    unknown = [k for k in kinds if k not in known]
    if not kinds or unknown:
        raise typer.BadParameter(
            f"invalid kinds {', '.join(unknown) or value!r}; use any of {', '.join(known)}",
            param_hint="--kinds",
        )
    return frozenset(kinds)


def _element_config(profile: str | None, config: Path | None, flag: str) -> Any:
    """The resolved configuration, checked before any work: ``flag`` needs elements, and the
    ``fast`` profile (layout ``none``) builds none."""
    from docrooter.rooter import RouterConfig

    resolved = RouterConfig.resolve(profile=_check_profile(profile), file=config)
    if resolved.layout == "none":
        raise DocrooterError(
            f"{flag} needs elements, and profile {resolved.profile!r} builds none (layout is "
            "'none'); use --profile balanced or accurate"
        )
    return resolved


def _check_writable(out: Path) -> None:
    """Refuse (usage error) an ``--out`` that could not be written, before the run."""
    folder = out.parent
    if not folder.is_dir():
        raise typer.BadParameter(f"the folder {folder} does not exist", param_hint="--out")
    if not os.access(folder, os.W_OK) or (out.exists() and not os.access(out, os.W_OK)):
        raise typer.BadParameter(f"{out} is not writable", param_hint="--out")


def _first_chars(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@app.command()
def elements(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    profile: Annotated[str | None, typer.Option(help=_PROFILE_HELP)] = None,
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help=_CONFIG_HELP),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Print the elements as a JSON list.")
    ] = False,
    kinds: Annotated[
        str | None,
        typer.Option(help="Only these element kinds, comma-separated, e.g. heading,table."),
    ] = None,
) -> None:
    """Print a document's elements (headings, paragraphs, tables, ...) in reading order.

    The profile and file choose the text, OCR and layout engines; nothing is classified.
    """
    from docrooter.rooter import process

    wanted = _parse_kinds(kinds)
    with _errors_exit_1():
        resolved = _element_config(profile, config, "docrooter elements")
        analysis = process(
            file,
            config=resolved,
            doc_type="none",
            sections="none",
            summaries="none",
            escalate="none",
            embeddings="none",
        )
        doc = analysis.document
    found = [el for el in doc.elements if wanted is None or el.kind in wanted]
    if as_json:
        typer.echo(json.dumps([el.model_dump(mode="json") for el in found], indent=2))
        return
    console = Console(highlight=False)
    table = Table("#", "Page", "Kind", "Text", "Section")
    for el in found:
        page = el.boxes[0].page if el.boxes else None
        section = doc.sections.get(el.section_id) if el.section_id else None
        table.add_row(
            str(el.order),
            "-" if page is None else str(page),
            Text(el.kind),
            Text(_first_chars(el.text, _TEXT_COLUMN_CHARS)),
            Text(section.title if section is not None else "-"),
        )
    console.print(table)
    console.print(Text(f"{len(found)} of {len(doc.elements)} elements"), style="dim")


@app.command("process")
def process_(
    file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, readable=True, help="Document to read.")
    ],
    profile: Annotated[str | None, typer.Option(help=_PROFILE_HELP)] = None,
    config: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, readable=True, help=_CONFIG_HELP),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", dir_okay=False, help="Write the analysis JSON here."),
    ] = None,
    chunks: Annotated[
        bool,
        typer.Option("--chunks", help="Add section-aware chunks (for indexing) under 'chunks'."),
    ] = False,
) -> None:
    """Run the rooter on a document and write the whole analysis as JSON.

    Loads the document (OCR and layout as the profile says), classifies its type and sections,
    and summarises it. The JSON goes to stdout unless --out is given.
    """
    from docrooter.chunking import chunk
    from docrooter.rooter import RouterConfig, process

    if out is not None:
        _check_writable(out)
    with _errors_exit_1():
        resolved = (
            _element_config(profile, config, "--chunks")
            if chunks
            else RouterConfig.resolve(profile=_check_profile(profile), file=config)
        )
        started = time.perf_counter()
        analysis = process(file, config=resolved)
        if chunks:
            data = json.loads(analysis.to_json())
            data["chunks"] = [c.model_dump(mode="json") for c in chunk(analysis)]
            text = json.dumps(data, indent=2)
        else:
            text = analysis.to_json(indent=2)
        if out is not None:
            out.write_text(text + "\n", encoding="utf-8")
    if out is None:
        typer.echo(text)
    doc = analysis.document
    typer.echo(
        f"{analysis.config.profile}: {len(doc.sections)} sections, {len(doc.elements)} elements, "
        f"{time.perf_counter() - started:.2f}s" + (f", wrote {out}" if out is not None else ""),
        err=True,
    )


def _toml_value(value: object) -> str:
    return json.dumps(value) if isinstance(value, str) else str(value)


@config_app.command("show")
def config_show(
    profile: Annotated[
        str | None, typer.Option(help="Profile to resolve (default: balanced).")
    ] = None,
) -> None:
    """Print the resolved rooter configuration as TOML you can save and edit.

    The accurate profile escalates to Jev only when TYPESAFE_API_KEY is set.
    """
    from docrooter.rooter import RouterConfig

    with _errors_exit_1():
        resolved = RouterConfig.resolve(profile=_check_profile(profile))
    for key, value in resolved.model_dump().items():
        if value is not None:  # max_pages unset: TOML has no null
            typer.echo(f"{key} = {_toml_value(value)}")


# --- corpus -------------------------------------------------------------------------------


@corpus_app.command("build")
def corpus_build(
    directory: Annotated[
        Path, typer.Argument(file_okay=False, help="Corpus store directory (created if needed).")
    ],
    from_manifest: Annotated[
        Path | None,
        typer.Option(
            "--from-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Manifest of the documents (JSON lines); default: every PDF in --pdf-dir.",
        ),
    ] = None,
    pdf_dir: Annotated[
        Path | None,
        typer.Option(
            "--pdf-dir",
            exists=True,
            file_okay=False,
            help="Folder holding <id>.pdf for every document (default: DIRECTORY/pdfs).",
        ),
    ] = None,
    profile: Annotated[
        str, typer.Option(help="Rooter profile whose engines read the pages.")
    ] = "accurate",
    max_pages: Annotated[
        int, typer.Option("--max-pages", min=1, help="Pages read per document, at most.")
    ] = 30,
    first_pages: Annotated[
        int,
        typer.Option(
            "--first-pages", min=0, help="Leading pages always read; the rest are sampled."
        ),
    ] = 10,
    seed: Annotated[int, typer.Option(help="Seed of the page sample.")] = 0,
) -> None:
    """Read PDFs into a corpus store: elements of the first and sampled pages, without page
    furniture or contents.

    Resumable: documents already in the store are skipped, and each document is saved as soon
    as it is read. Progress goes to stderr, the summary to stdout.
    """
    from docrooter.corpus import Corpus, build_corpus, read_manifest

    checked = _check_profile(profile) or "accurate"
    folder = pdf_dir if pdf_dir is not None else directory / "pdfs"
    with _errors_exit_1():
        if not folder.is_dir():
            raise DocrooterError(f"no PDF folder at {folder}; pass --pdf-dir")
        records = read_manifest(from_manifest) if from_manifest is not None else None
        report = build_corpus(
            Corpus(directory),
            folder,
            records=records,
            profile=checked,
            max_pages=max_pages,
            first_pages=first_pages,
            seed=seed,
            on_progress=lambda done, total, doc_id, status: typer.echo(
                f"[{done}/{total}] {doc_id}: {status}", err=True
            ),
        )
    typer.echo(
        f"{len(report.added)} added, {len(report.skipped)} skipped, {len(report.missing)} "
        f"missing, {len(report.mismatched)} mismatched, {len(report.failed)} failed; "
        f"{report.pages_read} pages, {report.elements} elements, {report.seconds:.1f}s; "
        f"store {directory}"
    )
    for doc_id, error in report.failed.items():
        typer.echo(f"failed {doc_id}: {error}")


@corpus_app.command("stats")
def corpus_stats(
    directory: Annotated[Path, typer.Argument(file_okay=False, help="Corpus store directory.")],
) -> None:
    """Print a corpus store's counts as JSON (and refresh its stats.json)."""
    from docrooter.corpus import Corpus

    with _errors_exit_1():
        if not (directory / "manifest.jsonl").is_file():
            raise DocrooterError(f"no corpus at {directory} (no manifest.jsonl)")
        stats = Corpus(directory).stats()
    typer.echo(json.dumps(stats, indent=2, sort_keys=True))


# --- serve --------------------------------------------------------------------------------


def _is_loopback(host: str) -> bool:
    import ipaddress

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _browser_url(host: str, port: int) -> str:
    """The URL to open for a server bound to ``host``; an any-address bind opens loopback."""
    shown = host.strip("[]")
    if shown in ("", "0.0.0.0", "::"):
        shown = "127.0.0.1"
    if ":" in shown:
        shown = f"[{shown}]"
    return f"http://{shown}:{port}/"


SHUTDOWN_GRACE_SECONDS = 2.0
"""How long ``docrooter serve`` lets open connections finish once asked to stop. A page that
follows a live run holds its event stream open until the run ends, so without this bound
Ctrl+C would wait for every followed run."""


def _playground_server(application: Any, host: str, port: int) -> Any:
    """The uvicorn server for the playground ``application`` (from ``create_app``).

    On shutdown it first ends the event streams that pages follow, so they close cleanly
    instead of being cancelled when :data:`SHUTDOWN_GRACE_SECONDS` runs out; the grace period
    still bounds any other connection, such as an upload in progress.
    """
    uvicorn = require("uvicorn", "ui")
    base: Any = uvicorn.Server

    class PlaygroundServer(base):
        async def shutdown(self, sockets: Any = None) -> None:
            application.state.playground.end_streams()
            await super().shutdown(sockets=sockets)

    config = uvicorn.Config(
        application, host=host, port=port, timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS
    )
    return PlaygroundServer(config)


def _open_when_started(server: Any, url: str, timeout: float = 30.0) -> None:
    """Open ``url`` in the browser once the uvicorn ``server`` reports it has started."""
    import webbrowser

    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() > deadline or server.should_exit:
            return
        time.sleep(0.1)
    webbrowser.open(url)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Address to bind; keep it local.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535, help="Port to listen on.")] = 8765,
    backend: Annotated[
        str,
        typer.Option("--backend", "-b", help="Backend selected by default, e.g. keyword."),
    ] = "keyword",
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the playground in the browser once it is up.")
    ] = False,
    env_file: Annotated[
        Path | None,
        typer.Option(
            "--env-file",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Read missing environment variables (e.g. TYPESAFE_API_KEY) from this file.",
        ),
    ] = None,
) -> None:
    """Start the local playground: upload a document and watch it being processed."""
    import threading

    with _errors_exit_1():
        require("uvicorn", "ui")  # fail early, with the install hint
        from docrooter.ui.server import check_backend_spec, create_app, load_env_file

        check_backend_spec(backend)
        if env_file is not None:
            load_env_file(env_file)
        application = create_app(backend, host=host, port=port)
    err = Console(stderr=True, highlight=False)
    if not _is_loopback(host):
        err.print(
            Text(
                f"warning: binding to {host} exposes the playground (uploads and model runs, "
                "no authentication) to other machines; use 127.0.0.1 to keep it local",
                style="yellow",
            ),
            soft_wrap=True,
        )
    url = _browser_url(host, port)
    server = _playground_server(application, host, port)
    if open_browser:
        threading.Thread(target=_open_when_started, args=(server, url), daemon=True).start()
    err.print(Text(f"docrooter playground: {url}"))
    server.run()
