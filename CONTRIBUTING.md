# Contributing to docrooter

Thank you for helping. This page covers how to set up, test and submit a change. By taking part you
agree to the [Code of Conduct](CODE_OF_CONDUCT.md). To report a security problem, follow
[SECURITY.md](SECURITY.md) and do not open a public issue.

## Set up

docrooter uses [uv](https://docs.astral.sh/uv/). Python 3.10 to 3.13 are supported.

```bash
git clone https://github.com/LandoCole/docrooter && cd docrooter
uv sync --extra models --extra pdf --extra ui --extra llm
```

Pick the extras you need (see the table in the [README](README.md#install)); add `--extra pdf-mit`
for the pdfplumber engine, `--extra jev` for the TypeSafe backend and `--extra bench` for the
benchmarks.

- **Always repeat the same `--extra` flags.** `uv sync` makes the environment match what you ask
  for, so a bare `uv sync` uninstalls the extras you installed before.
- **Use `PYTHONPATH=src` for the CLI and scripts.** On macOS the editable-install `.pth` file can
  get a hidden flag, and Python then skips it, so `docrooter` and `python -m benchmarks...` fail to
  import the package. Prefix the command, for example
  `PYTHONPATH=src uv run docrooter --help`. `pytest` does not need it: `pyproject.toml` already puts
  `src` on its path.
- Install the git hooks once with `uvx pre-commit install` (`pre-commit` is not a project
  dependency). They run `ruff` (with `--fix`), `ruff format` and two whitespace fixers.

## Run the checks

```bash
uv run ruff check .             # lint
uv run ruff format --check .    # formatting (drop --check to apply it)
uv run mypy                     # types for src/docrooter
uv run pytest                   # unit tests
uv run pytest --cov=docrooter   # with coverage (install the pdf extras to measure docrooter.pdf)
uv run pytest -m models         # model tests
```

- **Unit tests** (`tests/unit/`) are fast, download nothing and never import torch; an autouse
  fixture in `tests/unit/conftest.py` fails any test that does. Tests that need an extra
  (PyMuPDF, pdfplumber, FastAPI) skip when it is missing, so install `pdf`, `pdf-mit` and `ui` to
  run them. PDF tests build tiny PDFs inside the test.
- **Model tests** (`-m models`, `tests/integration/`) download models from Hugging Face (about
  600 MB for the default one) and need `--extra models`. They are deselected by default. The live
  Jev test skips unless `TYPESAFE_API_KEY` is set.
- CI fails the build under 80% line coverage on `docrooter.document`, `taxonomy`, `classify`,
  `views` and `pdf`, measured with the extras installed. A new line in those packages needs a test.
- Run the tests for what you changed while you work, and the whole suite before you push.
- A bug fix comes with a test that fails without it. A new behaviour comes with a test for it.
  Test behaviour, not mocks.
- Keep the README examples runnable: `tests/unit/test_examples.py` runs `examples/` and checks the
  README's snippets and relative links.

## Documentation

The site is built with MkDocs Material from `mkdocs.yml`; the hand-written pages are in
`docs/site/`, and `docs/hooks.py` adds the benchmark reports, this guide, the changelog and the
security policy as pages, and generates the CLI reference with `docs/gen_cli.py`.

```bash
PYTHONPATH=src uv run mkdocs serve             # live preview on http://127.0.0.1:8000
PYTHONPATH=src uv run mkdocs build --strict    # what CI runs: any broken link fails
```

- Every Python line on a docs page must be a line of a script in `examples/`; put new samples
  there first. `tests/unit/test_docs_snippets.py` checks this, runs the keyword-backend examples,
  and checks that every `docrooter` option in a `bash` block exists.
- Link to repository files with relative paths (`../../../examples/toc.py`); the build turns them
  into repository links, so they work both on the site and on the repository host.
- Numbers come from the committed results under `benchmarks/results/`, never from memory.

## Memory-budget etiquette

docrooter works within a memory budget of 8 GiB (see [docs/performance.md](docs/performance.md)),
and so should anything you add.

- **Run one model-loading process at a time.** Two at once can exhaust a 16 GB machine. Model
  benchmarks refuse to start while another one, or `docrooter serve`, is running. Wait for it, and
  do not override the guard with `--allow-concurrent` unless you know the machine has room.
- Do not set `PYTORCH_MPS_*` variables by hand. The benchmark entry points derive them from the
  budget, with 4 threads and `TOKENIZERS_PARALLELISM=false`. In your own scripts, call
  `docrooter.configure(memory_budget_gb=8, threads=4)` before the first model loads.
- Release models you load: use `with Classifier(...) as clf:` or call `close()`.
- Keep the core install torch-free. Torch and everything that needs it is imported lazily, behind
  the `models` extra, and never at import time of `docrooter`.
- Anything that loads a document must work with `max_pages`; anything that loops over a corpus must
  work in chunks, as `docrooter doctype --chunk-size` does.

## Benchmark rules

The benchmarks (`benchmarks/`) are how docrooter's claims are checked, so they follow strict rules.

- **Never tune on the held-out corpus.** `benchmarks/heldout/` (40 PDFs) exists to measure
  generalisation once. Nothing in `src/`, the taxonomies, the prompts, the thresholds or the
  benchmark scripts may be changed by looking at its documents or results. Its numbers are in
  [docs/benchmarks-heldout.md](docs/benchmarks-heldout.md) from a single frozen run. If you need to
  iterate, do it on the development corpus (`benchmarks/corpus/`, 51 PDFs). A held-out set that has
  been used for tuning is no longer held out, and a new one must be collected. The current set is
  spent: its `pdfplumber` result led to the fix `f3baa43` (developed on dev and synthetic PDFs
  only), so collect a new set before the next generalisation claim.
- **Report dev numbers as dev numbers.** Rules and defaults were chosen while looking at the
  development corpus, so its scores are optimistic. Say which corpus a number comes from, and keep
  the dev-versus-held-out comparison honest, including drops.
- **Commit labels before you run.** Gold and silver labels are written and committed before any
  model sees the documents they label.
- **Record provenance.** Results go to `benchmarks/results/<date>/` as JSON with the commit, the
  environment and the machine load. The tables in `docs/benchmarks*.md` between the
  `<!-- ...:begin -->` and `<!-- ...:end -->` markers, and all of `docs/benchmarks.md`, are
  generated from them (`python -m benchmarks.report`, `python -m benchmarks.heldout_eval render`
  and the `--docs` option of the PDF harnesses) and not edited by hand. The analysis around the
  markers, and the `decisions.md` that `benchmarks.report` appends, are hand-written: after every
  re-run, re-check each number in them against the JSON.
- **Do not hide failures.** Rejected experiments, caveats and negative results belong in the write-up.
- The PDFs themselves are not redistributed: the manifests list URLs and hashes, and the corpora are
  fetched into git-ignored folders.

## Commit style

Commits are small and focused, and the subject is a short imperative line in the form
`type(scope): summary`. The types in use are `feat`, `fix`, `docs`, `bench`, `tune`, `style` and
`chore`; the scope is optional (`pdf`, `cli`, `ui`, `memory`, `events`,
`taxonomies`, ...). For example:

```text
fix(pdf): keep numbered openers and slide titles out of section running lines
docs(readme): pdf-mit extra, PDF engine licensing, v0.2 Python API and CLI gating
bench: frozen held-out evaluation results and write-up
```

Explain the why in the body when it is not obvious. Do not commit `.env` files, API keys, model
weights or downloaded corpora.

## Pull requests

Branch from `main`, keep a pull request to one change, and describe what it does and how you checked
it. The checklist is also in the [pull request template](.github/pull_request_template.md):

- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy` pass.
- [ ] `uv run pytest` passes, and `uv run pytest -m models` if you touched a model, backend or
      training code.
- [ ] New behaviour has tests; a bug fix has a regression test.
- [ ] Unit tests still do not import torch, and the core install is still torch-free.
- [ ] Public API, CLI flags and defaults are documented in the README or `docs/`, examples still
      run, and `PYTHONPATH=src uv run mkdocs build --strict` passes.
- [ ] `CHANGELOG.md` has an entry under *Unreleased* for any user-visible change.
- [ ] Benchmark changes follow the rules above: no tuning on the held-out set, labels committed
      before the run, results with provenance.
- [ ] No secrets, personal data or large files are added.

Maintainers may ask for changes; please be patient and kind with each other.

## Releasing

Maintainers only. The version is set in one place, `src/docrooter/__init__.py`.

1. Move the *Unreleased* entries of `CHANGELOG.md` under the new version, set `__version__`, and
   merge to `main`.
2. Tag the merge commit and push the tag, for example `git tag v1.0.0 && git push origin v1.0.0`.
   The [Release workflow](.github/workflows/publish.yml) tests the tag, refuses a tag that is
   not `v<__version__>`, builds the sdist and wheel, and attaches them to the GitHub release of
   the tag. It needs no secrets.
3. Check the release from a fresh virtual environment:
   `pip install "git+https://github.com/LandoCole/docrooter@v1.0.0"`, or the wheel attached to
   the release.

**PyPI is manual.** The workflow uploads to PyPI only when it is run by hand, on the tag, with
trusted publishing, so no PyPI token is stored anywhere:

1. Once: on PyPI, add a trusted publisher for the `docrooter` project (a pending publisher
   before the first upload): owner `LandoCole`, repository `docrooter`, workflow `publish.yml`,
   environment `pypi`. In GitHub, create the `pypi` environment (Settings > Environments); add
   yourself as a required reviewer there to approve every upload.
2. Open Actions > Release > Run workflow, choose the tag (not a branch) as the ref, and run it.
   It tests and builds the tag again, then uploads with a short-lived OIDC token.

To upload from your own machine instead, build with `uv build` and run `uv publish` (or
`twine upload dist/*`) with a PyPI token in your environment, never in a file in the
repository.
