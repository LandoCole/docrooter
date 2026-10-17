## What this changes

<!-- A short description, and the issue it closes (Closes #123), if any. -->

## How you checked it

<!-- Commands you ran, documents you tried. -->

## Checklist

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the details.

- [ ] `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy` pass
- [ ] `uv run pytest` passes (and `uv run pytest -m models` if I touched a model, backend or training code)
- [ ] New behaviour has tests; a bug fix has a regression test
- [ ] Unit tests still do not import torch, and the core install is still torch-free
- [ ] Public API, CLI flags and defaults are documented, and the README examples still run
- [ ] `CHANGELOG.md` has an entry under *Unreleased* for any user-visible change
- [ ] Benchmark changes: nothing tuned on the held-out set, labels committed before the run, results recorded with provenance
- [ ] No secrets, personal data or large files are added
