import sys
import types

from typer.testing import CliRunner

from docrooter.cli import app

runner = CliRunner()
ULTRA = "dmis-lab/UltraBERT-Embed-150M"
ULTRA_REV = "367d5343b5a0164d57b810cedf5de4b8983d5984"


def _install_hub(monkeypatch, **funcs):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(**funcs))


def test_models_pull_uses_pinned_recipe_revision(monkeypatch):
    calls = []

    def snapshot_download(repo_id, revision=None):
        calls.append((repo_id, revision))
        return "/cache/snap"

    _install_hub(monkeypatch, snapshot_download=snapshot_download)
    result = runner.invoke(app, ["models", "pull", ULTRA])
    assert result.exit_code == 0, result.output
    assert calls == [(ULTRA, ULTRA_REV)]
    assert "/cache/snap" in result.output


def test_models_pull_explicit_revision_and_unpinned(monkeypatch):
    calls = []
    _install_hub(
        monkeypatch,
        snapshot_download=lambda repo_id, revision=None: calls.append((repo_id, revision)) or "/p",
    )
    assert runner.invoke(app, ["models", "pull", ULTRA, "--revision", "abc"]).exit_code == 0
    assert runner.invoke(app, ["models", "pull", "some/model"]).exit_code == 0
    assert calls == [(ULTRA, "abc"), ("some/model", None)]


def test_models_pull_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    result = runner.invoke(app, ["models", "pull", ULTRA])
    assert result.exit_code == 1
    assert "pip install 'docrooter[models]'" in result.output


def test_models_list(monkeypatch):
    rev = types.SimpleNamespace(commit_hash="367d5343b5a0164d57b810cedf5de4b8983d5984")
    repo = types.SimpleNamespace(
        repo_id=ULTRA, repo_type="model", revisions=[rev], size_on_disk=300_000_000
    )
    dataset = types.SimpleNamespace(
        repo_id="some/dataset", repo_type="dataset", revisions=[], size_on_disk=1
    )
    _install_hub(monkeypatch, scan_cache_dir=lambda: types.SimpleNamespace(repos=[repo, dataset]))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0, result.output
    assert ULTRA in result.output and "367d5343" in result.output and "300" in result.output
    assert "some/dataset" not in result.output


def test_models_list_empty(monkeypatch):
    _install_hub(monkeypatch, scan_cache_dir=lambda: types.SimpleNamespace(repos=[]))
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "no cached models" in result.output.lower()
