import json
from pathlib import Path

from typer.testing import CliRunner

from docrooter.cli import app
from docrooter.errors import DatasetError
from docrooter.train import trainer

runner = CliRunner()
FIXTURES = Path(__file__).parents[1] / "fixtures"
TAX = str(FIXTURES / "taxonomies" / "nested.json")
DATA = str(FIXTURES / "datasets" / "small.jsonl")


class _StubModel:
    def __init__(self):
        self.manifest = {
            "heads": {
                "__root__": {"file": "heads/a.joblib", "classes": ["financial", "legal"], "n": 8},
                "[b]legal[/b]": {"file": "heads/b.joblib", "classes": ["x", "y"], "n": 4},
            },
            "train_metrics": {"triplets": 13, "epochs": 1, "lr": 2e-5, "seconds": 2.25},
        }

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "manifest.json").write_text(json.dumps(self.manifest))


def test_train_command_forwards_options_and_prints_summary(monkeypatch, tmp_path):
    calls = []

    def fake_train(taxonomy, dataset, **kwargs):
        calls.append((len(taxonomy), len(dataset), kwargs))
        return _StubModel()

    monkeypatch.setattr(trainer, "train", fake_train)
    out = tmp_path / "bundle"
    args = ["train", "-t", TAX, "-d", DATA, "-o", str(out), "--base", "BAAI/bge-small-en-v1.5"]
    args += ["--no-finetune-body", "--epochs", "2", "--lr", "3e-5", "--min-examples", "2"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert calls == [
        (
            6,
            8,
            {
                "base_model": "BAAI/bge-small-en-v1.5",
                "finetune_body": False,
                "epochs": 2,
                "lr": 3e-5,
                "min_examples_per_child": 2,
            },
        )
    ]
    assert (out / "manifest.json").is_file()
    assert "__root__" in result.output and "financial, legal" in result.output
    assert "[b]legal[/b]" in result.output  # user strings are not read as rich markup
    assert "2 head(s), 13 triplet(s), 2.2s" in result.output


def test_train_command_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(trainer, "train", lambda tax, data, **kw: calls.append(kw) or _StubModel())
    result = runner.invoke(app, ["train", "-t", TAX, "-d", DATA, "-o", str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert calls == [
        {
            "base_model": "dmis-lab/UltraBERT-Embed-150M",
            "finetune_body": True,
            "epochs": None,
            "lr": None,
            "min_examples_per_child": 4,
        }
    ]


def test_train_command_reports_errors(monkeypatch, tmp_path):
    def boom(*args, **kwargs):
        raise DatasetError("line 3: bad path")

    monkeypatch.setattr(trainer, "train", boom)
    result = runner.invoke(app, ["train", "-t", TAX, "-d", DATA, "-o", str(tmp_path / "b")])
    assert result.exit_code == 1
    assert "error: line 3: bad path" in result.output
