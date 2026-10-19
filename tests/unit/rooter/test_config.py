"""``RouterConfig`` and the profiles (spec §7.2, §7.3): precedence, files, unknown keys."""

import json

import pytest
from pydantic import ValidationError

from docrooter.backends import jev
from docrooter.classify.classifier import DEFAULT_BACKEND
from docrooter.errors import DocrooterError
from docrooter.rooter import PROFILES, RouterConfig, profile_defaults
from docrooter.rooter import profiles as profiles_module

TASKS = ("text", "ocr", "layout", "doc_type", "sections", "summaries", "embeddings", "escalate")


def test_profiles_match_spec():
    common = {"text": "auto", "doc_type": DEFAULT_BACKEND, "sections": DEFAULT_BACKEND,
              "embeddings": "none", "escalate_below": 0.5}  # fmt: skip
    assert PROFILES == {
        "fast": {**common, "ocr": "none", "layout": "none", "summaries": "none",
                 "escalate": "none"},
        "balanced": {**common, "ocr": "auto", "layout": "heuristic", "summaries": "centroid",
                     "escalate": "none"},
        "accurate": {**common, "ocr": "auto", "layout": "heron", "summaries": "centroid",
                     "escalate": "jev:jev-1.13.0"},
    }  # fmt: skip
    assert DEFAULT_BACKEND == "embedding:dmis-lab/UltraBERT-Embed-150M"
    # the copies in profiles.py (kept there so `import docrooter` skips the jev module)
    assert profiles_module.API_KEY_ENV == jev.API_KEY_ENV
    assert profiles_module.JEV_ESCALATION == f"jev:{jev.DEFAULT_MODEL}"


def test_accurate_escalates_only_with_a_jev_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert profile_defaults("accurate")["escalate"] == "none"
    assert RouterConfig(profile="accurate").escalate == "none"
    monkeypatch.setenv("TYPESAFE_API_KEY", "   ")
    assert RouterConfig(profile="accurate").escalate == "none"
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    assert RouterConfig(profile="accurate").escalate == "jev:jev-1.13.0"
    assert RouterConfig(profile="balanced").escalate == "none"
    assert PROFILES["accurate"]["escalate"] == "jev:jev-1.13.0"  # the table is never mutated


@pytest.mark.parametrize("name", ["fast", "balanced", "accurate"])
def test_profile_fills_every_task(name, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    config = RouterConfig(profile=name)
    assert {task: getattr(config, task) for task in TASKS} == {
        task: PROFILES[name][task] for task in TASKS
    }
    assert config.escalate_below == 0.5
    assert (config.max_pages, config.ocr_lang, config.ocr_dpi, config.tables) == (
        None, "en", 200, "lines",
    )  # fmt: skip


def test_default_profile_is_balanced():
    assert RouterConfig() == RouterConfig(profile="balanced") == RouterConfig.resolve()


def test_precedence_overrides_file_profile(tmp_path):
    path = tmp_path / "router.toml"
    path.write_text('summaries = "lead"\nocr = "rapidocr"\nmax_pages = 40\n', encoding="utf-8")
    config = RouterConfig.resolve(profile="fast", file=path, ocr="tesseract")
    assert config.profile == "fast"
    assert config.ocr == "tesseract"  # override beats the file
    assert config.summaries == "lead" and config.max_pages == 40  # file beats the profile
    assert config.layout == "none"  # the profile fills the rest

    # the profile argument picks the defaults over the file's own profile
    path.write_text('profile = "accurate"\nsummaries = "lead"\n', encoding="utf-8")
    assert RouterConfig.resolve(file=path).layout == "heron"
    picked = RouterConfig.resolve(profile="fast", file=path)
    assert (picked.profile, picked.layout, picked.summaries) == ("fast", "none", "lead")
    # Python construction: explicit values over the profile
    assert RouterConfig(profile="accurate", layout="heuristic").layout == "heuristic"


def test_unknown_key_is_error(tmp_path):
    with pytest.raises(ValidationError, match="bogus"):
        RouterConfig(bogus=1)
    with pytest.raises(ValidationError, match="bogus"):
        RouterConfig.resolve(profile="fast", bogus=1)
    path = tmp_path / "router.json"
    path.write_text(json.dumps({"ocr": "none", "ocr_engine": "rapidocr"}), encoding="utf-8")
    with pytest.raises(ValidationError, match="ocr_engine"):
        RouterConfig.from_file(path)
    with pytest.raises(ValidationError, match="unknown profile"):
        RouterConfig(profile="turbo")


@pytest.mark.parametrize(
    "bad",
    [
        {"escalate_below": 1.5},
        {"max_pages": 0},
        {"ocr_dpi": 0},
        {"ocr_dpi": 71},  # final review R-I2: 72..600
        {"ocr_dpi": 601},
        {"tables": "text"},
        {"summaries": "abstractive"},
        {"text": "pymupdf4llm"},
        {"ocr": ""},
    ],
)
def test_invalid_values_are_errors(bad):
    with pytest.raises(ValidationError):
        RouterConfig(**bad)


def test_ocr_dpi_bounds_are_inclusive():
    assert RouterConfig(ocr_dpi=72).ocr_dpi == 72 and RouterConfig(ocr_dpi=600).ocr_dpi == 600


def test_toml_and_json_files(tmp_path):
    toml = tmp_path / "router.toml"
    toml.write_text(
        'profile = "fast"\ndoc_type = "keyword"\nsections = "none"\nescalate_below = 0.4\n'
        'tables = "none"\n',
        encoding="utf-8",
    )
    as_json = tmp_path / "router.JSON"
    as_json.write_text(
        json.dumps({"profile": "fast", "doc_type": "keyword", "sections": "none",
                    "escalate_below": 0.4, "tables": "none", "max_pages": None}),
        encoding="utf-8",
    )  # fmt: skip
    from_toml, from_json = RouterConfig.from_file(toml), RouterConfig.from_file(as_json)
    assert from_toml == from_json
    assert (from_toml.doc_type, from_toml.sections, from_toml.escalate_below) == (
        "keyword", "none", 0.4,
    )  # fmt: skip
    assert from_toml.ocr == "none"  # from the fast profile
    # the resolved configuration round-trips through JSON
    dumped = tmp_path / "dumped.json"
    dumped.write_text(from_toml.model_dump_json(), encoding="utf-8")
    assert RouterConfig.from_file(dumped) == from_toml


def test_bad_files(tmp_path):
    with pytest.raises(ValueError, match=".toml or .json"):
        RouterConfig.from_file(tmp_path / "router.yaml")
    with pytest.raises(FileNotFoundError):
        RouterConfig.from_file(tmp_path / "missing.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("ocr = ", encoding="utf-8")
    with pytest.raises(DocrooterError, match="cannot read router config"):
        RouterConfig.from_file(broken)
    listed = tmp_path / "list.json"
    listed.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(DocrooterError, match="object of settings"):
        RouterConfig.from_file(listed)


def test_config_is_frozen():
    config = RouterConfig()
    with pytest.raises(ValidationError):
        config.ocr = "none"  # type: ignore[misc]
