"""Packaging metadata and the release workflow: the extras' lower bounds and the publish guard
that refuses a release whose tag is not the package version."""

import shutil
import subprocess
from pathlib import Path

import pytest

import docrooter

ROOT = Path(__file__).parents[2]


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_ui_extra_has_the_lower_bounds_the_code_needs():
    ui = _pyproject()["project"]["optional-dependencies"]["ui"]
    # python_multipart (the import name the server requires) exists from 0.0.13; uvicorn has
    # timeout_graceful_shutdown from 0.22; fastapi 0.100 is the first with pydantic 2.
    assert sorted(ui) == ["fastapi>=0.100", "python-multipart>=0.0.13", "uvicorn>=0.22"]


def test_dev_group_installs_httpx_for_the_playground_tests():
    assert "httpx" in _pyproject()["dependency-groups"]["dev"]


def _tag_check() -> str:
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())
    steps = workflow["jobs"]["build"]["steps"]
    names = [step.get("name", "") for step in steps]
    check = names.index("Check the release tag matches the package version")
    assert check < names.index("Build sdist and wheel")  # fail before building anything
    return steps[check]["run"]


@pytest.mark.parametrize(
    ("tag", "ok"),
    [(f"v{docrooter.__version__}", True), ("v99.0.0", False), (docrooter.__version__, False)],
)
def test_publish_refuses_a_tag_that_is_not_the_package_version(tag, ok):
    script = _tag_check()
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")
    result = subprocess.run(
        [bash, "-eu", "-c", script],
        cwd=ROOT,
        env={"GITHUB_REF_NAME": tag, "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is ok, result.stdout + result.stderr
    if not ok:
        assert f"v{docrooter.__version__}" in result.stdout + result.stderr


def _publish_workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / ".github/workflows/publish.yml").read_text())


def test_pypi_upload_is_manual_and_uses_trusted_publishing():
    workflow = _publish_workflow()
    pypi = workflow["jobs"]["pypi"]
    assert pypi["if"] == "github.event_name == 'workflow_dispatch'"  # never on a tag push
    assert pypi["permissions"] == {"id-token": "write"}  # OIDC: no stored token
    text = (ROOT / ".github/workflows/publish.yml").read_text()
    assert "PYPI_API_TOKEN" not in text and "password:" not in text
    security = " ".join((ROOT / "SECURITY.md").read_text().split())
    assert "trusted publishing" in security and "manual" in security.lower()


def test_release_tests_run_the_same_checks_as_ci():
    runs = [step.get("run", "") for step in _publish_workflow()["jobs"]["test"]["steps"]]
    for check in ("ruff check .", "ruff format --check .", "mypy", "pytest"):
        assert any(check in run for run in runs), check


def test_security_policy_names_the_ocr_engines():
    security = " ".join((ROOT / "SECURITY.md").read_text().split())
    assert "does not run OCR" not in security
    for engine in ("tesseract", "rapidocr", "ocrmac"):
        assert engine in security.lower(), engine
