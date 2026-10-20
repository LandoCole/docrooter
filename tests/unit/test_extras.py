import pytest

from docrooter._extras import MissingExtraError, require


def test_require_returns_module():
    assert require("json", "models").dumps({}) == "{}"


def test_require_missing_names_extra():
    with pytest.raises(MissingExtraError) as exc:
        require("definitely_not_a_module_xyz", "models")
    # docrooter installs from GitHub: the hint is the GitHub form of the extra
    expected = 'pip install "docrooter[models] @ git+https://github.com/LandoCole/docrooter"'
    assert expected in str(exc.value)


def test_install_hint_names_the_github_source():
    from docrooter._extras import install_hint

    assert install_hint("ocr") == (
        'pip install "docrooter[ocr] @ git+https://github.com/LandoCole/docrooter"'
    )


def test_version():
    import docrooter

    assert docrooter.__version__ == "1.0.0"


def test_missing_extra_error_is_both_a_docrooter_and_an_import_error():
    from docrooter import DocrooterError
    from docrooter import MissingExtraError as Exported

    assert Exported is MissingExtraError
    with pytest.raises(DocrooterError):
        require("definitely_not_a_module_xyz", "models")
    with pytest.raises(ImportError):
        require("definitely_not_a_module_xyz", "models")


def test_cli_version_prints_the_package_version():
    from typer.testing import CliRunner

    import docrooter
    from docrooter.cli import app

    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == f"docrooter {docrooter.__version__}"
