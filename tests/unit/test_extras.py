import pytest

from docrooter._extras import MissingExtraError, require


def test_require_returns_module():
    assert require("json", "models").dumps({}) == "{}"


def test_require_missing_names_extra():
    with pytest.raises(MissingExtraError, match=r"pip install 'docrooter\[models\]'"):
        require("definitely_not_a_module_xyz", "models")


def test_version():
    import docrooter

    assert docrooter.__version__ == "0.1.0.dev0"


def test_missing_extra_error_is_both_a_docrooter_and_an_import_error():
    from docrooter import DocrooterError
    from docrooter import MissingExtraError as Exported

    assert Exported is MissingExtraError
    with pytest.raises(DocrooterError):
        require("definitely_not_a_module_xyz", "models")
    with pytest.raises(ImportError):
        require("definitely_not_a_module_xyz", "models")
