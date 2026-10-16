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
