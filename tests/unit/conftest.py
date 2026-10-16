import sys

import pytest

# Unit tests must never pull in torch. Snapshot at import time so a torch that something
# else imported before the unit tests ran does not count against them.
_TORCH_AT_START = "torch" in sys.modules
_reported = False


@pytest.fixture(autouse=True)
def _no_torch_import():
    yield
    global _reported
    if not _TORCH_AT_START and not _reported and "torch" in sys.modules:
        _reported = True  # name the first offender only; later tests would just inherit it
        pytest.fail("this unit test imported torch (unit tests must not)", pytrace=False)
