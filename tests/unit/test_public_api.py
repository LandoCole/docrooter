import subprocess
import sys

import docrooter

PUBLIC_NAMES = [
    "load",
    "Document",
    "Section",
    "Taxonomy",
    "TaxonomyNode",
    "SectionView",
    "Classifier",
    "Greedy",
    "Beam",
    "Flat",
    "Dataset",
    "train",
    "TrainedModel",
    "evaluate",
    "compare",
    "calibrate",
]


def test_public_api_surface():
    for name in PUBLIC_NAMES:
        assert hasattr(docrooter, name), name
        assert name in docrooter.__all__, name


def test_all_lists_only_existing_names():
    for name in docrooter.__all__:
        assert hasattr(docrooter, name), name


def test_core_import_does_not_import_torch():
    out = subprocess.run(
        [sys.executable, "-c", "import docrooter, sys; print('torch' in sys.modules)"],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "False", out.stderr
