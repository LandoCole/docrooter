"""``benchmarks.memory_check`` helpers (the cases themselves load models: not unit-tested)."""

import json

from benchmarks import memory_check


def test_peak_rss_is_bytes():
    assert memory_check.peak_rss_bytes() > 10 * 1024**2  # a Python process is > 10 MB


def test_save_merges_by_case_and_device(tmp_path):
    memory_check.save({"case": "train", "device": "cpu", "peak_rss_gb": 5.0}, tmp_path)
    memory_check.save({"case": "classify", "device": "cpu", "peak_rss_gb": 2.0}, tmp_path)
    memory_check.save({"case": "train", "device": "cpu", "peak_rss_gb": 4.0}, tmp_path)
    rows = json.loads((tmp_path / "memory.json").read_text())
    assert [(r["case"], r["peak_rss_gb"]) for r in rows] == [("classify", 2.0), ("train", 4.0)]


def test_mps_sampler_without_torch_records_nothing():
    with memory_check.MpsSampler(interval=0.01) as mps:
        pass
    assert mps.peak_driver == 0 and mps.peak_allocated == 0


def test_footprint_counts_at_least_the_resident_pages():
    import sys

    footprint = memory_check.footprint_bytes()
    if sys.platform != "darwin":
        assert footprint is None
        return
    current, peak = footprint
    assert 10 * 1024**2 < current <= peak
