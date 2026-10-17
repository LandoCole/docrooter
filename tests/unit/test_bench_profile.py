"""``benchmarks.profile_pipeline`` helpers (the profiling run itself loads a model)."""

from types import SimpleNamespace

from benchmarks import profile_pipeline as pp

SITE = "/venv/lib/python3.12/site-packages"
SRC = "/repo/src/docrooter"


def frame(function: str, file: str, line: int = 1) -> str:
    return f"{function}\x00{file}\x00{line}"


def stack(*frames: str) -> list[str]:
    return [frame("<module>", "/repo/run.py"), *frames]


def test_library_buckets_win_over_the_docrooter_stage():
    extract = frame("extract_lines", f"{SRC}/pdf/lines.py")
    assert pp.bucket_of(stack(extract, frame("get_text", f"{SITE}/pymupdf/__init__.py"))) == (
        "pymupdf"
    )
    render = frame("render", f"{SRC}/doc_view.py")
    tokenize = frame("encode", f"{SITE}/transformers/tokenization_utils_base.py")
    assert pp.bucket_of(stack(render, tokenize)) == "tokenisation"
    encode = frame("encode", f"{SITE}/sentence_transformers/SentenceTransformer.py")
    st_tokenize = frame("tokenize", f"{SITE}/sentence_transformers/models/Transformer.py")
    assert pp.bucket_of(stack(encode, st_tokenize, tokenize)) == "tokenisation"
    forward = frame("forward", f"{SITE}/torch/nn/modules/module.py")
    assert pp.bucket_of(stack(encode, forward)) == "inference"
    assert pp.bucket_of(stack(encode)) == "inference"


def test_innermost_docrooter_stage_names_our_own_code():
    load = frame("_load_pdf", f"{SRC}/pdf/load.py")
    assert pp.bucket_of(stack(load, frame("extract_lines", f"{SRC}/pdf/lines.py"))) == (
        "line_assembly"
    )
    assert pp.bucket_of(stack(load, frame("find_headings", f"{SRC}/pdf/headings.py"))) == (
        "headings"
    )
    assert pp.bucket_of(stack(load, frame("build_document", f"{SRC}/document/build.py"))) == (
        "tree_assembly"
    )
    normalize = frame("_normalize", f"{SRC}/classify/normalize.py")
    assert pp.bucket_of(stack(normalize, frame("render_all", f"{SRC}/doc_view.py"))) == "view"
    assert pp.bucket_of(stack(normalize)) == "other"
    # A function of the same name outside docrooter does not count.
    assert pp.bucket_of(stack(frame("render", f"{SITE}/jinja2/environment.py"))) == "other"


def test_split_and_own_python_share():
    records = [
        (stack(frame("get_text", f"{SITE}/pymupdf/__init__.py")), 0.5),
        (stack(frame("find_headings", f"{SRC}/pdf/headings.py")), 0.2),
        (stack(frame("forward", f"{SITE}/torch/nn/modules/module.py")), 0.3),
    ]
    split = pp.split_samples(records)
    assert split["pymupdf"] == 0.5 and split["headings"] == 0.2 and split["inference"] == 0.3
    assert set(split) == set(pp.BUCKETS)
    assert abs(pp.own_python(split) - 0.2) < 1e-9


def test_stage_seconds_pairs_start_and_end_events():
    def event(stage: str, status: str, t: float) -> SimpleNamespace:
        return SimpleNamespace(stage=stage, status=status, t=t)

    recorded = [
        event("extract", "start", 0.0),
        event("extract", "progress", 0.5),
        event("extract", "end", 1.0),
        event("view", "start", 1.0),
        event("view", "end", 1.25),
        event("view", "start", 2.0),  # a stage that runs twice adds up
        event("view", "end", 2.5),
        event("done", "end", 3.0),
    ]
    stages = pp.stage_seconds(recorded)
    assert stages["extract"] == 1.0 and stages["view"] == 0.75 and stages["normalize"] == 0.0
    assert list(stages) == list(pp.STAGES)


def test_upper_median_takes_the_upper_of_two_middle_values():
    assert pp.upper_median([0.9, 0.1, 0.5]) == 0.5
    assert pp.upper_median([0.4, 0.1, 0.3, 0.2]) == 0.3  # not the mean 0.25, not the lower 0.2


def test_aggregate_reports_the_upper_median_and_render_says_so():
    rows = [
        {
            "id": f"d{i}",
            "pages": 1,
            "seconds": 1.0,
            "stages": dict.fromkeys(pp.STAGES, 0.1),
            "profiled_seconds": 1.0,
            "split": {b: (1.0 if b == "other" else 0.0) for b in pp.BUCKETS},
            "own_python_share": share,
        }
        for i, share in enumerate((0.1, 0.2, 0.3, 0.4))
    ]
    agg = pp.aggregate(rows)
    assert agg["own_python_share_median_doc"] == 0.3
    result = {"docs": rows, "aggregate": agg}
    assert "30% upper median over documents" in pp.render(result)


def test_imported_docrooter_records_path_and_revision():
    from pathlib import Path

    import docrooter

    root = Path(__file__).resolve().parents[2]
    info = pp.imported_docrooter(root)
    if Path(docrooter.__file__).resolve().is_relative_to(root):
        assert info["docrooter_path"] == "src/docrooter"
    assert not str(info["docrooter_path"]).startswith(str(Path.home()))
    rev = info["docrooter_revision"]
    assert rev is None or rev.removesuffix("+dirty").isalnum()
