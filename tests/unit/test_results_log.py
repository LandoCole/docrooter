"""The JSONL results log: one provenance row per LLM decision, never a key, Jev never trainable."""

import json

import numpy as np

from docrooter.backends.base import TokenUsage
from docrooter.results_log import ROW_FIELDS, ResultsLog, text_hash

SECRET = "sk-proj-Q2xhdWRlU2VjcmV0S2V5MTIzNDU2Nzg5MGFiY2Rl"
TYPESAFE_SECRET = "tsk_live_9fQwErTy8UiOp7AsDfGh6JkLz"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def decision(**overrides):
    record = {
        "level": "terms",
        "view_hash": text_hash("the view"),
        "options": ["financial.annual_report", "legal.contract"],
        "label": "legal.contract",
        "probs": np.array([0.25, 0.75]),
        "model": "qwen3-4b",
        "endpoint_host": "http://127.0.0.1:8080/v1",
        "prompt_hash": text_hash("the prompt"),
        "tokens": TokenUsage(120, 1, 1, True),
    }
    record.update(overrides)
    return record


def test_log_marks_jev_untrainable_and_redacts(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", TYPESAFE_SECRET)
    path = tmp_path / "logs" / "results.jsonl"
    log = ResultsLog(path)

    log.write(decision())
    log.write(decision(model="jev-1.13.0", trainable=True))  # a caller cannot make Jev trainable
    log.write(decision(endpoint_host="https://api.typesafe.ai/v1?x=1"))
    log.write(decision(backend="jev:jev-1.13.0"))
    log.write(decision(trainable=False))  # provider terms may forbid training: kept
    log.write(
        decision(
            label=f"Bearer {SECRET}",
            note=f"server said: key {SECRET} and {TYPESAFE_SECRET} rejected",
            api_key="anything",
            headers={"Authorization": "Bearer abc"},
        )
    )

    text = path.read_text(encoding="utf-8")
    assert SECRET not in text and TYPESAFE_SECRET not in text and "abc" not in text
    assert "anything" not in text

    got = rows(path)
    assert [r["trainable"] for r in got] == [True, False, False, False, False, True]
    first = got[0]
    assert set(ROW_FIELDS) <= set(first)
    assert first["endpoint_host"] == "127.0.0.1"
    assert got[2]["endpoint_host"] == "api.typesafe.ai"
    assert first["probs"] == [0.25, 0.75]
    assert first["tokens"] == {"prompt": 120, "completion": 1, "calls": 1, "exact": True}
    assert first["view_hash"] == text_hash("the view") and len(first["view_hash"]) == 64
    assert first["timestamp"].endswith("+00:00")

    # Rows append across instances; missing fields are present as null.
    ResultsLog(path).write({"label": "x"})
    last = rows(path)[-1]
    assert set(ROW_FIELDS) <= set(last) and last["model"] is None and last["trainable"] is True
