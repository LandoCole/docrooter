"""The JSONL results log: one provenance row per LLM decision, never a key, Jev never trainable."""

import json

import numpy as np
import pytest

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
    assert first["tokens"] == {
        "prompt": 120,
        "completion": 1,
        "calls": 1,
        "exact": True,
        "estimated_calls": 0,
    }
    assert first["view_hash"] == text_hash("the view") and len(first["view_hash"]) == 64
    assert first["timestamp"].endswith("+00:00")

    # Rows append across instances; missing fields are present as null.
    ResultsLog(path).write({"label": "x"})
    last = rows(path)[-1]
    assert set(ROW_FIELDS) <= set(last) and last["model"] is None and last["trainable"] is True


# --- fix round 1: every leak path from the review ---------------------------------------------


def test_log_strips_urls_to_hosts(tmp_path):
    path = tmp_path / "r.jsonl"
    log = ResultsLog(path)
    log.write(decision(endpoint_host="api.example.com/v1?api_key=abc123"))
    log.write(decision(endpoint="https://u:pw@x.test/v1?key=short1#frag"))
    log.write(
        decision(base_url="http://proxy.test:8080/v1?sig=zz9", note="see https://u:p@h.test/a?k=q1")
    )
    text = path.read_text(encoding="utf-8")
    for leaked in ("abc123", "short1", "pw@", "u:p", "zz9", "q1", "frag", "/v1", "https"):
        assert leaked not in text, leaked
    got = rows(path)
    assert got[0]["endpoint_host"] == "api.example.com"
    assert got[1]["endpoint"] == "x.test"
    assert got[2]["base_url"] == "proxy.test" and got[2]["note"] == "see h.test"


def test_log_redacts_secret_named_fields_but_keeps_token_counts(tmp_path):
    path = tmp_path / "r.jsonl"
    ResultsLog(path).write(
        decision(
            access_token="tok123",
            x_api_key="k1",
            client_secret="s1",
            db_password="p1",
            Authorization="a1",
            extra={"refresh_token": "r1", "keys": ["kk1"]},
            prompt_tokens=12,
            completion_tokens=1,
            total_tokens=13,
        )
    )
    text = path.read_text(encoding="utf-8")
    for leaked in ("tok123", "k1", "s1", "p1", "a1", "r1", "kk1"):
        assert f'"{leaked}"' not in text, leaked
    row = rows(path)[0]
    assert (
        row["prompt_tokens"] == 12 and row["completion_tokens"] == 1 and row["total_tokens"] == 13
    )
    assert row["tokens"]["prompt"] == 120  # the TokenUsage field is a count, not a secret
    assert row["access_token"] == "***"


def test_log_scrubs_the_backends_own_key(tmp_path, monkeypatch):
    from docrooter.backends.openai import OpenAIBackend
    from docrooter.results_log import secrets_of

    for name in ("DOCROOTER_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    b = OpenAIBackend("m", base_url="http://llm.test/v1", api_key="plainkey9")  # no key shape
    assert secrets_of(b) == ("plainkey9",) and secrets_of(object()) == ()
    path = tmp_path / "r.jsonl"
    ResultsLog(path, secrets=secrets_of(b)).write(decision(note="rejected plainkey9 here"))
    assert "plainkey9" not in path.read_text(encoding="utf-8")


def test_log_trainable_must_be_bool_and_nan_is_null(tmp_path):
    path = tmp_path / "r.jsonl"
    log = ResultsLog(path)
    for bad in ("false", 0, "no"):
        with pytest.raises(TypeError, match="trainable"):
            log.write(decision(trainable=bad))
    log.write(decision(probs=np.array([np.nan, np.inf, 0.5])))
    line = path.read_text(encoding="utf-8").strip()
    assert "NaN" not in line and "Infinity" not in line
    assert json.loads(line)["probs"] == [None, None, 0.5]


def test_log_appends_whole_lines_from_many_writers(tmp_path):
    import threading

    path = tmp_path / "r.jsonl"
    big = ["option " + "x" * 50] * 400  # a row larger than an 8 KB write buffer
    logs = [ResultsLog(path), ResultsLog(path)]

    def work(log):
        for _ in range(10):
            log.write(decision(options=big))

    threads = [threading.Thread(target=work, args=(log,)) for log in logs * 2]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    got = rows(path)  # every line parses
    assert len(got) == 40 and all(len(r["options"]) == 400 for r in got)


def test_log_keeps_label_keys_that_look_like_secret_names(tmp_path):
    """Labels in probs are data: "legal.authorization" is not an Authorization header."""
    path = tmp_path / "r.jsonl"
    probs = {"legal.authorization": 0.6, "financial.key_figures": 0.3, "x.tokens": 0.1}
    ResultsLog(path).write(decision(probs=probs, options=list(probs)))
    row = rows(path)[0]
    assert row["probs"] == probs and row["options"] == list(probs)


def test_log_redacts_secret_named_fields_inside_label_fields(tmp_path):
    """Label fields keep their own keys (labels), but a mapping nested inside them is data with
    field names, and a secret named there is hidden."""
    path = tmp_path / "r.jsonl"
    ResultsLog(path).write(
        decision(
            options=[{"id": "a", "api_key": "nested-plain-1"}],
            label={"s1": {"auth_token": "nested-plain-2"}},
            probs={"legal.authorization": 0.5, "x": 0.5},
        )
    )
    text = path.read_text(encoding="utf-8")
    assert "nested-plain-1" not in text and "nested-plain-2" not in text
    row = rows(path)[0]
    assert row["options"][0]["id"] == "a" and row["probs"]["legal.authorization"] == 0.5


def test_log_redacts_key_value_secrets_in_free_text(tmp_path):
    path = tmp_path / "r.jsonl"
    note = 'retry with api_key=plainvalue1 and "token": "plainvalue2"; password: hunter22'
    ResultsLog(path).write(decision(note=note, other="prompt_tokens=12 kept"))
    text = path.read_text(encoding="utf-8")
    for leaked in ("plainvalue1", "plainvalue2", "hunter22"):
        assert leaked not in text, leaked
    row = rows(path)[0]
    assert "api_key=***" in row["note"] and row["other"] == "prompt_tokens=12 kept"


def test_log_ignores_literal_keys_too_short_to_scrub(tmp_path, monkeypatch):
    """Scrubbing a one-letter key would replace that letter everywhere: below 8 characters a
    literal key is not scrubbed (key-shaped values still are)."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    path = tmp_path / "r.jsonl"
    ResultsLog(path, secrets=["e", "abc"]).write(decision(note="keep the keyword text"))
    row = rows(path)[0]
    assert row["note"] == "keep the keyword text" and row["model"] == "qwen3-4b"
