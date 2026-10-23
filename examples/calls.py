"""Batched or one at a time: one document asked by an LLM both ways, and by Jev.

The guide is docs/site/guides/calls.md. It needs keys for both models, kept in a git-ignored
.env: DOCROOTER_LLM_API_KEY and DOCROOTER_LLM_BASE_URL (https://api.anthropic.com/v1 for
Claude; with Claude also DOCROOTER_LLM_EXTRA_BODY='{"thinking": {"type": "disabled"}}') and
TYPESAFE_API_KEY.

    uv run --env-file .env python examples/calls.py report.pdf
"""

from __future__ import annotations

import sys

from docrooter import classify


def main(path: str) -> None:
    batched = classify(path, backend="openai:claude-haiku-5-5")
    one_at_a_time = classify(path, backend="openai:claude-haiku-5-5", calls="per_item")
    decision_model = classify(path, backend="jev")
    for name, result in [
        ("batched", batched),
        ("one at a time", one_at_a_time),
        ("jev", decision_model),
    ]:
        print(name, result.calls, result.tokens.calls, result.tokens.prompt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report.pdf")
