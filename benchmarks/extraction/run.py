"""Demo extraction harness (spec §8.4; task S7): optional, outside the wheel, evaluation only.

docrooter selects; the application extracts. This harness plays the application: it asks an LLM
each question of the fetched evidence-QA dataset (:mod:`benchmarks.evidence_qa`) twice, once
with the context ``select()`` picks for the question at c = 0.9 (the question as a query need)
and once with the whole document, and scores each answer against the dataset's answer.

**It runs only when configured**, by either:

- ``DOCROOTER_EXTRACT_BASE_URL``, ``DOCROOTER_EXTRACT_MODEL`` and ``DOCROOTER_EXTRACT_API_KEY``
  (an OpenAI-compatible ``/chat/completions`` endpoint), or
- ``--mlx-model <id>`` (a local MLX model, through ``mlx_lm``).

Otherwise it prints "not configured" and exits 0. **Jev is never used**: it is a classifier, not
an extractor. A base URL containing "typesafe" or a model named like Jev is refused, and
``TYPESAFE_API_KEY`` is never read.

**Scoring** (:func:`score`): exact match after normalisation (case, punctuation, whitespace), or
numeric tolerance: the first number of the gold answer is matched when any number in the answer
is within 1% of it (relative). Scale words ("million") are not reconciled. A question whose
whole document exceeds ``--max-context-tokens`` is skipped and counted, never truncated.

**Results** go to ``benchmarks/results/<date>/selection-extraction.json``: question ids,
correctness and token counts only; no question, answer or model output is written::

    DOCROOTER_EXTRACT_BASE_URL=... DOCROOTER_EXTRACT_MODEL=... DOCROOTER_EXTRACT_API_KEY=... \\
    PYTHONPATH=src .venv/bin/python -m benchmarks.extraction.run --tagger <bundle> \\
        --selector <selector.json>
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import string
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmarks import _guard

ENV_BASE_URL = "DOCROOTER_EXTRACT_BASE_URL"
ENV_MODEL = "DOCROOTER_EXTRACT_MODEL"
ENV_API_KEY = "DOCROOTER_EXTRACT_API_KEY"
COMPLETENESS = 0.9
TOLERANCE = 0.01  # relative
MAX_ANSWER_TOKENS = 256
DEFAULT_MAX_CONTEXT_TOKENS = 100_000
RESULTS_FILE = "selection-extraction.json"
SYSTEM_PROMPT = (
    "Answer the question using only the document excerpts given. Reply with the answer alone: "
    "a number with its unit, or a short phrase."
)
_NUMBER = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


# --- configuration --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    model: str
    api_key: str = field(repr=False)


def refuse_jev(*, base_url: str = "", model: str = "") -> None:
    """Exit when the endpoint or model is TypeSafe's or Jev: never an extractor."""
    if "typesafe" in base_url.lower():
        raise SystemExit(f"refusing {base_url}: a typesafe endpoint is never used for extraction")
    if "jev" in model.lower():
        raise SystemExit(f"refusing model {model!r}: Jev is a classifier, never an extractor")


def endpoint_from_env() -> Endpoint | None:
    """The OpenAI-compatible endpoint when all three variables are set, else ``None``."""
    base_url = os.environ.get(ENV_BASE_URL, "").strip()
    model = os.environ.get(ENV_MODEL, "").strip()
    api_key = os.environ.get(ENV_API_KEY, "").strip()
    if not (base_url and model and api_key):
        return None
    refuse_jev(base_url=base_url, model=model)
    return Endpoint(base_url=base_url.rstrip("/"), model=model, api_key=api_key)


# --- scoring --------------------------------------------------------------------------------------


def normalise_answer(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCTUATION).split())


def numbers(text: str) -> list[float]:
    """The numbers in ``text``: thousands separators and ``$`` dropped, ``(x)`` negative."""
    out = []
    for token in _NUMBER.findall(text):
        negative = token.startswith("(") and token.endswith(")")
        cleaned = token.strip("()").replace("$", "").replace(",", "")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        out.append(-abs(value) if negative else value)
    return out


def score(prediction: str, gold: str, tolerance: float = TOLERANCE) -> bool:
    """Exact match after normalisation, or the gold answer's first number within ``tolerance``
    (relative) of any number in the prediction."""
    if normalise_answer(prediction) and normalise_answer(prediction) == normalise_answer(gold):
        return True
    wanted = numbers(gold)
    if not wanted:
        return False
    target = wanted[0]
    for value in numbers(prediction):
        if target == 0.0:
            if value == 0.0:
                return True
        elif abs(value - target) <= tolerance * abs(target):
            return True
    return False


# --- models ---------------------------------------------------------------------------------------


def _messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document excerpts:\n\n{context}\n\nQuestion: {question}"},
    ]


class OpenAICompatible:
    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint

    def ask(self, question: str, context: str) -> str:
        body = json.dumps(
            {
                "model": self.endpoint.model,
                "messages": _messages(question, context),
                "temperature": 0,
                "max_tokens": MAX_ANSWER_TOKENS,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.endpoint.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.endpoint.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            data = json.loads(response.read())
        return str(data["choices"][0]["message"]["content"] or "")

    def close(self) -> None:
        pass


class MlxLocal:
    def __init__(self, model_id: str) -> None:
        refuse_jev(model=model_id)
        import mlx_lm

        loaded: Any = mlx_lm.load(model_id)
        self.model, self.tokenizer = loaded[0], loaded[1]

    def ask(self, question: str, context: str) -> str:
        from mlx_lm import generate

        prompt = self.tokenizer.apply_chat_template(
            _messages(question, context), add_generation_prompt=True, tokenize=False
        )
        return str(
            generate(self.model, self.tokenizer, prompt=prompt, max_tokens=MAX_ANSWER_TOKENS)
        )

    def close(self) -> None:
        self.model = self.tokenizer = None


# --- run ------------------------------------------------------------------------------------------


def _context(doc: Any, element_ids: Sequence[str]) -> str:
    wanted = set(element_ids)
    return "\n\n".join(el.text for el in doc.elements if el.id in wanted and el.text.strip())


def _accuracy(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return sum(r[key]["correct"] for r in rows) / len(rows) if rows else None


def _mean_tokens(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    return sum(r[key]["tokens"] for r in rows) / len(rows) if rows else None


def run(args: argparse.Namespace, model: Any) -> int:
    from benchmarks import evidence_qa as qa
    from benchmarks.selection_bench import _commit, _out_dir, _write_json, load_selector
    from docrooter.backends.base import close_backend
    from docrooter.backends.registry import resolve_backend
    from docrooter.chunking import document_of
    from docrooter.errors import DocrooterError
    from docrooter.rooter.profiles import profile_defaults
    from docrooter.select.selection import need_features, prepare, select, select_scored
    from docrooter.tagger.element_tags import resolve_tagger

    started = time.perf_counter()
    dataset, _ = qa.load_cached(qa.CACHE)
    selector = load_selector(args.selector, args.tagger)
    embedder = args.embedder or selector.embedder
    if embedder is None:
        raise SystemExit("the selector records no embedder; pass --embedder")
    by_doc: dict[str, list[Any]] = {}
    for q in dataset.questions[: args.limit] if args.limit else dataset.questions:
        by_doc.setdefault(q.doc_id, []).append(q)
    tagger = resolve_tagger(args.tagger)
    encoder = resolve_backend(embedder)
    same = embedder == profile_defaults(args.profile)["sections"]
    reader = qa.make_reader(
        dataset.pdf_dir,
        profile=args.profile,
        max_pages=args.max_pages,
        sections_backend=encoder if same else None,
    )
    rows: list[dict[str, Any]] = []
    skipped = {"whole_document_too_long": 0, "documents_failed": 0}
    try:
        for doc_id, questions in by_doc.items():
            try:
                analysis = reader(qa.record_for(doc_id, dataset.pdf_dir))
                ctx = prepare(analysis, tagger=tagger, encoder=encoder, family=qa.FAMILY)
            except DocrooterError:
                skipped["documents_failed"] += 1
                continue
            doc = document_of(analysis)
            whole = select(analysis, questions[0].question, completeness=1.0)
            if whole.tokens > args.max_context_tokens:
                skipped["whole_document_too_long"] += len(questions)
                continue
            whole_context = _context(doc, whole.element_ids)
            for q in questions:
                _, X = need_features(ctx, q.question)
                chosen = select_scored(
                    analysis, ctx, X, q.question, completeness=COMPLETENESS, selector=selector
                )
                row: dict[str, Any] = {"id": q.id, "doc_id": doc_id}
                for key, sel, context in (
                    ("selected", chosen, _context(doc, chosen.element_ids)),
                    ("whole", whole, whole_context),
                ):
                    answer = model.ask(q.question, context)
                    row[key] = {"correct": score(answer, q.answer), "tokens": sel.tokens}
                rows.append(row)
    finally:
        reader.close()
        close_backend(tagger)
        close_backend(encoder)
        model.close()

    selected, whole_acc = _accuracy(rows, "selected"), _accuracy(rows, "whole")
    results = {
        "date": datetime.date.today().isoformat(),
        "dataset": dataset.name,
        "licence": dataset.licence,
        "evaluation_only": True,
        "completeness": COMPLETENESS,
        "model": args.mlx_model or os.environ.get(ENV_MODEL),
        "scoring": {"exact_or_numeric_relative_tolerance": TOLERANCE},
        "n": len(rows),
        "skipped": skipped,
        "accuracy": {
            "selected": selected,
            "whole": whole_acc,
            "difference": (
                selected - whole_acc if selected is not None and whole_acc is not None else None
            ),
        },
        "mean_tokens": {
            "selected": _mean_tokens(rows, "selected"),
            "whole": _mean_tokens(rows, "whole"),
        },
        "questions": rows,
        "commit": _commit(),
        "seconds": round(time.perf_counter() - started, 1),
    }
    path = _out_dir(args.out) / RESULTS_FILE
    _write_json(path, results)
    print(
        f"extraction: n = {len(rows)}, accuracy selected {selected} vs whole {whole_acc} -> {path}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mlx-model", default=None, help="a local MLX model id")
    parser.add_argument("--tagger", default=None, help="element tagger bundle directory")
    parser.add_argument("--selector", type=Path, default=None, help="selector.json")
    parser.add_argument("--embedder", default=None, help="embedding spec (default: selector's)")
    parser.add_argument("--profile", default="accurate")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--max-context-tokens", type=int, default=DEFAULT_MAX_CONTEXT_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="at most this many questions")
    parser.add_argument("--out", type=Path, default=None, help="results dir (default: dated)")
    _guard.add_guard_argument(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    endpoint = endpoint_from_env()
    if endpoint is None and not args.mlx_model:
        print(
            f"extraction harness: not configured (set {ENV_BASE_URL}, {ENV_MODEL} and "
            f"{ENV_API_KEY}, or pass --mlx-model); nothing to do"
        )
        return 0
    if args.mlx_model:
        refuse_jev(model=args.mlx_model)
    if not args.tagger:
        raise SystemExit("--tagger is required to select context")
    _guard.single_model_process("benchmarks.extraction.run", allow_concurrent=args.allow_concurrent)
    model = MlxLocal(args.mlx_model) if args.mlx_model else OpenAICompatible(endpoint)  # type: ignore[arg-type]
    return run(args, model)


if __name__ == "__main__":
    sys.exit(main())
