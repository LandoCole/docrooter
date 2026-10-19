"""IETF RFCs as PDF (research brief §3.4).

The RFC Editor publishes a PDF rendering of every RFC in the v3 XML format, from RFC 8650 on, at
``https://www.rfc-editor.org/rfc/rfc<N>.pdf``. No listing request is needed: RFC numbers 8650-9850
are taken in an order shuffled with seed 0, and numbers with no PDF (404) are skipped. RFCs are
technical, not finance, so this stays a small slice (target 40, cap 60).

Licence: IETF Trust Legal Provisions: RFCs may be copied and distributed in full without
modification; we keep them unmodified and local.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from benchmarks.training_corpus._net import Net
from benchmarks.training_corpus.sources.base import Candidate, Source

NAME = "rfc"
LICENCE = "IETF Trust Legal Provisions (copy unmodified); kept locally, not redistributed"
BASE = "https://www.rfc-editor.org/rfc"
FIRST, LAST = 8650, 9850


def numbers(seed: int = 0) -> list[int]:
    order = list(range(FIRST, LAST + 1))
    random.Random(f"{seed}:{NAME}").shuffle(order)
    return order


def candidates(net: Net) -> Iterator[Candidate]:
    for number in numbers():
        yield Candidate(
            id=f"rfc-{number}",
            url=f"{BASE}/rfc{number}.pdf",
            source=NAME,
            licence=LICENCE,
            doc_type_hint="research.technical_standard",
            title="",
            publisher="RFC Editor",
            extra={"rfc": number},
        )


SOURCE = Source(
    name=NAME,
    target=40,
    cap=60,
    probe_url=f"{BASE}/",
    candidates=candidates,
)
