"""Commercial contracts from CUAD v1, the Atticus Project's Contract Understanding Atticus
Dataset (controller ruling, in place of the blocked EDGAR exhibit-10 source).

CUAD v1 is licensed CC BY 4.0. Its about 510 contract PDFs were originally filed on EDGAR as
material-contract exhibits. The official archive (``CUAD_v1.zip`` on Zenodo record 4595826,
about 106 MB) is downloaded once into ``<cache>/archives/cuad/``, politely (zenodo.org's robots.txt
crawl delay is 10 s), with a ZIP-magic check, a size cap and a pinned sha256. The archive is
untrusted data: it is never extracted as a tree and nothing in it is executed. The fetcher reads
single PDF members out of it, with the same magic and size checks as a download.

Only the contract PDFs are used: they are listed by archive path, shuffled with seed 0, and the
fetcher takes about 100 (cap 150). CUAD's clause annotations (``master_clauses.csv``, the JSON
and the label spreadsheets) are never read, so no CUAD label enters training, calibration or the
gold set.

Attribution, kept in every manifest record: Hendrycks, Burns, Chen and Ball, "CUAD: An
Expert-Annotated NLP Dataset for Legal Contract Review" (NeurIPS 2021), The Atticus Project,
CC BY 4.0.
"""

from __future__ import annotations

import hashlib
import random
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from benchmarks.training_corpus._net import FetchError, Net, Rejected
from benchmarks.training_corpus.sources.base import Candidate, Source, safe_id

NAME = "cuad"
ARCHIVE_URL = "https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1"
ARCHIVE_NAME = "CUAD_v1.zip"
ARCHIVE_SHA256 = "88b694d99007d39777fa44cd72daf8297773d285dc3eab0091ba32078888d18e"
ARCHIVE_MAX_BYTES = 200 * 1024 * 1024
ZIP_MAGIC = b"PK\x03\x04"
LICENCE = "CC-BY-4.0"
ATTRIBUTION = (
    "CUAD v1, The Atticus Project: Hendrycks, Burns, Chen and Ball, 'CUAD: An Expert-Annotated "
    "NLP Dataset for Legal Contract Review', NeurIPS 2021; CC BY 4.0; "
    "https://www.atticusprojectai.org/cuad"
)
CONTRACT_DIR = "full_contract_pdf"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_archive(net: Net, expected_sha256: str | None = None) -> Path:
    """The cached archive, downloaded once (and re-downloaded only if its hash is wrong)."""
    expected_sha256 = expected_sha256 or ARCHIVE_SHA256
    if net.archive_dir is None:
        raise FetchError("no archive directory configured")
    path = net.archive_dir / NAME / ARCHIVE_NAME
    if path.exists() and _sha256(path) == expected_sha256:
        return path
    path.unlink(missing_ok=True)
    net.download(ARCHIVE_URL, path, max_bytes=ARCHIVE_MAX_BYTES, magic=ZIP_MAGIC)
    actual = _sha256(path)
    if actual != expected_sha256:
        path.unlink()
        raise Rejected(f"CUAD archive sha256 {actual} is not the pinned {expected_sha256}")
    return path


def contract_members(archive: Path) -> list[str]:
    """The archive's contract PDFs: ``.pdf`` members under ``full_contract_pdf/``, without
    macOS resource forks or odd paths. Read from the central directory only."""
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    out = []
    for name in names:
        parts = PurePosixPath(name).parts
        if CONTRACT_DIR not in parts or not name.lower().endswith(".pdf"):
            continue
        if any(p in ("..", "") or p.startswith(".") or p == "__MACOSX" for p in parts):
            continue
        out.append(name)
    return sorted(out)


def _title(member: str) -> tuple[str, str]:
    """(title, party) from a CUAD file name such as
    ``LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.pdf``."""
    stem = PurePosixPath(member).stem
    party = re.split(r"_\d", stem, maxsplit=1)[0].replace("_", " ").strip()
    kind = stem.rsplit("-", 1)[-1].strip() if "-" in stem else "agreement"
    return f"{party} {kind}".strip(), party


def sample(members: list[str], seed: int = 0) -> list[str]:
    order = list(members)
    random.Random(f"{seed}:{NAME}").shuffle(order)
    return order


def candidates(net: Net) -> Iterator[Candidate]:
    archive = ensure_archive(net)
    for member in sample(contract_members(archive)):
        title, party = _title(member)
        yield Candidate(
            id=safe_id("cuad", PurePosixPath(member).stem),
            url=f"{ARCHIVE_URL.split('?', 1)[0]}#{member}",
            source=NAME,
            licence=LICENCE,
            doc_type_hint="legal.contract",
            title=title,
            publisher=party,
            extra={"cuad_file": member, "attribution": ATTRIBUTION},
            archive=archive,
            archive_member=member,
        )


SOURCE = Source(
    name=NAME,
    target=100,
    cap=150,
    probe_url=ARCHIVE_URL,
    candidates=candidates,
)
