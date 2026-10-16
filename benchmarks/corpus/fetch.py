#!/usr/bin/env python3
"""Download the benchmark corpus PDFs listed in manifest.jsonl and verify them.

Stdlib only. The PDFs are not redistributed; each one is fetched from its original publisher.

    python benchmarks/corpus/fetch.py                # fetch everything missing into ./pdfs
    python benchmarks/corpus/fetch.py --dest /tmp/x  # fetch into another directory
    python benchmarks/corpus/fetch.py --only doc_001 doc_002
    python benchmarks/corpus/fetch.py --lenient      # sha256 mismatch is a warning, not a failure

Files that already exist with the right sha256 are skipped. A sha256 mismatch means the publisher
changed the document at that URL since the manifest was written (some publishers overwrite
files in place); the file is kept but reported, and the exit status is 1 unless --lenient.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER_AGENT = "docrooter-benchmark/0.1 (research)"
DELAY_SECONDS = 1.5
TIMEOUT_SECONDS = 120


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        data = response.read()
        encoding = (response.headers.get("Content-Encoding") or "").lower()
    if encoding == "gzip" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    elif encoding == "deflate":
        data = zlib.decompress(data)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=HERE / "manifest.jsonl")
    parser.add_argument("--dest", type=Path, default=HERE / "pdfs", help="output directory")
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these doc ids")
    parser.add_argument("--lenient", action="store_true", help="do not fail on sha256 mismatch")
    args = parser.parse_args()

    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.only:
        entries = [e for e in entries if e["id"] in set(args.only)]
    args.dest.mkdir(parents=True, exist_ok=True)

    ok = mismatched = failed = skipped = 0
    for entry in entries:
        target = args.dest / Path(entry["file"]).name
        if target.exists() and sha256_of(target) == entry["sha256"]:
            skipped += 1
            continue
        try:
            data = download(entry["url"])
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"FAIL     {entry['id']} {entry['url']} ({exc})")
            failed += 1
            time.sleep(DELAY_SECONDS)
            continue
        if not data.startswith(b"%PDF"):
            print(f"FAIL     {entry['id']} {entry['url']} (response is not a PDF)")
            failed += 1
            time.sleep(DELAY_SECONDS)
            continue
        target.write_bytes(data)
        if hashlib.sha256(data).hexdigest() == entry["sha256"]:
            print(f"OK       {entry['id']} {len(data)} bytes")
            ok += 1
        else:
            print(
                f"MISMATCH {entry['id']} sha256 differs from manifest "
                f"(publisher changed the file): {entry['url']}"
            )
            mismatched += 1
        time.sleep(DELAY_SECONDS)

    print(
        f"downloaded={ok} already_ok={skipped} sha_mismatch={mismatched} "
        f"failed={failed} total={len(entries)}"
    )
    if failed or (mismatched and not args.lenient):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
