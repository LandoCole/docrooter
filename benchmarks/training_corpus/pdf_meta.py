"""Page count and text-layer size of one downloaded PDF, printed as one JSON line.

The fetcher runs this as ``python -I pdf_meta.py <file>``, in a child process isolated from the
environment and the current directory, because downloaded PDFs are untrusted data. It only
parses: nothing in the file is executed. Stand-alone on purpose (imports nothing from the repo).
"""

from __future__ import annotations

import json
import sys

TEXT_PAGES = 5  # the text-layer check reads the first pages only
MEMORY_CAP = 2 << 30  # 2 GB of address space, where the OS enforces RLIMIT_AS / RLIMIT_DATA


def cap_memory(limit: int = MEMORY_CAP) -> bool:
    """Lower this process's memory limits to ``limit`` bytes; whether any limit took. Linux
    enforces it; macOS refuses it, so the parent also watches this process's resident size."""
    try:
        import resource
    except ImportError:  # not POSIX
        return False
    capped = False
    for name in ("RLIMIT_AS", "RLIMIT_DATA"):
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            _, hard = resource.getrlimit(which)
            value = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(which, (value, hard))
            capped = True
        except (ValueError, OSError):
            continue
    return capped


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: pdf_meta.py FILE", file=sys.stderr)
        return 2
    capped = cap_memory()
    import pymupdf

    with pymupdf.open(argv[1]) as doc:
        pages = doc.page_count
        encrypted = bool(doc.needs_pass)
        chars = 0
        if not encrypted:
            chars = sum(len(doc[i].get_text()) for i in range(min(TEXT_PAGES, pages)))
    info = {"pages": pages, "text_chars": chars, "encrypted": encrypted, "memory_cap": capped}
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
