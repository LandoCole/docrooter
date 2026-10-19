"""Page count and text-layer size of one downloaded PDF, printed as one JSON line.

The fetcher runs this as ``python -I pdf_meta.py <file>``, in a child process isolated from the
environment and the current directory, because downloaded PDFs are untrusted data. It only
parses: nothing in the file is executed. Stand-alone on purpose (imports nothing from the repo).
"""

from __future__ import annotations

import json
import sys

TEXT_PAGES = 5  # the text-layer check reads the first pages only


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: pdf_meta.py FILE", file=sys.stderr)
        return 2
    import pymupdf

    with pymupdf.open(argv[1]) as doc:
        pages = doc.page_count
        if doc.needs_pass:
            print(json.dumps({"pages": pages, "text_chars": 0, "encrypted": True}))
            return 0
        chars = sum(len(doc[i].get_text()) for i in range(min(TEXT_PAGES, pages)))
    print(json.dumps({"pages": pages, "text_chars": chars, "encrypted": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
