"""The TOC engine: load a document, walk its section tree, and inspect how headings were found.

Usage: ``python examples/toc.py [FILE]``. Without a file it uses the Markdown below; with a PDF
it needs ``docrooter[pdf]`` or ``docrooter[pdf-mit]``. Needs no models.
"""

import sys

from docrooter import Document, load, load_pdf

MARKDOWN = """\
# Annual report

Preamble text before the first numbered section.

## 1. Business
We design and sell products.

### 1.1 Products
Phones and computers.

## 2. Risk factors
Our business faces risks.
"""


def show(doc: Document) -> None:
    for entry in doc.toc():  # reading order; depth is the position in the tree
        section = doc.sections[entry.id]
        flags = f"  {section.flags}" if section.flags else ""
        print(
            f"{'  ' * entry.depth}{entry.title} [level {entry.level}, p. {section.page_start}, "
            f"score {section.heading_score:.2f}]{flags}"
        )


def main(path: str | None) -> None:
    if path is None:
        doc = load(MARKDOWN)
        show(doc)
        print([leaf.title for leaf in doc.leaves()])  # sections without children
        print(doc.section("2. Risk factors").text)  # look up by id or by title
        return

    doc = load(path)  # .pdf goes to load_pdf; .md / .markdown / .txt are read as Markdown
    show(doc)
    print({key: doc.meta[key] for key in ("engine", "n_pages", "timings", "warnings")})

    if path.endswith(".pdf"):
        # The same PDF without bookmark help, reading at most 50 pages.
        plain = load_pdf(path, use_bookmarks=False, max_pages=50)
        print(len(plain.toc()), "headings without bookmarks;", plain.meta["warnings"])


# A document can also be built from headings you already have (e.g. from another parser).
built = Document.from_sections(
    [
        {"title": "Introduction", "level": 1, "text": "Why this matters."},
        {"title": "Background", "level": 2, "text": "Prior work."},
        {"title": "Method", "level": 1},
    ]
)
print([(entry.title, entry.depth) for entry in built.toc()])

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
