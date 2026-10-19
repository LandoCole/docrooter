"""Gate a pipeline on the document type: keep only the financial documents of a folder.

Usage: ``python examples/doctype_gate.py [FOLDER]``. Without a folder it writes three small
Markdown files to a scratch directory first. Needs no models (keyword backend); use the default
embedding backend or ``"jev"`` for real accuracy. The CLI equivalent is
``docrooter doctype FOLDER --keep financial --paths-only``.
"""

import sys
import tempfile
from pathlib import Path

from docrooter import Classifier, builtin_taxonomy, load

SAMPLES = {
    "results.md": "# Quarterly results\n\n## Revenue\nNet revenue and earnings per share rose.\n",
    "lease.md": "# Lease agreement\n\n## Term\nThe tenant shall pay rent. The landlord agrees.\n",
    "paper.md": "# A sparse attention method\n\n## Abstract\nWe report experiments.\n",
}


def main(folder: Path) -> None:
    paths = sorted(p for p in folder.rglob("*") if p.suffix in {".pdf", ".md", ".txt"})
    docs = [load(path) for path in paths]

    # One classifier for every document: the model (if any) loads once, and the
    # documents are classified in one batch. threshold=0.0 always assigns a type.
    with Classifier(builtin_taxonomy("doc_types"), "keyword") as classifier:
        results = classifier.classify_documents(docs, threshold=0.0)

    for path, result in zip(paths, results, strict=True):
        node_ids = result.best.node_ids  # e.g. ["financial", "financial.earnings_release"]
        keep = bool(node_ids) and node_ids[0] == "financial"
        print(f"{'keep' if keep else 'skip'}  {path.name}: {result.best.labels}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        scratch = Path(tempfile.mkdtemp())
        for name, text in SAMPLES.items():
            (scratch / name).write_text(text, encoding="utf-8")
        main(scratch)
