"""Build the two corpus-style annual reports the ``models`` selection test reads
(``tests/integration/test_select_models.py``). The companies, people and numbers are invented.

    PYTHONPATH=src .venv/bin/python tests/fixtures/select/generate.py

The PDFs are built only when missing, so a rerun keeps the committed files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WIDTH, HEIGHT, MARGIN = 612, 792, 72
BODY_SIZE, LEADING = 10.5, 14.5

REPORTS: dict[str, dict[str, Any]] = {
    "northwind_annual_report.pdf": {
        "title": "Northwind Instruments Corporation",
        "subtitle": "Annual Report on Form 10-K for the fiscal year 2031",
        "sections": [
            (
                "Item 1. Business",
                [
                    "Northwind Instruments designs and sells pressure sensors, flow meters and "
                    "control software for water utilities and chemical plants in eleven "
                    "countries.",
                    "We sell through a direct sales force of 140 engineers and through regional "
                    "distributors. Our largest customer accounted for 9% of revenue.",
                ],
            ),
            (
                "Item 1A. Risk Factors",
                [
                    "Our results depend on capital spending by water utilities, which can be "
                    "delayed when municipal budgets tighten. A delay of large projects could "
                    "reduce our revenue materially.",
                    "We buy specialised semiconductors from two suppliers. A shortage or a price "
                    "increase could interrupt production and harm our margins.",
                    "About 40% of our sales are in currencies other than the dollar, so a "
                    "stronger dollar could reduce our reported revenue and earnings.",
                    "A cyber attack on our control software could expose customers to outages "
                    "and expose us to liability claims and lost business.",
                ],
            ),
            (
                "Item 2. Properties",
                [
                    "We own a factory of 210,000 square feet in Ashford and lease two "
                    "assembly sites and five sales offices.",
                ],
            ),
            (
                "Item 3. Legal Proceedings",
                [
                    "A former distributor filed a claim in 2031 alleging breach of contract and "
                    "seeking damages of 6 million. We believe the claim is without merit and "
                    "intend to defend it.",
                ],
            ),
            (
                "Item 7. Management's Discussion and Analysis",
                [
                    "Revenue grew 8% to 912 million, led by flow meters. Gross margin fell one "
                    "point to 44% as component prices rose.",
                    "Operating cash flow was 131 million. We repaid 40 million of term debt and "
                    "repurchased shares for 25 million.",
                ],
            ),
        ],
    },
    "harbor_lane_annual_report.pdf": {
        "title": "Harbor Lane Foods, Inc.",
        "subtitle": "Annual Report 2031",
        "sections": [
            (
                "Business Overview",
                [
                    "Harbor Lane Foods makes frozen meals and soups sold under four brands in "
                    "grocery stores across the northern region.",
                    "We operate three plants and a central distribution centre, and employ about "
                    "2,300 people.",
                ],
            ),
            (
                "Risk Factors",
                [
                    "Prices of wheat, vegetables and packaging can rise quickly. If we cannot "
                    "pass higher costs on to retailers, our margins would decline.",
                    "Three grocery chains buy more than half of our products. The loss of one of "
                    "them would reduce our sales significantly.",
                    "A product recall caused by contamination could damage our brands and lead "
                    "to costly claims.",
                    "Our plants depend on a small number of skilled workers, and labour "
                    "shortages could limit production.",
                ],
            ),
            (
                "Legal Matters",
                [
                    "We are party to routine claims arising from our business. We do not expect "
                    "any of them to have a material effect on our results.",
                ],
            ),
            (
                "Results of Operations",
                [
                    "Net sales rose 5% to 618 million. Operating income was 49 million, compared "
                    "with 44 million the year before.",
                    "Capital spending was 31 million, mostly for a new soup line at the "
                    "Brookfield plant.",
                ],
            ),
        ],
    },
}


def _wrap(text: str, font: Any, size: float, width: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if font.text_length(trial, fontsize=size) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def build(name: str, spec: dict[str, Any]) -> Any:
    import pymupdf

    font = pymupdf.Font("helv")
    doc = pymupdf.open()
    page = doc.new_page(width=WIDTH, height=HEIGHT)
    y = 110.0
    page.insert_text((MARGIN, y), spec["title"], fontsize=20, fontname="hebo")
    y += 28
    page.insert_text((MARGIN, y), spec["subtitle"], fontsize=12, fontname="helv")
    y += 40
    number = 1
    for heading, paragraphs in spec["sections"]:
        if y > HEIGHT - 200:
            page.insert_text((WIDTH / 2, HEIGHT - 40), str(number), fontsize=8, fontname="helv")
            page = doc.new_page(width=WIDTH, height=HEIGHT)
            number += 1
            y = 90.0
        page.insert_text((MARGIN, y), heading, fontsize=14, fontname="hebo")
        y += 24
        for text in paragraphs:
            for line in _wrap(text, font, BODY_SIZE, WIDTH - 2 * MARGIN):
                page.insert_text((MARGIN, y), line, fontsize=BODY_SIZE, fontname="helv")
                y += LEADING
            y += 18  # a blank line between paragraphs
        y += 14
    page.insert_text((WIDTH / 2, HEIGHT - 40), str(number), fontsize=8, fontname="helv")
    return doc


def main() -> None:
    for name, spec in REPORTS.items():
        path = HERE / name
        if path.exists():
            continue
        doc = build(name, spec)
        doc.set_metadata({"title": spec["title"], "producer": "docrooter test fixture"})
        doc.save(path, garbage=4, deflate=True)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
