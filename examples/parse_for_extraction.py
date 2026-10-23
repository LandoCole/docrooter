"""Parse a document once; the extraction step reads the stored JSON. Needs no models.

The call for a real run on a PDF (needs the Jev key; see the guide) is:

    parse("q3.pdf", include="text", backend="jev")

Here the keyword backend stands in, on a small Markdown filing, so the script runs offline.
"""

from docrooter import ParseResult, parse

MARKDOWN = """\
# Acme Corp Quarterly Report

## Balance Sheet
Total assets were 120 and total liabilities were 70.

### Current assets
Cash and receivables.

## Income Statement
Revenue was 90 and net income was 12.

## Risk Factors
Our business faces risks.
"""

result = parse(MARKDOWN, include="text", backend="keyword")
print(f"document type: {result.doc_type}")
for tag, matches in result.section_text.items():
    for match in matches:
        print(f"{tag}: {match.heading}")

stored = result.model_dump_json()  # keep this in your database
# Later, in the extraction step: read the JSON back and ask for a tag. No call to doc-rooter.
balance_sheet = ParseResult.model_validate_json(stored).get_section("financial.balance_sheet")
print(balance_sheet.status)
print(balance_sheet.sections[0].text)
