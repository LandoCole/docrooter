"""Section normalisation: map each heading to a canonical section of the document's type.

Needs no models (keyword backend), so expect some wrong labels; the default embedding backend
and ``"jev"`` are far better (see the benchmarks).
"""

from docrooter import Taxonomy, load, normalize_sections

CONTRACT = """\
# Master Services Agreement

## 1. Definitions
In this agreement the following terms have the meanings given.

## 2. Term
This agreement starts on the effective date and lasts two years.

## 3. Fees and payment
The customer shall pay the fees within 30 days of the invoice.

## 4. Termination
Either party may terminate the agreement on written notice.
"""

doc = load(CONTRACT)

# The document type picks the section taxonomy: legal.contract -> section_types/contract.
norm = normalize_sections(doc, doc_type="legal.contract", backend="keyword")
print(norm.taxonomy, norm.doc_type)
for section_id, section in norm.sections.items():
    print(f"{section_id} {section.title!r} -> {section.canonical} ({section.score:.2f})")

# Or name a built-in section taxonomy directly, and choose which heading levels to label.
norm = normalize_sections(doc, taxonomy="contract", backend="keyword", levels=[1])
print(sorted({section.canonical for section in norm.sections.values()}))

# Without doc_type or taxonomy, the document type is classified first (same backend);
# its confidence is recorded so a pipeline can flag a doubtful choice.
auto = normalize_sections(doc, backend="keyword")
print(auto.doc_type, round(auto.doc_type_confidence, 2), auto.taxonomy)

# Your own canonical sections: any Taxonomy works. Examples (synonyms) help most backends.
mine = Taxonomy.from_dict(
    {
        "nodes": [
            {"id": "money", "label": "Money", "description": "Fees, payment and invoices"},
            {"id": "ending", "label": "Ending", "description": "Termination and expiry"},
            {"id": "other", "label": "Other", "description": "Anything else"},
        ]
    }
)
custom = normalize_sections(doc, taxonomy=mine, backend="keyword")
print({section.title: section.canonical for section in custom.sections.values()})
