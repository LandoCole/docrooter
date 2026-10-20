"""classify(): tag the sections, then let their tags decide or narrow the document type.

Needs no models and no API key: the ``keyword`` backend only counts shared words, so it runs
right after ``pip install docrooter`` and shows the API, the context ladder and the token
accounting. Its tags are weak and its confidences low, so the doc-type rules are not met here
and the document step chooses among every type (``rules_unmet``). Swap in
``backend="openai:<model>"`` for a hosted or local LLM, whose tokens are then exact.
"""

from docrooter import LEVELS, classify, load
from docrooter.ladder import document_view, section_view

MARKDOWN = """\
# Sparse retrieval for long documents

## Abstract
A short summary of the problem, our approach and the main findings of this paper.

## Related work
A survey of prior literature and existing approaches to retrieval.

## Method
The proposed approach and algorithm, and how the data were collected.

## Results
The measurements and findings of the experiments, in tables and figures.

## References
Bibliography and citations of the sources used.
"""

doc = load(MARKDOWN)

# The context ladder: each rung shows more of every section and of the document.
for level in LEVELS:
    rendered = section_view(level).render_all(doc, [s.id for s in doc.walk()])
    section_tokens = sum(view.n_tokens for view in rendered.values())
    doc_tokens = document_view(level).render(doc).n_tokens
    print(f"{level:<9} sections {section_tokens:>3} tokens, document {doc_tokens:>3} tokens")

# Sections first: tag the sections against the catalogue, then decide the document type.
result = classify(doc, level="terms", backend="keyword")
flags = ", ".join(result.flags) or "none"
print(f"doc type: {result.doc_type} (decided by {result.decided_by}; flags: {flags})")
for tag in result.sections:
    print(f"  {tag.title}: {tag.label}")
kind = "exact" if result.tokens.exact else "estimated"
print(f"tokens ({kind}): {result.tokens.prompt:,} prompt over {result.tokens.calls} calls")
