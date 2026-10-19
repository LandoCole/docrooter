"""Send only the relevant parts of a document to your LLM: ``select()`` and the completeness dial.

Usage: ``python examples/select_for_extraction.py``. It needs no model and no trained bundle: the
tagger and the embedder are keyword stand-ins and the selector's weights and dial are set by hand,
so you can see what a Selection holds. A hand-set dial is NOT calibrated: its guarantee means
nothing. A real one comes from ``docrooter fit-selector`` (see the guide, "Select what to send
to an LLM").

With a trained bundle (a tagger and its ``selector.json``), edit the two paths in
``select_from_bundle`` and run ``python examples/select_for_extraction.py --bundle``.

docrooter selects; your application extracts. The last step here only prints the text you would
put in your prompt.
"""

import sys

import numpy as np

from docrooter import Analysis, Document, RouterConfig, builtin_fields, process, select
from docrooter.classify.results import DocTypeResult, PathResult
from docrooter.elements.models import Element, TagResult
from docrooter.select import Selector
from docrooter.select.crc import DialTable
from docrooter.select.relevance import FEATURE_NAMES, Combiner
from docrooter.select.selection import readout

RISK = {f.id: f for f in builtin_fields("annual_report")}["annual_report.risk_factors"]


def select_from_bundle() -> None:
    """The real thing: a tagged document, a fitted selector (edit these two paths)."""
    risk = {f.id: f for f in builtin_fields("annual_report")}["annual_report.risk_factors"]
    analysis = process("report.pdf", element_tags="bundles/tagger-a")
    selection = select(analysis, risk, completeness=0.9, selector="bundles/tagger-a")
    print(readout(selection))
    print("\n\n".join(chunk.text for chunk in selection.chunks))


# --- the guide's snippets: each one needs a field file, a bundle or a corpus ----------------------


def list_fields() -> None:
    """The built-in fields of a family."""
    from docrooter import builtin_fields

    for field in builtin_fields("fund_document"):
        print(field.id, "-", field.label)


def check_your_fields():
    """Your own field set (JSON): read it, then check it against the taxonomies."""
    from docrooter import builtin_taxonomy, load_fields
    from docrooter.select import validate_fields

    fields = load_fields("my_fields.json")
    topics = builtin_taxonomy("element_topics")
    functions = builtin_taxonomy("element_functions")
    validate_fields(fields, topics, functions)
    return fields


def read_the_selection(analysis, field) -> str:
    """Select for one field and read the one-line summary."""
    from docrooter import select
    from docrooter.select.selection import readout

    selection = select(analysis, field, completeness=0.9, selector="bundles/tagger-a")
    print(readout(selection))
    context = "\n\n".join(chunk.text for chunk in selection.chunks)
    # Your application, not docrooter: send `context` to your model with your own prompt.
    return context


def refit_with_your_fields(fields) -> None:
    """Fit a selector on your corpus with your own field set (the dial is calibrated on it)."""
    from docrooter.corpus import Corpus
    from docrooter.select import fit_selector

    selector = fit_selector(
        Corpus("corpus/"),
        "bundles/tagger-a",
        "embedding:dmis-lab/UltraBERT-Embed-150M",
        fields_by_family={"annual_report": fields},
        pdf_dir="corpus/pdfs",
    )
    selector.save("bundles/tagger-a/selector.json")


# --- stand-ins: no model, no training ----------------------------------------------------------

SECTIONS = ["Business", "Risk factors", "Properties", "Financial statements"]
BODY = [
    (0, "We make sensors for factories and power plants."),
    (0, "Demand grew in every region during the year."),
    (1, "Interest rate risk could raise our borrowing costs."),
    (1, "A shortage of components is a supply risk for our two plants."),
    (1, "Currency risk follows from sales in six currencies."),
    (2, "We own one factory and lease three warehouses."),
    (3, "Revenue rose to 1,204 million."),
    (3, "Operating income was 311 million."),
]


class KeywordTagger:
    """Tags a paragraph that mentions risk as risk factors, and any other as the family's
    catch-all. The real one is the student you trained (``element_tags``)."""

    def tag(self, elements, family):
        out = []
        for element in elements:
            leaf = "risk_factors" if "risk" in element.text.lower() else "other"
            topic = f"{family}.{leaf}"
            tags = TagResult(labels=[topic], probs={topic: 1.0}, confidence=1.0, source="keyword")
            out.append({"topic": tags})
        return out


class KeywordEmbedder:
    """One direction for texts that mention risk, another for the rest."""

    def encode(self, texts, prompt=""):
        return np.array([[1.0, 0.0] if "risk" in t.lower() else [0.0, 1.0] for t in texts])


def hand_set_selector() -> Selector:
    """Relevance is the tagger's topic probability alone; the dial keeps what scores 0.5 or more
    at c = 0.9. A real selector is fitted: this one is only for the demonstration."""

    def combiner(names, weights):
        return Combiner.from_dict(
            {
                "format": "docrooter-relevance-combiner",
                "version": 1,
                "feature_names": list(names),
                "l2": 1.0,
                "max_iter": 100,
                "mean": [0.0] * len(names),
                "scale": [1.0] * len(names),
                "coef": [weights.get(name, 0.0) for name in names],
                "intercept": -5.0,
                "n_train": 10,
                "n_docs": 2,
                "converged": True,
            }
        )

    field = combiner(FEATURE_NAMES, {"p_topic": 10.0})
    fallback = combiner(("sim", "section_hit"), {"sim": 10.0})
    dial = DialTable(
        family="annual_report",
        n=100,
        n_docs=100,
        n_excluded=0,
        grid=(0.5, 0.9),
        curve_lambdas=[0.0, 0.5, 1.0],
        curve_risk=[0.0, 0.0, 1.0],
        combiner_fingerprint=field.fingerprint(),
    )
    return Selector(
        field=field,
        field_fallback=fallback,
        query=combiner(FEATURE_NAMES, {"sim": 10.0}),
        query_fallback=fallback,
        dials={"annual_report": dial},
    )


def sample_analysis() -> Analysis:
    """An annual report of four sections, built by hand (no PDF, no model)."""
    doc = Document.from_sections([{"title": title, "level": 1} for title in SECTIONS])
    elements = []
    for number, title in enumerate(SECTIONS):
        section_id = f"s{number + 1}"
        items = [("heading", title, f"h{number}")]
        items += [
            ("paragraph", text, f"p{i}") for i, (owner, text) in enumerate(BODY) if owner == number
        ]
        for kind, text, element_id in items:
            fields = {"level": 1} if kind == "heading" else {}
            elements.append(
                Element(
                    id=element_id,
                    kind=kind,
                    text=text,
                    boxes=[],
                    order=len(elements),
                    section_id=section_id,
                    **fields,
                )
            )
            doc.sections[section_id].element_ids.append(element_id)
    doc.elements = elements
    result = DocTypeResult(
        source="example", best=PathResult(node_ids=["financial", "financial.annual_report"])
    )
    return Analysis(document=doc, doc_type=result, config=RouterConfig(profile="fast"))


def main() -> None:
    if "--bundle" in sys.argv[1:]:
        select_from_bundle()
        return
    analysis = sample_analysis()
    selection = select(
        analysis,
        RISK,
        completeness=0.9,
        selector=hand_set_selector(),
        tagger=KeywordTagger(),
        embedder=KeywordEmbedder(),
    )
    print(readout(selection))
    print(f"guarantee={selection.guarantee} capped={selection.capped}")
    print("elements:", ", ".join(selection.element_ids))
    print()
    print("\n\n".join(chunk.text for chunk in selection.chunks))
    whole = select(analysis, RISK, completeness=1.0)  # c = 1.0: the whole document
    print()
    print(readout(whole))


if __name__ == "__main__":
    main()
