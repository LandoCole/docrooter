"""The pinned MLX teacher (Qwen3-4B-Instruct-2507, 4-bit) on 20 hand-written elements.

Run with ``pytest -m teach -s`` on Apple silicon with the ``teach-mlx`` extra, and only when no
other model process runs (one model process at a time, 8 GiB budget: ``benchmarks._guard``).
The first run downloads the pinned weights (~2.3 GB) into the Hugging Face cache.

Checks that the family prefix (spec §4.0 ruling 5) is scored as the whole prompt would be,
that the cached prefix and the last-position head change nothing but speed, and that the
teacher gets most of the hand-written items right. Prints items/s with and without the prefix
cache and the peak memory.
"""

from __future__ import annotations

import platform
import resource
import sys
import time
import warnings

import pytest

from benchmarks import _guard
from docrooter import builtin_taxonomy
from docrooter.teach import TeachItem, context_text, function_question, get_teacher, table_text
from docrooter.teach.mlx import DEFAULT_TEACHER

pytestmark = [
    pytest.mark.teach,
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64", reason="mlx needs Apple silicon"
    ),
]
pytest.importorskip("mlx_lm")

TOPICS = builtin_taxonomy("element_topics")
FUNCTIONS = builtin_taxonomy("element_functions")
DOC_TYPE = {
    "annual_report": "Annual report",
    "fund_document": "Prospectus",
    "contract": "Contract",
    "research_paper": "Research paper",
    "technical_document": "Technical manual",
    "generic": "Government report",
}

# (family, heading path, kind, text, gold leaf, four distractor leaves): written for this test.
ITEMS = [
    (
        "annual_report",
        ["Item 1A. Risk Factors"],
        "paragraph",
        "A prolonged disruption at any of our three contract manufacturers could delay shipments "
        "and materially reduce our revenue and margins.",
        "risk_factors",
        ["business_overview", "management_discussion", "legal_proceedings", "properties"],
    ),
    (
        "annual_report",
        ["Item 2. Properties"],
        "paragraph",
        "Our headquarters occupy 210,000 square feet of leased office space in Austin, Texas, "
        "and we own two distribution centers in Ohio and Nevada.",
        "properties",
        ["business_overview", "risk_factors", "exhibits", "sustainability"],
    ),
    (
        "annual_report",
        ["Item 11. Executive Compensation", "Summary Compensation Table"],
        "table",
        table_text(
            "Summary Compensation Table",
            "| Name | Year | Salary ($) | Stock awards ($) |\n|---|---|---|---|\n"
            "| J. Rivera, CEO | 2025 | 1,100,000 | 6,250,000 |\n"
            "| K. Osei, CFO | 2025 | 640,000 | 2,100,000 |",
        ),
        "executive_compensation",
        [
            "corporate_governance",
            "financial_statements",
            "market_for_equity",
            "notes_to_financial_statements",
        ],
    ),
    (
        "annual_report",
        ["Notes to Consolidated Financial Statements", "Note 2. Revenue Recognition"],
        "paragraph",
        "Revenue from subscription contracts is recognized ratably over the contract term, "
        "beginning on the date the service is made available to the customer.",
        "notes_to_financial_statements",
        [
            "financial_statements",
            "management_discussion",
            "controls_and_procedures",
            "business_overview",
        ],
    ),
    (
        "fund_document",
        ["Investment Objective"],
        "paragraph",
        "The Fund seeks long-term growth of capital.",
        "investment_objective",
        ["principal_investment_strategies", "performance", "management", "principal_risks"],
    ),
    (
        "fund_document",
        ["Fees and Expenses of the Fund"],
        "table",
        table_text(
            "Annual Fund Operating Expenses",
            "| | Class A | Class I |\n|---|---|---|\n| Management fees | 0.55% | 0.55% |\n"
            "| Distribution (12b-1) fees | 0.25% | None |\n| Total | 0.92% | 0.67% |",
        ),
        "fees_and_expenses",
        ["performance", "purchase_and_sale", "holdings_and_allocation", "management"],
    ),
    (
        "fund_document",
        ["Principal Risks"],
        "paragraph",
        "Interest Rate Risk. When interest rates rise, the value of the Fund's bond holdings will "
        "generally fall, and you could lose money on your investment.",
        "principal_risks",
        ["principal_investment_strategies", "performance", "tax_information", "fees_and_expenses"],
    ),
    (
        "fund_document",
        ["Purchase and Sale of Fund Shares"],
        "paragraph",
        "The minimum initial investment is $2,500 for regular accounts. You may redeem shares on "
        "any business day by mail, by telephone, or through your financial intermediary.",
        "purchase_and_sale",
        ["fees_and_expenses", "tax_information", "management", "investment_objective"],
    ),
    (
        "contract",
        ["Article 9. Confidentiality"],
        "paragraph",
        "Each party shall hold the other party's Confidential Information in strict confidence and "
        "shall not disclose it to any third party without prior written consent.",
        "confidentiality",
        ["intellectual_property", "warranties", "general_provisions", "termination"],
    ),
    (
        "contract",
        ["Article 12. Governing Law"],
        "paragraph",
        "This Agreement shall be governed by the laws of the State of Delaware, and any "
        "dispute shall be resolved exclusively in the state or federal courts located in "
        "Wilmington, Delaware.",
        "governing_law_disputes",
        ["general_provisions", "termination", "indemnification", "definitions"],
    ),
    (
        "contract",
        ["Article 4. Fees and Payment"],
        "paragraph",
        "Customer shall pay all undisputed invoices within thirty (30) days of receipt. Late "
        "amounts accrue interest at one percent (1%) per month.",
        "payment",
        ["scope_services", "term", "limitation_of_liability", "warranties"],
    ),
    (
        "contract",
        ["Article 1. Definitions"],
        "list_item",
        '"Affiliate" means any entity that directly or indirectly controls, is controlled by, '
        "or is under common control with a party.",
        "definitions",
        ["parties_recitals", "general_provisions", "scope_services", "intellectual_property"],
    ),
    (
        "research_paper",
        ["Abstract"],
        "paragraph",
        "We introduce a sparse attention mechanism that reduces the memory cost of long-context "
        "transformers from quadratic to linear, and show it matches dense attention on four "
        "benchmarks.",
        "abstract",
        ["introduction", "conclusion", "method", "results"],
    ),
    (
        "research_paper",
        ["4 Experiments", "4.1 Setup"],
        "paragraph",
        "We train all models for 100k steps with AdamW, a batch size of 256 and a cosine "
        "learning-rate "
        "schedule, on eight A100 GPUs.",
        "experiments",
        ["method", "results", "appendix", "related_work"],
    ),
    (
        "research_paper",
        ["References"],
        "list_item",
        "[12] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention is all you need. In Advances in "
        "Neural Information Processing Systems, 2017.",
        "references",
        ["related_work", "acknowledgements", "appendix", "background"],
    ),
    (
        "research_paper",
        ["2 Related Work"],
        "paragraph",
        "Earlier work on efficient attention falls into two families: low-rank approximations, "
        "such as Linformer, and fixed sparsity patterns, such as Longformer and BigBird.",
        "related_work",
        ["introduction", "background", "method", "discussion"],
    ),
    (
        "technical_document",
        ["Installation"],
        "list_item",
        "Download the installer, run sudo ./install.sh, then restart the service with systemctl "
        "restart fluxd.",
        "installation_getting_started",
        ["configuration", "usage", "troubleshooting", "overview"],
    ),
    (
        "technical_document",
        ["Troubleshooting", "Error 503 on startup"],
        "paragraph",
        "If the daemon exits with error 503 on startup, check that port 8443 is free and that the "
        "certificate path in fluxd.conf points to a readable file.",
        "troubleshooting",
        ["configuration", "installation_getting_started", "reference", "usage"],
    ),
    (
        "generic",
        ["Executive Summary"],
        "paragraph",
        "This report finds that the agency's grant oversight improved in 2025, but that three of "
        "its eleven programs still lack performance measures. We make four recommendations.",
        "summary",
        ["introduction", "conclusion", "body", "front_matter"],
    ),
    (
        "generic",
        ["Appendix II: Scope and Methodology"],
        "paragraph",
        "To address our objectives, we reviewed agency documents from fiscal years 2021 through "
        "2025 and interviewed officials at four regional offices.",
        "appendix",
        ["body", "introduction", "references", "summary"],
    ),
]

# (function tag, text, gold answer)
YES_NO = [
    (
        "forward_looking",
        "We expect capital expenditures to decline by about 10% in fiscal 2027.",
        True,
    ),
    (
        "forward_looking",
        "Revenue for fiscal 2025 was $4.2 billion, up 8% from the prior year.",
        False,
    ),
    (
        "obligation",
        "The Supplier shall deliver all Products to the Customer's warehouse within ten days.",
        True,
    ),
    ("citation", "Revenue for fiscal 2025 was $4.2 billion, up 8% from the prior year.", False),
]

INSTRUCTION = (
    "You label the elements of a {family}. Which part of the document is this element about? "
    "Use the section heading as a hint, but decide by the element's content."
)


def leaf_option(leaf_id: str) -> tuple[str, str]:
    node = TOPICS.node(leaf_id)
    return leaf_id, f"{node.label}: {node.description}"


def family_options(family: str) -> list[tuple[str, str]]:
    return [leaf_option(child.id) for child in TOPICS.children(family)]


def build(item: tuple) -> tuple[str, TeachItem, str]:
    family, path, kind, text, gold, distractors = item
    shortlist = [f"{family}.{leaf}" for leaf in [gold, *distractors]]
    shortlist.sort()  # the gold leaf is not first
    if f"{family}.other" not in shortlist:
        shortlist.append(f"{family}.other")
    context = context_text(DOC_TYPE[family], path, kind)
    options = [leaf_option(leaf) for leaf in shortlist]
    return family, TeachItem(text=text, context=context, options=options), f"{family}.{gold}"


def run_all(teacher, built) -> list[dict[str, float]]:
    out = []
    for family, item, _ in built:
        label = TOPICS.node(family).label.lower()
        (probs,) = teacher.choose(
            [item],
            instruction=INSTRUCTION.format(family=label),
            prefix_key=f"topic/{family}",
            prefix_options=family_options(family),
        )
        out.append(probs)
    return out


@pytest.fixture(scope="module")
def guarded():
    _guard.apply_env_caps()
    others = _guard.other_model_processes()
    if others:
        pytest.skip(f"another model process is running: {others[0][1][:80]}")


def test_mlx_teacher_labels_hand_written_items(guarded, capsys):
    built = [build(item) for item in ITEMS]
    teacher = get_teacher(DEFAULT_TEACHER, max_cached_prefixes=8)
    try:
        started = time.perf_counter()
        teacher.choose(  # load the model, outside the timings
            [built[0][1]], instruction="Warm up.", prefix_key="warmup"
        )
        load_s = time.perf_counter() - started

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)  # every prompt splits on a boundary
            started = time.perf_counter()
            cached = run_all(teacher, built)  # six family prefixes prefilled on the way
            cached_s = time.perf_counter() - started
            started = time.perf_counter()
            again = run_all(teacher, built)  # all prefixes cached: the steady state
            steady_s = time.perf_counter() - started

            questions = []
            for tag, text, gold in YES_NO:
                node = FUNCTIONS.node(tag)
                (p_yes,) = teacher.yes_no(
                    [TeachItem(text=text, context=context_text(None, [], "paragraph"))],
                    question=function_question(node.label, node.description),
                )
                questions.append((tag, p_yes, gold))

        # the head on the answer position alone gives the full logits' scores
        runner = teacher._runner
        head, runner.head = runner.head, None
        teacher._prefixes.clear()
        full = run_all(teacher, built[:4])
        runner.head = head

        mean_mass = teacher.stats["answer_mass_sum"] / teacher.stats["items"]
        peak_gb = teacher.peak_memory_gb()
        stats = dict(teacher.stats)
    finally:
        teacher.close()

    nocache = get_teacher(DEFAULT_TEACHER, use_prefix_cache=False)
    try:
        nocache.choose([built[0][1]], instruction="Warm up.", prefix_key="warmup")
        started = time.perf_counter()
        uncached = run_all(nocache, built)
        uncached_s = time.perf_counter() - started
        peak_gb = max(peak_gb, nocache.peak_memory_gb())
    finally:
        nocache.close()

    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3  # bytes on macOS
    correct = [max(p, key=p.get) == gold for p, (_, _, gold) in zip(cached, built, strict=True)]
    yes_no_correct = [(p >= 0.5) == gold for _, p, gold in questions]
    with capsys.disabled():
        print(f"\nload {load_s:.1f}s; items {len(built)}")
        print(f"cached:   {len(built) / cached_s:.2f} items/s ({cached_s:.1f}s, with 6 prefills)")
        print(f"steady:   {len(built) / steady_s:.2f} items/s ({steady_s:.1f}s, prefixes cached)")
        print(f"uncached: {len(built) / uncached_s:.2f} items/s ({uncached_s:.1f}s)")
        print(f"peak MLX memory {peak_gb:.2f} GiB; peak RSS {rss_gb:.2f} GiB")
        print(f"accuracy {sum(correct)}/{len(correct)}; yes/no {sum(yes_no_correct)}/{len(YES_NO)}")
        print(f"mean answer mass {mean_mass:.3f}; stats {stats}")
        for p, (_, _, gold), ok in zip(cached, built, correct, strict=True):
            best = max(p, key=p.get)
            print(f"  {'ok ' if ok else 'BAD'} gold={gold} best={best} p={p[best]:.2f}")
        for tag, p_yes, gold in questions:
            print(f"  yes/no {tag} gold={gold} p_yes={p_yes:.2f}")

    # the cached prefix, trimmed after every item, scores as a fresh prefill of it does
    for a, b in zip(cached, uncached, strict=True):
        assert a.keys() == b.keys()
        assert max(abs(a[k] - b[k]) for k in a) < 0.02
    for a, b in zip(cached, again, strict=True):
        assert max(abs(a[k] - b[k]) for k in a) < 1e-3
    for a, b in zip(cached[:4], full, strict=True):
        assert max(abs(a[k] - b[k]) for k in a) < 0.02
    assert all(abs(sum(p.values()) - 1) < 1e-6 for p in cached)
    assert sum(correct) >= 15, correct  # most hand-written items right
    assert sum(yes_no_correct) >= 3, questions
    assert mean_mass > 0.5  # the model answers with a letter
    assert peak_gb < 6.5  # within the Metal share of the 8 GiB budget
