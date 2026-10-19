# Element tags

[Elements](elements.md) say what a block **is**: a paragraph, a table, a footnote. Version 0.5
adds two facets that say what it is **about** and what it **does**. Both are optional: nothing
is tagged unless you set the rooter's `element_tags` task, and an `Element` without tags
serialises exactly as it did in v0.4.

| Facet | Question | Shape | Taxonomy |
| --- | --- | --- | --- |
| `topic` | What is this element about? | One label, from a tree | `element_topics` |
| `function` | What does this element do? | Zero or more labels, from a flat list | `element_functions` |

The use is selection and routing: read only the elements of a filing that describe fees, skip
the safe-harbour boilerplate, or send every `obligation` in a contract to a second step.

## The topic facet

`element_topics` is one merged tree, root, then a **family**, then a leaf. A family is a section
taxonomy that already ships with docrooter ([section normalisation](../guides/section-normalisation.md)
uses the same ones), plus an `other` leaf so that an element that fits nowhere has somewhere to go.

| Family | Leaves (besides `other`) |
| --- | --- |
| `annual_report` | `shareholder_letter`, `business_overview`, `risk_factors`, `management_discussion`, `financial_statements`, `notes_to_financial_statements`, `corporate_governance`, `executive_compensation`, `controls_and_procedures`, `legal_proceedings`, `properties`, `market_for_equity`, `exhibits`, `sustainability` |
| `fund_document` | `investment_objective`, `fees_and_expenses`, `principal_investment_strategies`, `principal_risks`, `performance`, `holdings_and_allocation`, `management`, `purchase_and_sale`, `tax_information` |
| `contract` | `parties_recitals`, `definitions`, `scope_services`, `term`, `payment`, `confidentiality`, `intellectual_property`, `warranties`, `indemnification`, `limitation_of_liability`, `termination`, `governing_law_disputes`, `general_provisions`, `signatures_exhibits` |
| `research_paper` | `abstract`, `introduction`, `related_work`, `background`, `method`, `experiments`, `results`, `discussion`, `limitations`, `conclusion`, `acknowledgements`, `references`, `appendix` |
| `technical_document` | `overview`, `installation_getting_started`, `configuration`, `usage`, `reference`, `specifications`, `troubleshooting`, `appendix` |
| `generic` | `front_matter`, `summary`, `introduction`, `body`, `conclusion`, `references`, `appendix` |

Leaf ids are prefixed by their family: `fund_document.fees_and_expenses`. Every node has a
definition, and `builtin_taxonomy("element_topics")` returns a fresh copy.

**The family comes from the document type.** The same map that picks a section taxonomy picks the
family: annual and quarterly reports and proxy statements use `annual_report`; prospectuses and
the new `financial.fact_sheet` type use `fund_document`; contracts `contract`; research papers
`research_paper`; user manuals and datasheets `technical_document`; every other type, and an
unknown one, `generic`. `Element.tags["topic"]` reports the best leaf **within that family**.

!!! note "Government, court and RFC documents"
    These keep the structural `generic` family, so their topic is mostly structural
    (`front_matter`, `body`, `references`). Expect their `function` tags to carry most of the
    value.

The student scores every element over the whole tree, then restricts the answer to the
document's family. The unconstrained family distribution is kept in `TagResult.family_probs`, so
a document type can be cross-checked against what its elements are about without that check being
circular.

## The function facet

`element_functions` is a flat list of twelve tags. An element carries zero or more of them. Each
tag has a definition, written examples and a "not this" clause.

| Tag | Meaning |
| --- | --- |
| `forward_looking` | Statements about expected future results, plans or outlook |
| `risk_disclosure` | Describes a risk, uncertainty or adverse scenario |
| `financial_data` | Reports quantitative financial figures: statement rows, KPIs, fees |
| `accounting_policy` | States how amounts are measured or recognised |
| `legal_boilerplate` | Standard legal text: disclaimers, safe-harbour, notices, copyright |
| `definition` | Defines a term |
| `obligation` | Imposes a duty or prohibition on a party: shall, must, may not |
| `procedure` | Step-by-step instructions or a process |
| `contact_or_admin` | Addresses, phone numbers, filing metadata, form headers |
| `signature` | Signature blocks, attestations |
| `citation` | Bibliography or reference entries, footnote citations |
| `marketing` | Promotional claims about products or performance |

## What an element carries

`Element.tags` is `None` by default and, when a tagger ran, a dictionary with the keys `topic`
and `function`. Each value is a `TagResult`:

- `labels`: for `topic`, the best leaf; for `function`, every tag over the threshold (possibly
  none);
- `probs`: a probability per label, calibrated when the tagger is;
- `confidence`: the tagger's confidence in `labels` as a whole;
- `source`: what produced the tags (a bundle, `embedding:<model>` or `teacher:<spec>`);
- `family_probs` (topic only): the unconstrained family distribution.

```json
{
  "topic": {
    "labels": ["fund_document.fees_and_expenses"],
    "probs": {"fund_document.fees_and_expenses": 0.0, "fund_document.other": 0.0},
    "confidence": 0.0,
    "source": "<bundle or spec>"
  },
  "function": {"labels": ["financial_data"], "probs": {"financial_data": 0.0}, "confidence": 0.0, "source": "<bundle or spec>"}
}
```

The zeros only show the shape of the record. Only paragraphs, list items, tables, captions,
footnotes and formulas are tagged; headings, titles, figures, contents and page furniture keep
`tags=None`.

## Three taggers, one setting

The `element_tags` setting of a [`RouterConfig`](rooter.md) takes one of four forms. It is `none`
in every profile.

| Value | What tags the elements | Needs |
| --- | --- | --- |
| `none` | Nothing | |
| a bundle directory | The trained student, loaded once per `process_many` | extra `models` |
| `embedding:<model>` | A zero-shot baseline: the leaf whose definition is nearest, and each function tag when the element is in the document's top fifth for it | extra `models` |
| `teacher:<engine spec>` | A teacher LLM answers for every element (13 forward passes each: slow, for checking the student) | `teach-mlx`, or `models`, or `llm` |

On the shell, `docrooter tag report.pdf --bundle BUNDLE` prints the tagged elements (add `--json`
for the full records), and `docrooter serve --element-tags BUNDLE` shows tags in the
[playground](../guides/playground.md). To make a bundle, see
[Train your own element tagger](../guides/train-your-tagger.md).

## How good is it?

The tagger is measured on an independently annotated gold set, never on teacher labels. The
numbers belong on one page, [Element tags (gold set)](../benchmarks/element-tags.md), which is
filled in once the trained student has been evaluated. Three things to know before you read any
of them:

- **The gold set is Claude-annotated.** Claude subagents wrote the labels from the taxonomy
  definitions alone, without seeing any model output. All of them share one underlying model, so
  the agreement between them (Cohen's kappa 0.953 for topics on the 180 double-annotated
  elements) measures consistency, not human-verified correctness.
- **Performance depends on corpus size.** A tagger trained on a few hundred elements is not a
  tagger trained on tens of thousands. The benchmark page has a learning curve for this.
- **The topic facet leans on the document type.** A wrong document type picks the wrong family.

## Next

- [Train your own element tagger](../guides/train-your-tagger.md): corpus, teacher, student.
- [Elements](elements.md) and [the rooter](rooter.md): where tags are produced.
- [Taxonomies](taxonomies.md): the other built-in taxonomies.
