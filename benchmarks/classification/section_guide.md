# Section annotation guide

You are building the evaluation-only gold set for section tags: for each sampled section of a
public document, one label from the section catalogue. The benchmark scores a classifier's
section tags against your labels, so a label is a judgement about what the section *is*, made
the same way every time. Read this guide first, then the catalogue (`catalogue.json`, next to
your batch files: every group and leaf with its description and examples).

## What you get, and what you hand back

Each item is a JSON line with `item` (an opaque id), `level` (1 or 2), `title`, `heading_path`
(the titles of the section's ancestors, outermost first, empty at the top) and `text` (the
opening of the section's own text, cut at 600 characters). You never see the document's name,
its type, or any model's guess, and you must not look for them.

Hand back one JSON line per item:

```json
{"item": "s0042", "label": "financial.income_statement", "annotator": "A1"}
```

The label is exactly one **leaf** id of the catalogue (`group.leaf`). A group id alone
(`financial`) is not a label. Label every item in your batch; do not skip, and do not give two
labels. If two leaves seem right, apply the rules below to pick one.

## How to judge

1. **Title first.** The title usually names the section. Use the catalogue's descriptions and
   examples; a title is matched by meaning, not by wording ("Our Approach" may be a method).
2. **Heading path second.** It says where the section sits. "Statement of Cash Flows" under
   "Notes to the financial statements" is a note about cash flows, not the statement.
3. **Opening text third.** Use it to settle what the title leaves open, and to tell a heading
   that introduces real content from one that does not.
4. **Label the section itself.** A level-1 heading such as "Part II" or "Financial section"
   that only holds other sections is labelled by what it names; if it names nothing in the
   catalogue, use the best-fitting `general` leaf (`general.body`). Do not label a section by
   what its subsections might contain, which you cannot see.
5. **Group first, then leaf.** Decide which group's domain the section belongs to, then which
   leaf. The groups are `general`, `research`, `narrative`, `financial`, `disclosure`, `esg`,
   `fund`, `offering`, `contract`, `law`, `technical` and `standard`.
6. **Do not infer the document type** to pick a label. A "Termination" heading is
   `contract.termination` whatever the document is; the catalogue is the same for all of them.

## The `other` leaves

Every group has an `other` leaf for content that is clearly in *that group's domain* but fits
no leaf of it (`general.other`, `research.other`, `narrative.other`, `financial.other`,
`disclosure.other`, `esg.other`, `fund.other`, `offering.other`, `contract.other`, `law.other`,
`technical.other`, `standard.other`).

- Use `<group>.other` only when you can name the group and no leaf fits. It is not a place for
  "I am not sure": first try the other rules below.
- `general.other` is for brief or administrative content that belongs to no domain (contact
  details, author biographies, disclaimers, colophons, a fragment of a heading with no
  content). Substantive main text that fits no domain leaf is `general.body`, not
  `general.other`.
- A section about a group's domain but with a more specific leaf elsewhere goes to that leaf.
  The catalogue descriptions say where such content belongs; follow them.

## Rules for confusable pairs

Apply the first rule that fits.

**Financial statements, notes and narrative**

- The statement itself versus its note: a section that *presents* the statement's line items
  for a period or date is the statement (`financial.income_statement`,
  `financial.balance_sheet`, `financial.cash_flow_statement`,
  `financial.changes_in_equity`). A section that *explains* accounting policies, estimates or a
  line item, usually numbered ("Note 7. Leases"), is `financial.notes_to_financial_statements`.
  A heading that is a statement's name but sits inside the notes is a note.
- Income statement versus MD&A tables: a table of results in the audited statements is
  `financial.income_statement`. Management's explanation of the same figures (why revenue
  moved, segment results, liquidity and capital resources, market risk) is
  `narrative.management_discussion`, even when it quotes tables.
- The set versus one statement: a section headed "Financial statements" (Item 8) that holds
  the statements together is `financial.financial_statements`; a section that holds a single
  statement is that statement's leaf.
- Highlights versus statements: a short table of headline figures, "selected financial data"
  or "key figures" is `financial.financial_highlights`; the full statements are not.
- Supporting financial data that is not a statement or a note (non-GAAP reconciliations,
  segment tables, supplementary unaudited data) is `financial.other`.
- Auditor versus management: the independent auditor's opinion is `financial.auditors_report`;
  management's own report on internal control is `disclosure.controls_and_procedures`.
- A fund's schedule of investments (every position, with cost and fair value) is
  `financial.schedule_of_investments`; a summary of top holdings or allocation by sector is
  `fund.holdings_and_allocation`.
- A customer's own account statement (bank, brokerage, insurance) is `financial.account_statement`,
  not a company's financial statements.

**Narrative versus disclosure versus offering**

- A letter addressed to shareholders or investors, signed by a chairman, CEO or fund manager,
  is `narrative.shareholder_letter`, even if it discusses results and outlook.
- What the company does (products, markets, strategy; Item 1) is
  `narrative.business_overview`; the guidance and expectations for the coming period are
  `narrative.outlook`; a review of the period just ended is `narrative.management_discussion`.
- Risks: a discussion of risks to the business, a fund or a security is
  `disclosure.risk_factors`. Market-risk disclosures inside MD&A (interest-rate and currency
  sensitivity tables, Item 7A) are `narrative.management_discussion`.
- A safe-harbor or "forward-looking statements" legend is
  `disclosure.forward_looking_statements`; the forward-looking *content* is `narrative.outlook`.
- An offering document's description of the securities and the offering is
  `offering.offering_terms`; the issuer's own business is `narrative.business_overview`.

**General versus specific**

- `general.summary` versus `research.abstract`: a paper's abstract is `research.abstract`;
  an executive summary or overview of any other document is `general.summary`.
- `general.introduction` versus `general.background`: the opening that states purpose and
  scope, or motivates a paper, is the introduction; history, definitions and preliminaries the
  reader needs first are the background. A court's recital of the facts is `law.opinion`.
- `general.findings` versus `general.recommendations` versus `general.conclusion`: what was
  found is findings; what the authors recommend someone do is recommendations; the wrap-up of
  the document is the conclusion. A paper's measured numbers are `research.results`.
- `general.references` versus `standard.normative_references`: a bibliography or list of
  citations is `general.references`; a standard's lists of normative and informative
  references are `standard.normative_references`.
- `general.appendix` versus `technical.reference`: supplementary material after the body
  (extra tables, proofs, glossaries, revision history) is `general.appendix`; lookup material
  that is the *point* of the document (API, command, register descriptions) is
  `technical.reference`.
- `technical.overview` versus `general.summary` and `general.introduction`: an overview of a
  product or system (what it is, its features, intended users) is `technical.overview`; a
  summary of the document's own points is `general.summary`; the opening that states the
  document's purpose and scope is `general.introduction`.
- `general.acknowledgements` is thanks, funding statements and author contributions; ethics and
  data-availability statements are `research.other`; author biographies are `general.other`.
- `general.front_matter`: table of contents, title page, preface, lists of figures and
  copyright notices. A fund document's or filing's cover page also goes here.

**Research**

- `research.method` (how the approach works or the study was designed) versus
  `research.experiments` (datasets, baselines, metrics, protocol) versus `research.results`
  (the measurements) versus `research.discussion` (what the measurements mean).
- `research.related_work` is a survey of prior work against this paper;
  `general.background` is context the reader needs, not a comparison of competitors.
- `research.limitations` is an explicit statement of weaknesses; a conclusion that mentions
  limitations in passing is `general.conclusion`.

**Fund documents**

- `fund.investment_objective` (the goal) versus `fund.principal_investment_strategies` (how it
  pursues the goal) versus `disclosure.risk_factors` (what could go wrong).
- `fund.fees_and_expenses` covers costs of buying, holding and selling shares, including the
  expense example and turnover; `fund.performance` covers returns; `fund.fund_facts` covers
  identifiers, inception date and net assets.
- `fund.management` is who runs the fund; the fees paid to them belong to
  `fund.fees_and_expenses`.
- `fund.purchase_and_sale` is how to buy or redeem; the charges on those belong to
  `fund.fees_and_expenses`; `fund.tax_information` is how distributions are taxed.

**Contracts and policies**

- `contract.definitions` (an agreement's defined terms) versus `standard.terminology` (a
  standard's terms and conventions).
- `contract.confidentiality` (protecting the parties' information) versus
  `contract.privacy_data` (collecting and protecting personal data, as in a privacy policy).
- `contract.acceptable_use` (what users may not do with a published service) versus
  `contract.scope_services` (the work a party owes a named counterparty).
- Boilerplate (notices, assignment, amendment, severability, entire agreement) is
  `contract.general_provisions`; non-competition, insurance and audit rights are
  `contract.other`.
- `contract.signatures_exhibits` is the signature blocks and attached schedules of an
  agreement; a filing's list of exhibits is `disclosure.exhibits`.

**Law**

- The court's own reasoning and disposition is `law.opinion`; a separate concurrence or dissent
  is `law.separate_opinion`; the reporter's headnote is `law.syllabus`; the docket block is
  `law.caption`.
- The operative rules of a statute or regulation are `law.provisions`; changes to other laws
  are `law.amendments`; titles, enacting clauses and statements of purpose are `law.preamble`.

**Technical documents and standards**

- `technical.specifications` (quantitative data in tables), `technical.interfaces` (pins,
  ports, signals), `technical.package_mechanical` (outline drawings, footprints, marking) and
  `technical.ordering_information` (part numbers and variants) are four different leaves.
- `technical.usage` (how to operate in normal use) versus `technical.installation_getting_started`
  (set-up and first run) versus `technical.configuration` (settings and options) versus
  `technical.troubleshooting` (diagnosing problems and FAQs).
- `technical.safety` is safety warnings and compliance notices; a standard's security analysis
  is `standard.security_considerations`.
- `standard.specification` (the normative body for implementers) versus `technical.reference`
  (a vendor's user-facing API reference) versus `standard.conformance` (what an implementation
  must meet to claim conformance).
- `standard.scope` (what the standard covers) versus `general.introduction` (motivation).

**ESG**

- `esg.sustainability` is the overall approach when no single topic is the subject;
  `esg.climate_environment` and `esg.social` are topics; board governance is
  `disclosure.corporate_governance`; data tables and framework indexes (GRI, SASB, TCFD) are
  `esg.esg_metrics`.

## Awkward cases

- **Not a real heading.** The reading picks up some headings that are not (a running page
  header, "Table 3", a figure caption). Label such an item by the content it heads, as if it
  were a heading. If there is no content to judge, use `general.other`.
- **Numbering and prefixes.** Ignore "1.", "Item 7." and "Article 4" for wording, but use item
  numbers you recognise from the title (Item 1A, Item 7, Item 9A) to confirm a label.
- **Empty or very short text.** Judge by the title and the heading path.
- **Other languages.** Label by meaning as for English.
- **Genuinely ambiguous.** Choose the leaf the catalogue's descriptions put first, apply the
  rules above, and move on. Do not leave an item unlabelled.

## Working rules

- Work blind: do not discuss items with other annotators, and do not look at any other
  annotator's labels or any model's output.
- Do not look up the document. Judge from what the item shows.
- Label at the leaf; the benchmark scores both the leaf and its group.
- Double-annotated items are labelled independently by a second annotator; disagreements go to
  a third annotator, who sees the two labels but not the annotators' names.
