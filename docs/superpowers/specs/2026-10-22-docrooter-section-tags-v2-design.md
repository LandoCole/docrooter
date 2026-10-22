# doc-rooter section tags v2: tags that normalise meaning, on Jev

2026-10-22. Asked for by the user after reviewing the demo:

- Switch the demo and benchmark to Jev.
- Rebuild the section tags from the ground up.
- Redesign the demo around what the model saw.

## Why

v1's section-tag F1 was low: 0.56 leaf macro F1 for Haiku, sections first, at headings. The
breakdown (951 gold sections, Haiku, headings) showed three causes:

1. **Sections never sent.** Only level-1 and level-2 headings were tagged. The PyMuPDF4LLM
   reader often nests "Balance sheet" or "Notes" at level 3 to 6, so they got no tag. 151 of
   the 161 untagged gold sections in one run were at level 3 or deeper.
2. **Gold drift.** The gold was sampled from the removed heuristic reader. 36% of it (339
   sections) no longer matches a section of the new reading and can't be scored.
3. **Generic tags.** `general.*` (summary, introduction, background, body, findings,
   conclusion…) made up a third of the gold. Those labels are fuzzy by nature: tagged sections
   were only 55% right, generic or not. The user's point is that they also aren't useful.

The user's definition of a section tag: **it normalises what a heading means, so the same
kind of section can be found across documents however each one words it.** "Statement of
financial position" and "Consolidated balance sheets" are both `financial.balance_sheet`.
"Report of independent registered public accounting firm" and "Audit opinion" are both
`financial.auditors_report`. Generic structure (summary, background, conclusion) isn't
something to look for across documents, so it gets no tag.

## Decisions

### T1. The catalogue (`section_catalogue.json`, v2)

- **`none` replaces the `general` group.** It is a top-level leaf, "No tag". It covers
  generic structure (front matter, contents, introduction, summary, background, body
  chapters, conclusion, acknowledgements, appendix) and headings that only number or title
  something. Tagging `none` is a correct answer, not a failure. Anyone who wants structural
  tags can pass their own catalogue (`catalogue=`); the guide shows how.
- **A new `government` group** for audit and oversight reports (GAO, inspectors general,
  agency audits). Their sections have stable meanings under varying titles:
  - highlights ("What GAO found", "Results in brief");
  - objectives, scope and methodology;
  - findings;
  - recommendations ("Recommendations for executive action", "Matters for congressional
    consideration");
  - agency comments.
- **Research** gains `research.references`: references and bibliography are something to look
  for. Its introduction and conclusion stay `none`.
- **Financial** gains `financial.comprehensive_income`. Its examples spell out the synonyms
  the user listed: statement of financial position, condition or net assets for the balance
  sheet; statement of operations, profit or loss, or activities for the income statement;
  audit opinion for the auditor's report.
- The `<group>.other` leaves stay. "Financial content, no more specific tag" is still a
  normalised meaning.
- **Doc-type rules:**
  - `government.government_report` requires a `government` tag.
  - `research.white_paper` loses its rule, which required generic tags; the model decides it.

### T2. Tag every heading

- `select_sections` takes every titled section at any depth, level 1 first, then level 2, and
  so on, in reading order, up to `max_sections` (raised from 40 to 400).
- The section view already includes the heading path, so "12 Leases" under "Notes to the
  financial statements" reads as a note.
- `none` tags are not evidence:
  - the rule gate's share counts domain tags only;
  - rules never match `none`.
- **doc_first** tags against the same catalogue as sections_first. The flows differ only in
  how the type is decided, which is what the demo explains. The per-type section taxonomies
  stay for `normalize_sections`.

### T3. Gold v2

- Sampled from the PyMuPDF4LLM reading (the reading cache) at every depth, up to 30 headings
  per document, spread across depths and reading order.
- Annotated with guide v2 by Claude annotators, evaluation only:
  - one full pass;
  - 20% double-annotated, for Cohen's kappa;
  - an adjudication of every disagreement.
- The v1 gold stays in git history; v2 replaces the files.

### T4. Scoring

- A tag of `none` abstains:
  - **precision** is the share of domain tags given that were right;
  - **recall** is the share of gold domain sections tagged right;
  - **macro F1** is over domain leaves.
- Agreement on `none` is reported on its own.
- Group-level F1 stays.

### T5. Jev everywhere

- Jev (`jev-1.13.0`) reports exact tokens per request (the backend's `usage()`).
- The benchmark grid:
  - both flows at the four rungs, per item, on the 113 gold documents;
  - the held-out doc type at headings, terms and lead.
- The README and demo lead with Jev. Haiku and UltraBERT stay as comparison rows.
- The demo's live upload calls Jev (`TYPESAFE_API_KEY` in Vercel).

### T6. Demo

- **The model's input is the main panel**, at the chosen rung. Each heading in it is marked
  with its tag (and the gold tag, on samples). This replaces the separate section-tag list.
- **A per-document comparison of the rungs:** doc type right or wrong, tags right out of the
  gold ones, tokens and cost for each rung, so one document shows what each rung bought.
- **Every doc-type option with Jev's probability,** as the document classifier shows it.
- **The flows are explained in plain words:** "Read the document, then tag" against "Tag the
  sections, then decide".
- doc-rooter is introduced as the next step after the document classifier.
