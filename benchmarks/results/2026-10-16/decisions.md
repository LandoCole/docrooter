## Decisions

<!-- Links in this file are relative to its own directory; `python -m benchmarks.report`
rewrites them when it appends this file to docs/benchmarks.md. -->

The v0.1 spec wrote these rules down before any run
([section classification design, "Decision rules"](../../../docs/superpowers/specs/2026-10-15-docrooter-section-classification-design.md)).
Each rule is quoted verbatim, then applied to the results files above. Results that contradict
expectations are recorded as they are. Mean hF1 is over B1, B2 and the fixed B3 subset (the
*Summary* tables), unless stated otherwise.

### Default zero-shot backend

> **Default zero-shot backend:** highest mean hierarchical F1 across B1–B3, among backends with
> at least 20 sections per second on M1 MPS. Ties go to the smaller model.

Results: [zero-shot.json](zero-shot.json).

- **The throughput criterion was not measured.** Every zero-shot run in `zero-shot.json` ran on
  the CPU, not on M1 MPS, so the rule cannot be applied as written. On the CPU no backend reaches
  20 sections/s: the slowest dataset of each gives UltraBERT-Embed 3.4, bge-small 15.0,
  `ultrabert-mlm` 0.2 and `deberta-nli` 0.1 sections/s. The only zero-shot MPS measurement is
  `deberta-nli` on B2 in [strategies.json](strategies.json):
  0.2 (Beam(3)) and 0.4 (Greedy) sections/s.
- **Accuracy.** Mean hF1: UltraBERT-Embed 0.425, `deberta-nli` 0.314, bge-small 0.292;
  `ultrabert-mlm` has no B3 run, so no mean. `deberta-nli`'s mean mixes sample sizes: n=200 on
  B1 and B2, where the other two backends have n=1000, and n=300 on the B3 subset. On the full
  B3 (n=1000) UltraBERT-Embed scores 0.381 against bge-small's 0.337.
- **Ruling: UltraBERT-Embed (`embedding:dmis-lab/UltraBERT-Embed-150M`) stays the default.** It
  has the highest mean hF1 by 11 points. Its intended use, the document-type gate, classifies one
  view per document, so per-section throughput is not what limits it. bge-small
  (`embedding:BAAI/bge-small-en-v1.5`) is the speed alternative: about 4.4 times as fast on the
  CPU (15.0 against 3.4 sections/s) at a mean hF1 of 0.292.

### `mlm-choice` status

> **`mlm-choice` status:** ships non-experimental only if, on both B1 and B2, it is within 3
> hierarchical-F1 points of the best zero-shot backend.

Results: [zero-shot.json](zero-shot.json).

| Dataset | Best zero-shot backend | `ultrabert-mlm` | Gap |
| --- | --- | --- | --- |
| B1 | UltraBERT-Embed 0.752 (n=1000) | 0.435 (n=200) | 31.7 points |
| B2 | `deberta-nli` 0.272 (n=200) | 0.259 (n=200) | 1.3 points |

**Ruling: `mlm-choice` stays experimental.** It is within 3 points on B2 but 31.7 points behind
on B1, and the rule asks for both. (The B1 samples differ in size, 200 against 1000; a gap of
this size does not depend on that.) The B2 margin is fragile too: in
[strategies.json](strategies.json) `deberta-nli` with Beam(3) scores 0.292 on B2 with n=1000 (on
MPS), and against that figure `ultrabert-mlm` (0.259, n=200) is 3.3 points behind, outside the
margin.

### Default training base model

> **Default training base model:** highest hierarchical F1 at 16 examples per class, averaged
> over B1–B3.

Results: [few-shot.json](few-shot.json).

| Base model (trained heads, k=16) | B1 | B2 | B3 (n=1000) | Mean | Mean with the B3 subset |
| --- | --- | --- | --- | --- | --- |
| UltraBERT-Embed | 0.888 | 0.922 | 0.719 | 0.843 | 0.837 |
| bge-small | 0.864 | 0.854 | 0.673 | 0.797 | 0.787 |

- **What ran.** Trained heads on a frozen body ran on all three datasets at 4, 8 and 16 examples
  per class (on the CPU). The fine-tuned body ran only on B1 at k=16 (on MPS): UltraBERT-Embed
  0.936, bge-small 0.923. So the averaged rule is applied to the heads, and the fine-tuned B1
  pair agrees with it but cannot be averaged.
- **Ruling: UltraBERT-Embed stays the default training base** (`train(base_model=...)` defaults
  to `dmis-lab/UltraBERT-Embed-150M`): 0.843 against 0.797 averaged over B1-B3 at k=16, and
  ahead at every k and on every dataset.

### Default strategy

> **Default strategy:** Beam(3), unless Greedy is within 1 point of Beam(3) path accuracy on B2
> and at least 1.5× faster. In that case Greedy becomes the default.

Results: [strategies.json](strategies.json) (B2, n=1000, MPS).
Path accuracy is the *Exact* column.

| System | Beam(3) exact | Greedy exact | Greedy behind by | Greedy speed-up |
| --- | --- | --- | --- | --- |
| UltraBERT-Embed trained heads (k=16) | 0.881 | 0.770 | 11.1 points | 1.00× (7.8 against 7.8 sections/s) |
| `deberta-nli` zero-shot | 0.233 | 0.219 | 1.4 points | 1.63× (0.36 against 0.22 sections/s) |

**Ruling: Beam(3) stays the default.** Neither system meets the accuracy condition. For the
trained heads Greedy is also no faster; for the NLI backend it is fast enough but 1.4 points
behind. In hF1 the heads show the same gap (0.922 against 0.833), while NLI's two strategies tie
(0.292 against 0.291).
