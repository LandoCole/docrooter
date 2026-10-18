# Element extraction, routing, tagging and the completeness dial: research brief

Date: 2026-10-18. Scope: the next docrooter programme, covering (a) element extraction, (b) the rooter, (c) TELEClass-style element tagging and (d) the completeness dial.
Constraints: Apple M1, 8 GB per model process, torch-free core, Apache-2.0 package.

Conventions:
- **[unverified]** means I could not confirm the point from a primary source. Treat it as a lead.
- **[estimate]** means the number is my own arithmetic or engineering judgement, not a published figure.
- All licence statements were read from the project's own README, LICENSE or model card unless marked otherwise. Re-check them at pin time, because licences change (for example, Marker's code licence and PyMuPDF Layout's licence both changed during 2025-26).

---

## 0. Top-line recommendations

1. **Element extraction.**
   - **Default (torch-free):** pdfplumber (MIT) plus our own heuristics, with recursive XY-cut for reading order.
   - **ML layout extra:** Docling's `heron` layout detector via its official ONNX export on onnxruntime. It is Apache-2.0, has 17 classes and needs no torch.
   - **Avoid in defaults:** DocLayout-YOLO (AGPL), Surya/Marker weights (revenue-capped licence) and PyMuPDF Layout (AGPL or PolyForm-NC).
2. **OCR.**
   - **macOS default:** `ocrmac` (Apple Vision, MIT wrapper).
   - **Everywhere else, and the macOS fallback:** RapidOCR (Apache-2.0, onnxruntime, pip-only).
   - **Tesseract:** an optional extra that needs a system binary.
   - Do not ship EasyOCR or Paddle as defaults.
3. **The Jev Master Customer Agreement forbids using Jev Output for model distillation (§2.3(b)).** Jev cannot be the TELEClass teacher. The teacher should be a permissively licensed open-weight LLM, or an API LLM whose terms allow the use.
4. **The dial should use conformal risk control (CRC) with a per-document miss-rate loss.** It gives a finite-sample guarantee on expected recall, and needs a few hundred calibrated documents per corpus or document type. Learn-then-Test (LTT) is the variant for a high-probability guarantee. The exact algorithm is in §6.
5. **Student encoder.**
   - **Primary:** UltraBERT-150M (Apache-2.0, 32K context, GLUE 89.5 reported by its authors).
   - **Fallbacks:** ModernBERT-base and Ettin-150M.
   - **Fit on 8 GB:** the model fits for fine-tuning at seq 512 with bf16/fp16 autocast and a small batch, or with gradient checkpointing [estimate].
6. **Corpus.**
   - **Native PDFs:** EDGAR Form ARS ("glossy" annual reports, required as PDF on EDGAR since 11 Jan 2023), plus govinfo, Federal Reserve Board, RFC and arXiv PDFs.
   - **HTML filings:** turn 10-K/10-Q and fund filings into PDFs with headless Chromium, which also yields DOM-derived weak labels.
   - **Licences:** keep training and evaluation datasets with non-commercial licences (LEDGAR, FinanceBench, OmniDocBench) out of anything shipped.

---

## 1. Element and layout detection

### 1.1 Options

| Tool | Licence (code / weights) | What it gives | Torch? | Notes |
|---|---|---|---|---|
| **PyMuPDF** `get_text("dict")` | AGPL-3.0 or Artifex commercial (already an optional extra) | Blocks: type 0 text, type 1 image, type 3 vector. Spans carry font, size, flags (bold/italic/mono/super) and bbox. `sort=True` gives a simple vertical-then-horizontal order. Image blocks need `TEXT_PRESERVE_IMAGES`; vector blocks need `TEXT_COLLECT_VECTORS`. | No | [docs](https://pymupdf.readthedocs.io/en/latest/textpage.html) |
| PyMuPDF `find_tables` | same | Default "lines" strategy (vector rulings); "text" strategy for unruled tables, which is prone to false positives on prose | No | [Artifex blog](https://artifex.com/blog/solving-common-issues-with-table-detection-and-extraction), [forum](https://forum.mupdf.com/t/fitz-page-find-tables-misses-last-column-when-using-horizontal-strategy-text/57) |
| **PyMuPDF Layout** (`pymupdf-layout`, used by default in pymupdf4llm) | PyPI 1.28.2 (6 Aug 2026) says **AGPL-3.0 or Artifex commercial**. Earlier listings said **PolyForm Noncommercial or commercial**. | A GNN over PDF internals (not page images), served on onnxruntime. Gives titles, headings, headers/footers, tables, images. Claims "~10×" the speed of vision tools on CPU. | No | [PyPI](https://pypi.org/project/pymupdf-layout/), [Artifex blog](https://artifex.com/blog/pymupdf-layout-10-faster-pdf-parsing-without-gpus), [pymupdf4llm docs](https://pymupdf.readthedocs.io/en/latest/pymupdf-layout/index.html), [AUR licence field](https://aur.archlinux.org/packages/python-pymupdf-layout-bin). The licence history conflicts **[unverified which applies to older versions]**. Treat it as copyleft or NC, so it is not suitable for defaults. |
| **pdfplumber** | MIT | Char/word/rect/line/curve objects; tables via `lines`, `lines_strict`, `text` and `explicit` strategies; experimental `layout=True` text | No | [README](https://github.com/jsvine/pdfplumber). The README itself says PyMuPDF is "substantially faster". |
| **Docling** | MIT (code) | Full pipeline: layout model, TableFormer (FAST / ACCURATE modes), OCR plug-ins (EasyOCR, ocrmac, RapidOCR, Tesseract) | **Yes** (full pipeline) | [tech report](https://arxiv.org/html/2408.09869v5), [pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/) |
| Docling **heron** layout model | **Apache-2.0** | RT-DETRv2-R50, **42.9M params**, 17 classes: Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, Document Index, Code, Checkbox ×2, Form, Key-Value Region | Official **ONNX export** exists ([docling-layout-heron-onnx](https://huggingface.co/docling-project/docling-layout-heron-onnx), Apache-2.0, ~169 MB FP32 per a [mirror](https://huggingface.co/xberg-io/layout-models)) | [model card](https://huggingface.co/ds4sd/docling-layout-heron) |
| Docling egret-m/l/x | permissive (per paper) | D-FINE backbones, 19.5–62.7M params | via export | [arXiv 2509.11720](https://arxiv.org/html/2509.11720v1) |
| **DocLayout-YOLO** | **AGPL-3.0** code (builds on Ultralytics). Weights are listed as AGPL on HF mirrors; one mirror claims Apache **[unverified]**. | YOLOv10 layout detector. DocLayNet AP50 93.4 / mAP 79.7 with DocSynth300K pre-training. | Yes (pip package) | [repo](https://github.com/opendatalab/DocLayout-YOLO), [HF mirror](https://huggingface.co/anonymous-iccv1968/DocLayout-YOLO-DocStructBench/blob/main/README.md). **Do not ship as a default.** |
| **Surya / Marker** | Code Apache-2.0. **Weights: modified AI Pubs OpenRAIL-M**, "free for research, personal use, and startups under $5M funding/revenue". | Layout plus reading order, OCR, tables (Surya); full conversion (Marker) | Yes | [surya](https://github.com/datalab-to/surya), [marker](https://github.com/datalab-to/marker). The revenue/funding cap makes the weights unsuitable for an Apache default. Optional, user-installed, with a loud warning at most. |
| **PP-DocLayout** (Paddle) | **Apache-2.0** (HF card) | 23 classes, including paragraph title, footnote, header/footer, figure/table caption, aside text and references. -L is RT-DETR-L (paper: 90.4 mAP@0.5). -S is listed at 14.5 ms/page on CPU in the paper. | ONNX via RapidLayout | [HF card](https://huggingface.co/PaddlePaddle/PP-DocLayout-L), [paper](https://arxiv.org/html/2503.17213v1), [PaddleX timings](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html) |
| RapidLayout | Apache-2.0 code; weights vary by model | onnxruntime runner for PP-layout, YOLOv8-layout and DocLayout-YOLO models | No | [repo](https://github.com/RapidAI/RapidLayout). Its DocLayout-YOLO weights carry the AGPL question above. |
| unstructured | Apache-2.0 (library) | Strategies `fast`, `hi_res` (model-based: `yolox`, `detectron2_onnx`), `ocr_only` and `vlm` | heavy deps | [docs](https://docs.unstructured.io/open-source/concepts/partitioning-strategies). Licence of the yolox weights **[unverified]**. |
| layoutparser | Apache-2.0 | Detectron2 model zoo | Yes (detectron2) | Last release 0.3.4, "4 years ago" per [Cloudsmith](https://cloudsmith.com/navigator/pypi/layoutparser). Effectively unmaintained: skip. |

**Speed and memory figures from primary sources:**
- **Heron, per image:**
  - CPU (AMD EPYC 7763, 4 threads, batch 32): 0.643 s.
  - Apple M3 Max on MPS: 0.044 s.
  - egret-m on CPU: 0.334 s, at slightly lower mAP (0.765 vs 0.776 canonical DocLayNet).
  - Source: [arXiv 2509.11720](https://arxiv.org/html/2509.11720v1).
- **Full Docling pipeline, OCR off** ([tech report Table 1](https://arxiv.org/html/2408.09869v5)):
  - M3 Max, 4 threads: 1.27 pages/s with peak memory 6.2 GB (native backend), or 2.18 pages/s with 2.56 GB (pypdfium backend).
  - TableFormer takes 2–6 s per table on CPU.
  - The 6.2 GB native-backend peak is too close to our 8 GB cap.
- **M1:** no published M1 numbers. Expect slower than M3 Max **[unverified]**.

### 1.2 Reading order
- **Recursive XY-cut** splits on the widest whitespace valley in the projection profile: vertical gutters give columns, horizontal gaps give bands ([PDF Oxide guide](https://pdf.oxide.fyi/docs/guides/reading-order)).
  - It fails when a wide element, such as a banner or a full-width table, crosses every candidate cut ([arXiv 2607.01018](https://arxiv.org/pdf/2607.01018)).
- On a 140-page multi-column OmniDocBench subset, the same paper reports:
  - XY-cut (PP-StructureV3): 75% macro edge accuracy.
  - LayoutReader: 25%, blamed on a word-level vs line-level mismatch.
  - Their own method: 88%.
  - This is a single recent paper **[unverified replication]**.
- **LayoutReader / ReadingBank** (500k pages, seq2seq) ([MSR](https://microsoft.com/en-us/research/?p=770647)) needs torch and transfers poorly to line-level input.
- **Practical recipe [estimate]:**
  1. Remove headers and footers first (see below).
  2. Pull out full-width elements (title, wide tables or figures) as band separators.
  3. Run XY-cut inside each band, using a minimum gutter of about 1.5× the median char width × 3 and requiring the gutter to be at least 60% of the band height.
  4. Order cells top-to-bottom, then left-to-right.
  5. Attach captions to the nearest figure or table in the reading direction.
  6. If the layout model is on, use its boxes as the units for the cut.

### 1.3 Recommendation
- **Default (torch-free, MIT):** pdfplumber objects (PyMuPDF when the AGPL extra is installed), with rule-based element typing:
  - **Paragraphs:** line clustering by leading and indent, merged across column and page breaks when the last line has no terminal punctuation.
  - **Lists:** bullet and enumerator regex plus hanging indent.
  - **Headers and footers:** text in the top or bottom ~8% band that repeats on at least 50% of pages after digit normalisation.
  - **Footnotes:** smaller font, bottom of page, and a superscript marker that matches a reference in the body (span flag bit 0 is superscript in PyMuPDF).
  - **Captions:** `^(Figure|Fig\.|Table|Exhibit|Chart)\s*\d+` adjacent to an image or table.
  - **Tables:** `lines` strategy only by default; `text` strategy only inside regions flagged by the ML extra.
  - **Figures:** image objects, or vector-graphics clusters with sparse text (likely charts).
  - All thresholds above are my starting points **[estimate]**. Tune them on DocLayNet (§3).
- **ML extra (`docrooter[layout]`, torch-free):** heron ONNX on onnxruntime.
  - **Why:** Apache-2.0, an official artefact, a class set that matches our element list almost one-to-one, and 0.6 s/page on CPU.
  - **Model choice:** offer egret-m as a "fast" option if an ONNX export is published **[unverified: no egret ONNX found]**.
  - **Quantisation:** a backbone-only INT8 quantisation reportedly gives 2.6–3× on AVX-VNNI CPUs ([community card](https://huggingface.co/stefanj0/docling-layout-heron-int8-onnx)) **[unverified on ARM]**.
  - **Running it:** render pages at 640 px with pypdfium2 (no torch needed), detect, then fill each detected box with text-layer characters.
- **Heavy extra (`docrooter[docling]`):** full Docling, for users who want TableFormer cell structure. Document its ~2.5–6 GB peak and run it in its own process.
- **Do not depend on:** DocLayout-YOLO, Surya/Marker weights, PyMuPDF Layout or layoutparser.

---

## 2. OCR engines and the "needs OCR?" decision

### 2.1 Engines

| Engine | Licence | Install | Apple silicon | Speed / accuracy notes |
|---|---|---|---|---|
| **RapidOCR** 3.10.0 (8 Oct 2026) | Apache-2.0. Models derived from PaddleOCR and redistributed under the same licence. | `pip install rapidocr onnxruntime`; 27 MB wheel. Pip-only. | onnxruntime arm64 wheels; CoreML EP possible **[unverified for RapidOCR]** | Engines: ORT, OpenVINO, MNN, Paddle, TensorRT, torch. No official benchmark numbers. 75% item recall on one tiny 24-item invoice test ([codesota](https://codesota.com/ocr/best-for-python)) **[weak evidence]**. Sources: [repo](https://github.com/RapidAI/RapidOCR), [PyPI](https://pypi.org/project/rapidocr/) |
| **ocrmac** (Apple Vision / LiveText) | MIT wrapper; OS framework | `pip install ocrmac` (pyobjc). No model download. macOS 10.15+; LiveText needs Sonoma+. | Native (Neural Engine) | M3 Max: about 207 ms "accurate", 131 ms "fast", 174 ms LiveText per call. LiveText always reports confidence 1. Source: [repo](https://github.com/straussmaximilian/ocrmac). No independent CER benchmark found **[unverified accuracy]**. |
| **Tesseract** 5.5.1 + pytesseract | Apache-2.0 (binary). pytesseract is Apache-2.0 from 0.3.1 (older releases GPL-3). | **System binary** (`brew install tesseract`) | yes (Homebrew arm64) | Last of 5 engines on SROIE receipts (CER 33.6%), [vendor benchmark](https://imagetotable.ai/references/receipt-ocr-benchmark-results) **[vendor, receipts only]**. Sources: [Homebrew](https://formulae.brew.sh/formula/tesseract), [pytesseract PyPI](https://pypi.org/project/pytesseract/0.3.10/) |
| **EasyOCR** 1.7.2 (Sep 2024) | Apache-2.0 | torch + torchvision | no MPS mention | 80+ languages. Issues older than 6 months auto-closed; latest release is from 2024. Source: [repo](https://github.com/JaidedAI/EasyOCR). **Skip.** |
| **PaddleOCR** 3.7 (Jun 2026) | Apache-2.0 | paddlepaddle (or ONNX/Transformers backends since 3.5) | macOS badge; "6.1× on Apple M4 (tiny)" claim | PP-OCRv5/v6 and PP-StructureV3. Heavy framework. Source: [repo](https://github.com/PaddlePaddle/PaddleOCR). Use its models via RapidOCR instead. |
| **docTR** | Apache-2.0 | torch | no MPS mention | Source: [repo](https://github.com/mindee/doctr). Its ORT port **OnnxTR** (Apache-2.0, Python ≥3.11, no torch, CoreML EP auto-detected on Apple silicon) is about 2× faster than docTR on CPU (0.57 vs 1.29 s/page on FUNSD, i7-14700K) ([OnnxTR](https://github.com/felixdittrich92/OnnxTR)). Best-ranked on the receipt benchmark above (CER 19.7%). |

**Recommendation:**
- **macOS:** default to `ocrmac` with `accurate`. It needs no download and uses the Neural Engine. Fall back to RapidOCR.
- **Linux and Windows:** default to RapidOCR, with OnnxTR as an optional "accurate" engine. Tesseract is an optional extra for users who already have the binary or need its languages.
- **Engine abstraction:** all engines sit behind one `OcrEngine` protocol that returns line boxes, text and confidence. That matches the Docling engine set (`AUTO`, `EASYOCR`, `OCRMAC`, `RAPIDOCR`, `TESSERACT`), which makes later interop easy.
- **Benchmark gap:** no head-to-head, document-level CER benchmark covering Apple Vision was found. Build a small one from DocLayNet page renders: text-layer ground truth vs OCR output **[todo]**.

### 2.2 Deciding a page (or region) needs OCR
Prior art:
- **Docling:**
  - Older versions OCR the whole page when bitmap coverage exceeds `max(0.75, bitmap_area_threshold)` (default threshold 0.05). Between the threshold and that cutoff they OCR only the bitmap regions; below the threshold they skip OCR ([older source copy](https://huggingface.co/spaces/Mqleet/AutoPage/blob/main/docling/models/base_ocr_model.py), [defaults](https://aidoczh.com/docling/reference/pipeline_options/index.html)).
  - Current `main` is layout-aware. It OCRs a layout cluster only when the cluster overlaps a bitmap and has no visible programmatic text. It sets invisible/clipped text cells aside and merges cells PDF-first ([source](https://raw.githubusercontent.com/docling-project/docling/main/docling/models/base_ocr_model.py)).
- **pdf-inspector (Firecrawl):** flags garbage text from:
  - U+FFFD replacement characters;
  - broken-CMap `$`-for-space patterns;
  - letter-frequency deviation from English;
  - a non-alphanumeric share above 50% (minimum 50 chars).
  - It also exposes a per-region `needs_ocr` flag ([walkthrough](https://instagit.com/firecrawl/pdf-inspector/text-quality-analysis-garbage-text-detection/)). This is a secondary source **[unverified against repo]**.
- **OCRmyPDF:** `--skip-text` leaves pages with text alone. `--redo-ocr` strips invisible (non-printing) OCR text and re-OCRs. It cannot always tell a hidden OCR layer from printable text ([man page](https://manpages.ubuntu.com/manpages/lunar/man1/ocrmypdf.1.html), [errors doc](https://github.com/jbarlow83/OCRmyPDF/blob/v15.4.1/docs/errors.rst)).

**Proposed rooter rule.** All thresholds are tunable defaults **[estimate]**.

For each page, compute:
- `char_density`: printable chars per in² of page area. A dense text page is roughly 15–30 chars/in².
- `image_cov`: the union area of image objects divided by page area.
- `garbled`: the share of chars that are U+FFFD, pdfminer `(cid:N)`, Private Use Area (U+E000–F8FF) or C0/C1 control characters. Also flag the page if more than 50% of 50+ chars are non-alphanumeric, or if the share of tokens that are dictionary words or numbers falls below about 0.5.
- `invisible_text`: text with render mode 3 (an existing OCR layer).

Decide:
- **Whole-page OCR** if:
  - `image_cov ≥ 0.75` and `char_density < 2`; or
  - `garbled > 0.10`; or
  - the page has an `invisible_text` layer that fails the garble checks.
- **Region OCR** for each image region of at least 5% of the page area that contains no text-layer chars. This catches scanned exhibits and charts with baked-in labels.
- **Otherwise**, use the text layer.

Log every decision as an element attribute, so the dial and evaluation can audit routing.

---

## 3. Datasets for element-level training and evaluation

| Dataset | Licence (commercial / redistribution) | Size | Labels | Use for us |
|---|---|---|---|---|
| **DocLayNet** | **CDLA-Permissive-1.0** (commercial OK, redistribution OK with licence) | 80,863 pages; 6 categories (financial reports, manuals, scientific articles, laws/regulations, tenders, patents) | 11 layout classes (box level) | Main layout evaluation, and tuning the heuristics. [repo](https://github.com/DS4SD/DocLayNet), [LICENSE](https://raw.githubusercontent.com/DS4SD/DocLayNet/main/LICENSE) |
| **PubLayNet** | Annotations CDLA-Permissive-1.0; images under PMC OA commercial-use subset terms | ~360k pages per the paper (arXiv 1908.07836) **[size unverified here]** | text, title, list, table, figure | Scientific-only layout check. [repo](https://github.com/ibm-aur-nlp/PubLayNet), [LICENSE](https://raw.githubusercontent.com/ibm-aur-nlp/PubLayNet/master/LICENSE.md) |
| **OmniDocBench** | Code Apache-2.0; **dataset "research purposes only and not for commercial use"** | 1,651 pages, 9–10 document types | 28 block + 4 span types; **reading order**; table HTML/LaTeX | Internal reading-order and multi-column evaluation only. [repo](https://github.com/opendatalab/OmniDocBench) |
| **CUAD** | CC BY 4.0 | 510 contracts, 13k+ labels | 41 clause types (span level) | Legal clause tagging, training and evaluation. [site](https://www.atticusprojectai.org/cuad) |
| **LEDGAR** | **CC BY-NC 4.0** (paper/ZHAW). LexGLUE's HF metadata says cc-by-4.0, which conflicts. | LexGLUE version: 80k provisions, 100 classes. Original: ~100k provisions, 12k+ labels, 60k+ contracts. | Provision type (paragraph level) | Evaluation only. Assume NC. [LexGLUE](https://huggingface.co/datasets/coastalcph/lex_glue), [ZHAW record](https://srv-clst-301-data66.zhaw.ch/handle/11475/20087), [ACL](https://aclanthology.org/2020.lrec-1.155) |
| **ContractNLI** | CC BY 4.0 | 607 NDAs × 17 hypotheses | Entail / contradict / not-mentioned + evidence spans | Evidence-recall evaluation for the dial. [site](https://stanfordnlp.github.io/contract-nli/) |
| **EDGAR-CORPUS** | Apache-2.0 (dataset card); underlying filings public | 220k 10-K annual reports, 1993–2020, 40.7 GB | Text split into Items 1, 1A, 1B, 2…15 | **Free weak paragraph labels** (Item membership), for 10-K taxonomy enrichment and student pre-training. [HF](https://huggingface.co/datasets/eloukas/edgar-corpus) |
| **PubMed 200k RCT** | **No explicit licence**; NLM notes some abstracts may be copyrighted | 200k abstracts, 2.3M sentences | background / objective / method / result / conclusion | Evaluation only; legal review before training. [repo](https://github.com/Franck-Dernoncourt/pubmed-rct) |
| **unarXive IMRaD** | CC BY-SA 4.0 (share-alike: trained weights are arguably fine, redistributed data must stay SA) | 530k CS paragraphs | i / m / r / d / related-work | Paragraph-to-IMRaD training. [HF](https://huggingface.co/datasets/saier/unarXive_imrad_clf) |
| S2ORC rhetorical sections (2026) | Paper CC BY 4.0; **dataset licence not stated** | 15.6M papers | Section-header labels (I / M / R / D…) as offsets into S2ORC | Lead only. [arXiv 2607.03381](https://arxiv.org/abs/2607.03381) |
| SASC (LREC 2026) | **[unverified]** | 4,896 articles | 9 coarse + 47 fine section classes | Lead. [ACL preview](https://preview.aclanthology.org/ingest-lrec/2026.lrec-main.587/) |
| Classified OA paragraphs (Oct 2025) | CC BY 4.0 | 833k paragraphs | acknowledgements / data / software / clinical-trial mentions | Auxiliary paragraph tagging. [arXiv 2510.21762](https://arxiv.org/html/2510.21762v1) |
| **FinanceBench** | **CC BY-NC 4.0** (HF) | 150 open of 10,231 questions; PDFs included | `evidence_page_num` (0-indexed), evidence text and full page | **Dial evaluation**: did the selected elements contain the evidence? Evaluation only. [HF](https://huggingface.co/datasets/PatronusAI/financebench), [repo](https://github.com/patronus-ai/financebench) |
| **TAT-QA** | CC BY 4.0 (data), MIT (code) | 16,552 Qs over 2,757 hybrid contexts | Table+text QA, answer source | Table/paragraph evidence. [repo](https://github.com/NExTplusplus/TAT-QA) |
| **TAT-DQA** | cc-by-4.0 (HF metadata) | 3,067 pages from 182 reports, 16,558 QA | Document-page QA with evidence | Dial evaluation on real PDF pages. [paper](https://arxiv.org/pdf/2207.11871), [HF](https://huggingface.co/datasets/next-tat/TAT-DQA/tree/main) |
| **FinQA** | MIT (repo) | ~8k QA **[size unverified here]** | Reasoning programs + `gold_inds` (supporting facts) | Supporting-fact recall. [repo](https://github.com/czyssrs/FinQA) |
| MAUD / LegalBench / ACORD | Licences per task, or not stated **[unverified]** | MAUD 39k+ examples; ACORD 114 queries, 126k pairs | Deal points; legal reasoning; clause retrieval | Leads. [LegalBench](https://arxiv.org/pdf/2308.11462), [ACORD](https://arxiv.org/pdf/2501.06582) |

**2024–26 gap.** I found no public, permissively licensed dataset of paragraph-level semantic labels for financial filings beyond Item membership, which EDGAR-CORPUS provides. A search for 10-K paragraph-classification benchmarks turned up QA sets instead, such as Multi-Doc-2025 and HC-RAG ([arXiv 2608.12335](https://arxiv.org/pdf/2608.12335)). This gap is a reason to bootstrap with TELEClass.

**Recommendation:**
- **Layout evaluation:** DocLayNet, plus OmniDocBench for reading order (internal only).
- **Element-tag training:** CC BY / CDLA / Apache sets (CUAD, ContractNLI, TAT-QA/DQA, EDGAR-CORPUS item labels, unarXive with care), plus teacher labels on our own corpus.
- **Dial evaluation:** FinanceBench, TAT-DQA, ContractNLI and CUAD, measuring evidence recall vs tokens.
- **Licence register:** keep an `eval-only` flag in the dataset registry for NC sets.

---

## 4. Unlabelled PDF corpus sources

| Source | Access and rate rules | Format | Licence | Notes |
|---|---|---|---|---|
| **SEC EDGAR** | **≤10 requests/s**; must send a User-Agent with company name and admin email (e.g. `DocRooter Research admin@example.com`) and `Accept-Encoding: gzip, deflate`; no crawling botnets. Bulk: `full-index` / `daily-index` (company, form, master, xbrl), `Feed/` daily tarballs, data.sec.gov JSON APIs. | Official documents are ASCII/HTML (and XBRL); **PDF is an "unofficial copy"**, optional except for a few submission types | US government records; filings are public | [Accessing EDGAR data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data), [Filer Manual vol. II ch. 5](https://www.sec.gov/files/edgar/filermanual/edgarfilermmanual-vol2-c5.pdf), [2013 rule (PDF official for 497AD etc.)](https://www.sec.gov/files/rules/final/2013/33-9382.pdf) |
| EDGAR **Form ARS** | same | **PDF required since 11 Jan 2023**: glossy annual reports, unchanged in graphics and style | public | The best source of designed, multi-column native PDFs on EDGAR. [Debevoise](https://www.debevoise.com/insights/publications/2023/01/reminder-glossy-annual-reports-must-be-furnished), [Skadden](https://www.skadden.com/insights/publications/2023/03/annual-meeting-filing-and-disclosure-requirements), [example](https://www.sec.gov/Archives/edgar/data/1069258/000106925823000015/ars.pdf). Some ARS submissions are still HTML. |
| EDGAR fund documents | same | **497K** summary prospectuses are HTML + images on EDGAR ([example](https://www.sec.gov/Archives/edgar/data/773674/000077367425000022/0000773674-25-000022-index.htm)). **N-CSR** tailored shareholder reports have needed Inline XBRL since 24 Jul 2024 ([Proskauer](https://www.proskauer.com/alert/sec-answers-questions-on-new-tailored-shareholder-report-requirements)). | public | Rule 498 requires the summary and statutory prospectus to be posted on a website ([SEC guide](https://www.sec.gov/rules/final/2009/33-8998-secg.htm)). Fund sites typically post designed PDFs **[unverified as a general rule]**, but crawling those sites is outside EDGAR's policy and subject to each site's terms. |
| **govinfo** | Bulk XML at `/bulkdata` (FR, CFR, eCFR, BILLS, PLAW, STATUTE, etc.); API via api.data.gov key (default 1,000 req/h; GovInfo may allow more **[unverified]**) | Bulk is XML; per-package PDFs via API/links | US government works, generally public domain | [bulk data](https://www.govinfo.gov/bulkdata/), [api.data.gov limits](https://api.data.gov/docs/developer-manual/). Federal Register and CFR PDFs are three-column. Good for column detection. |
| **CourtListener / RECAP** | API: token auth; default **5/min, 50/h, 125/day** | Bulk CSV in S3 `com-courtlistener-storage/bulk-data/` (us-west-2) | Bulk data marked **Public Domain Mark 1.0** | [API overview](https://wiki.free.law/c/courtlistener/help/api/rest/v4/overview), [bulk data](https://wiki.free.law/c/courtlistener/help/api/bulk-data/bulk-legal-data). Whether RECAP PDFs are in the bulk export is **[unverified]**; RECAP is mirrored to the Internet Archive ([free.law/recap](https://free.law/recap/)). |
| **arXiv** | S3 bucket `arxiv` (us-east-1), **requester pays**; PDFs ~2.7 TB (Mar 2023), ~100 GB/month growth. API/OAI: **1 request / 3 s**, single connection. | PDF + source | Mostly arXiv's non-exclusive distribution licence, under which arXiv cannot grant redistribution rights. Do not host or serve the PDFs; storing them for research is allowed. A minority are CC (see OAI-PMH metadata). | [bulk S3](https://info.arxiv.org/help/bulk_data_s3.html), [API ToU](https://info.arxiv.org/help/api/tou.html). Use CC-BY papers for any redistributable derived set. |
| **RFC Editor** | rsync modules `rfcs-text-only`, `rfcs-pdf-only`, `rfcs-pdf-all`, `rfcs-json-only` | PDF / text / JSON | IETF Trust TLP: copying and distributing unmodified RFCs is allowed; derivative works generally are not | [bulk](https://www.rfc-editor.org/retrieve/bulk/), [TLP](https://trustee.ietf.org/documents/trust-legal-provisions/tlp-5). Monospace layout gives little layout diversity. |
| **Federal Reserve Board** | website | PDFs (reports, minutes, FEDS papers) | Board site: public domain unless indicated; cite the Board | [disclaimer](https://www.federalreserve.gov/disclaimer.htm). Regional Reserve Banks assert copyright, often with non-commercial terms ([Atlanta Fed](https://atlantafed.org/disclaimers-and-terms-of-use)). |
| data.gov | catalog | Dataset metadata, mostly not documents | varies | Use it for discovery only. |
| SafeDocs CC-MAIN-2021-31-PDF-UNTRUNCATED | HTTPS / S3 | ~8M web PDFs (re-fetched untruncated) | Common Crawl ToU + Digital Corpora ToU; no per-file licence | [Digital Corpora](https://digitalcorpora.org/cc-main-2021-31-pdf-untruncated/). Very diverse. Research only; do not redistribute. |
| GovDocs1 | download | ~1M mixed .gov files (PDF a minority) | "freely redistributable to the best of our knowledge" | [Digital Corpora](https://digitalcorpora.org/corpora/files/) |

**HTML vs PDF.** EDGAR 10-K, 10-Q, 8-K and 497K filings are HTML. Unofficial PDFs are optional and rare. I found no standard public pipeline that turns them into a PDF layout corpus. Common practice is headless-Chromium print-to-PDF (Playwright), which renders CSS and tables faithfully ([guide](https://apitemplate.io/blog/a-guide-to-generate-pdfs-in-python/)), or a paid API ([sec-api.io](https://sec-api.io/docs/sec-filings-to-pdf-generator-api/python-example.md)).

**Recommendation: a mixed corpus.**
1. **About 40% native PDFs** for realistic layout: ARS (2023+), govinfo FR/CFR, Fed Board, arXiv CC-BY, RFC and SafeDocs samples.
2. **About 60% EDGAR HTML rendered to PDF** with Playwright (Letter size, print CSS). Keep the DOM-to-PDF character mapping (inject span IDs before rendering, or align text). This gives **free weak labels**: `<h*>`/bold-only paragraphs → heading, `<table>` → table, `<li>` → list, Item boundaries → section, page-break footers → footer. Rendered HTML is mostly single-column with clean headers, so it must not dominate the layout evaluation.
3. **Optionally, HTML-to-text directly** for the semantic-tagging student. It sees text plus coarse layout features, so it does not need pixels.

The split percentages are a judgement call **[estimate]**. Keep the EDGAR client at ≤5 req/s, half the allowed rate, with a contact User-Agent.

---

## 5. TypeSafe Jev terms

- **Applicable agreement.** The [Master Customer Agreement](https://typesafe.ai/legal/mca) (last updated 23 Sep 2026) covers the API and Web Interface. An "Order" includes the website checkout page, and acceptance occurs by using or paying for the Services, so self-serve API users appear to be bound. How it applies to free or trial access is not stated.
- **§2.3(b): distillation is prohibited.** The customer may not "use the Services or any Output … to perform model distillation", train models to imitate the Services, or develop a similar or competing product (paraphrased; the quoted fragment is verbatim).
- **§4.2: output ownership.** TypeSafe disclaims ownership of Output and assigns any rights it has in Output to the customer.
- **§4.1: no training on customer data.** TypeSafe will not put Customer Data (Input and Output) into training datasets without consent. **§4.3 Telemetry** may be processed without restriction.
- **Other documents.** The [legal index](https://docs.typesafe.ai/legal.md) lists the DPA, the MCA and the Privacy Policy, plus zero data retention (ZDR) for enterprise. The [site Terms of Use](https://typesafe.ai/terms) cover the website only.

**Implications:**
- Using Jev labels to train the docrooter student is **model distillation**, so it is not permitted under the MCA as written.
- Jev is fine as a **runtime** option in the rooter and as a user-selected classifier.
- Using Jev labels only to **calibrate** thresholds or to **evaluate** the student is a grey area. Get written confirmation from TypeSafe (sales@typesafe.ai) or a separate agreement before doing either.
- **Other teachers:**
  - OpenAI's terms prohibit using Output "to develop models that compete with OpenAI" ([terms](https://openai.com/policies/row-terms-of-use/); not fetched, HTTP 403; wording per [secondary summary](https://conductatlas.com/platform/openai/openai-eu-terms-of-use/provision/CA-P-059288/no-use-of-output-to-compete-with-openai/)). A document-element tagger is arguably not competing, but that needs legal review.
  - The cleanest teacher is a **local or self-hosted open-weight LLM with a permissive licence** (e.g. Qwen3, reported Apache-2.0 with no output-use restriction ([LWN thread](https://lwn.net/Articles/1033239)) **[verify per model card]**).
  - The user's own provider is acceptable if the user accepts that provider's terms. Make "teacher" a user-configured engine and record the provenance of every label.

---

## 6. Calibration and conformal selection: the completeness dial

### 6.1 Methods
- **Conformal risk control** ([Angelopoulos et al., arXiv 2208.02814](https://arxiv.org/html/2208.02814v4)).
  - **Applies to:** any loss `L_i(λ)` that is non-increasing in λ and bounded by B.
  - **Rule:** choose `λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + B/(n+1) ≤ α}`.
  - **Guarantee:** `E[L_{n+1}(λ̂)] ≤ α`. It is tight to within `2B/(n+1)` for i.i.d. data.
  - **Worked example in the paper:** a false-negative-rate (FNR) loss, `1 − |Y ∩ C_λ|/|Y|`. On MS-COCO with α=0.1 it achieved mean FNR 0.0996.
  - This is exactly "expected recall ≥ 1−α".
- **Learn-then-Test** ([Angelopoulos et al., arXiv 2110.01052](https://arxiv.org/html/2110.01052)).
  - **Guarantee:** a high-probability guarantee, `P(R(λ̂) ≤ α) ≥ 1−δ`.
  - **Method:** Hoeffding–Bentkus p-values (`min(exp(−n·h₁(R̂∧α, α)), e·P(Bin(n,α) ≤ ⌈nR̂⌉))`), combined with **fixed-sequence testing** over a λ grid ordered a priori. Fixed sequence is more powerful than Bonferroni when the risk is near-monotone.
- **Temperature, Platt and isotonic calibration** ([scikit-learn guide](https://scikit-learn.org/stable/modules/calibration.html); temperature scaling: [Guo et al. 2017](https://arxiv.org/abs/1706.04599)).
  - Use sigmoid/Platt below ~1,000 calibration samples and isotonic above. Isotonic creates ties.
  - Temperature scaling has one parameter, is natively multiclass and does not change the argmax.
  - These make scores interpretable. **CRC does not need calibrated scores to be valid**, but calibrated scores make the threshold transfer better across document types.
- **Applied to RAG:**
  - [Chakraborty et al., ECIR 2026 (arXiv 2511.17908)](https://arxiv.org/abs/2511.17908): conformal filtering of retrieved snippets "consistently meets its target coverage" and cuts context 2–3×, with stable or better downstream factuality on NeuCLIR.
  - [TRAQ (NAACL 2024)](https://aclanthology.org/2024.naacl-long.210.pdf): answer-set coverage.
  - [C-RAG](https://par.nsf.gov/servlets/purl/10524902): certified generation risk.
  - [Conformal factuality (ICML 2024)](https://proceedings.mlr.press/v235/mohri24a.html): claim back-off.
  - **Caveat:** exchangeability is not automatic for retrieval scores ([arXiv 2605.05244](https://arxiv.org/html/2605.05244v1)).

### 6.2 Exact algorithm: choose a threshold so that expected recall ≥ target

**Units.** The exchangeable unit is the **document**, or a (document, query) pair when selection is query-conditioned. It is never the element, because elements within a document are dependent.

**Data.** A calibration set of n documents that are disjoint from the student's training data. For each document i, take its elements j with:
- `s_ij ∈ [0,1]`, the student's (temperature-scaled) relevance or keep score;
- `w_ij`, the weight: 1 for element recall, or token count for token-weighted recall;
- `r_ij ∈ {0,1}`, gold relevance from the teacher, adjudicated by a human on a subset.

**Algorithm:**
1. **Target.** The dial value `d ∈ [0,1]` maps to target recall `ρ`. The mapping is the identity, or a product curve such as `{0:0.5, 0.5:0.9, 0.8:0.95, 1:1.0}`. Set `α = 1 − ρ`. If `ρ = 1`, keep everything.
2. **Per-document loss** at threshold t (element kept iff `s_ij ≥ t`):
   `L_i(t) = 1 − Σ_j w_ij·r_ij·1[s_ij ≥ t] / Σ_j w_ij·r_ij`.
   Set `L_i(t) = 0` if document i has no relevant elements. `L_i(t)` is non-decreasing in t, and B = 1.
3. **Candidate thresholds.** `T` = all distinct `s_ij`, plus 0, sorted descending.
4. **Choose t̂.** Scan from the highest threshold down, using `R̂(t) = mean_i L_i(t)`:
   `t̂ = max{t ∈ T : (n/(n+1))·R̂(t) + 1/(n+1) ≤ α}`.
   - This is CRC with λ = −t. The set is never empty, because `R̂(0) = 0` and `1/(n+1) ≤ α` whenever `n ≥ 1/α − 1`.
   - Require a much larger n in practice, below.
5. **Deploy.** Keep the elements with `s ≥ t̂`.
   - **Guarantee:** `E[recall of a new document] ≥ ρ`, marginal over documents.
   - **Ties:** if two elements tie at t̂, keep both.
6. **Optional high-probability variant (LTT).**
   - For each t on a grid ordered from high to low, compute the HB p-value of `H_t: R(t) > α` from the n per-document losses, which lie in [0,1].
   - Walk the grid in order and reject while `p_t ≤ δ` (e.g. δ = 0.1). `t̂` is the last rejected t.
   - **Guarantee:** `P(E_doc[1−recall] ≤ α) ≥ 1−δ`.
7. **Groups (Mondrian).** If document types differ, run steps 2–4 separately per type, using the existing 18-type classifier. Only do this when a type has enough documents; otherwise pool.
8. **Budget mode, the dual problem.** To answer "maximum recall for K tokens", report the empirical recall curve `1 − R̂(t)` against mean tokens kept. This is descriptive, with no guarantee unless it is re-run as CRC.

**Sizing [estimate].**
- The guarantee holds for any n with `1/(n+1) ≤ α`. The slack term `1/(n+1)` and the threshold's variance argue for **n ≥ 200** documents per group at α = 0.05; n ≈ 500 is better.
- With n = 200, the `B/(n+1)` term alone uses 0.005 of a 0.05 budget.
- Recalibrate per user corpus, because the guarantee assumes the calibration and test documents are exchangeable.
- Monitor realised recall on a rolling, human-checked sample.

**Caveat on what is guaranteed.** The recall is relative to the gold definition `r_ij`, which means teacher labels. Teacher error bounds what the dial can promise. Measure teacher-vs-human agreement on the calibration subset and report it alongside the dial.

---

## 7. Encoder choice for the student

| Model | Params | Context | Licence | Classification evidence | Notes |
|---|---|---|---|---|---|
| **UltraBERT-150M** (dmis-lab, Oct 2026) | ~152M, 24 layers, h=768, 6 heads | 32,768 (local 128-token windows on 2 of every 3 layers; global attention on every third) | Apache-2.0 | GLUE 89.5 vs Ettin 88.9, ModernBERT 88.4, NomicBERT 84.0 (authors' table) | Prefers learning rates up to 10× lower than other encoders (sweep 5e-6–8e-5); AdamW β2 0.98, wd 0.1, 5% warm-up. Two MLM heads; the default 5% head is distilled from a 1.3B teacher. [card](https://huggingface.co/dmis-lab/UltraBERT-150M), [paper](https://arxiv.org/abs/2610.15028) |
| UltraBERT-380M | 380M, h=1024 | 32K | Apache-2.0 | GLUE 91.0, BEIR 47.7 | Too big to fine-tune comfortably on 8 GB **[estimate]** |
| UltraBERT-Embed-150M (current docrooter default) | 151M | 32K | Apache-2.0 | MTEB(eng v2) 67.24 | Embedding model; CLS pooling; needs sentence-transformers ≥6, transformers ≥5.20. [card](https://huggingface.co/dmis-lab/UltraBERT-Embed-150M) |
| ModernBERT-base | 149M, 22 layers | 8,192 | Apache-2.0 | GLUE 88.4 | Mature ecosystem. [card](https://huggingface.co/answerdotai/ModernBERT-base) |
| Ettin encoder (17M–1B) | 150M base | 8K | MIT | GLUE 88.9 (base) | A size ladder is useful for distillation. [card](https://huggingface.co/jhu-clsp/ettin-encoder-150m) |
| NeoBERT | ~250M | 4,096 | MIT | Claims to beat ModernBERT under identical fine-tuning; GLUE 89.0 per the UltraBERT table | Needs `xformers` and `trust_remote_code`, a poor fit for M1. [card](https://huggingface.co/chandar-lab/NeoBERT) |
| gte-modernbert-base | 149M | 8,192 | Apache-2.0 | MTEB 64.38 | Embedder. [card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) |
| bge-small-en-v1.5 | 33M | 512 | MIT | — | Fast fallback. [card](https://huggingface.co/BAAI/bge-small-en-v1.5) |
| nomic-embed-text v1.5 | ~137M | 8,192 | Apache-2.0 | — | **[not re-checked this session]** |

**Memory for fine-tuning a ~150M model at seq 512 [estimate].**
- **Static state:** fp32 weights 0.61 GB + gradients 0.61 GB + AdamW moments 1.22 GB ≈ **2.4 GB**.
- **Activations:** about `s·b·h·(34 + 5·a·s/h)` bytes per layer with 16-bit activations ([Korthikanti et al., arXiv 2205.05198](https://arxiv.org/abs/2205.05198)).
  - UltraBERT-150M (h=768, a=6, L=24, s=512): ≈ 21 MB per sample per layer, or ~0.5 GB per sample in bf16/fp16 and ~1 GB in fp32.
  - This assumes dense attention. Local-window kernels would cut the s² term, but MPS will likely materialise it.
- **What fits in 8 GB:**
  - **batch 4–6** with bf16/fp16 autocast and gradient accumulation (≈ 4.5–5.5 GB total); or
  - **batch 16+** with gradient checkpointing (activations then ≈ 20 MB per sample, plus one layer's worth).
  - fp32 at batch 8 will not fit.
- Measure on the M1 itself, under the existing process-wide MPS cap. Prefer pre-tokenised, length-bucketed batches; most paragraphs are under 256 tokens, which roughly halves activation memory.

**Recommendation:**
- **Student:** **UltraBERT-150M** (the backbone, not Embed).
  - It is Apache-2.0, has the best reported base-size GLUE, and has a 32K context that allows "element + surrounding section" inputs.
  - It shares a tokenizer and family with the current embedder.
  - **Risks:** it is two weeks old, so the benchmarks are the authors' own; and it needs new transformers versions.
- **Fallback:** keep ModernBERT-base or Ettin-150M as a drop-in backup behind the same training script.
- **Cheaper stage first:** before fine-tuning, train a logistic-regression or MLP head on the UltraBERT-Embed embeddings that docrooter already computes. Inference stays numpy-only, which keeps a torch-free path. Promote to full fine-tuning only if the head plateaus.

---

## 8. LLM-as-teacher distillation for classification

**Evidence:**
- **TELEClass** ([arXiv 2403.00165](https://arxiv.org/html/2403.00165)):
  - **Teacher stage:** GPT-3.5 generates class terms and picks core classes from a top-down candidate set (keeping l+3 children per level).
  - **Taxonomy enrichment:** AutoPhrase terms scored by the geometric mean of popularity, BM25 distinctiveness and embedding similarity, keeping the top 20 per class.
  - **Confidence:** the **similarity gap** (the largest drop in the sorted document-class similarities) is the confidence, and the **top 75%** of documents are kept.
  - **Student and augmentation:** a BERT-base matching classifier, plus 5 LLM-generated pseudo-documents per path.
  - **Results:** Example-F1 0.648 on Amazon-531 and 0.863 on DBPedia-298, vs ChatGPT zero-shot at 0.516 and 0.482. It trails GPT-4 level-by-level prompting on Amazon (0.699). Inference costs under $1, vs $60–2,500 for LLM labelling of the test set.
- **CanDist** ([ACL 2025, arXiv 2506.03857](https://arxiv.org/html/2506.03857)):
  - **Method:** ask the teacher for **candidate label sets** when it is unsure, averaging 1.07–1.70 labels each, and let the student resolve them ("distribution refinery").
  - **Results:** a RoBERTa-base student beats its GPT-3.5 teacher on all 6 tasks. Test accuracy gains range from +0.8 (MA) to +10.6 (TREC), with Banking-77 at +7.4.
  - **Scale:** about 5–11k teacher-labelled examples per task.
- **Students exceeding teachers is a repeated finding:**
  - [LLMaAA (EMNLP Findings 2023)](https://arxiv.org/pdf/2310.19596): "within only hundreds of annotated examples", for NER/RE.
  - "[Student as an inherent denoiser](https://arxiv.org/pdf/2312.10185)".
  - [FreeAL](https://arxiv.org/pdf/2311.15614): the student filters clean samples back to the LLM.
- **Fine-tuned small models beat zero-shot LLMs** across classification tasks ([Bucher & Martini 2024](https://arxiv.org/html/2406.08660v2)).
- **Uncertainty routing to humans** with proxy validation ([arXiv 2406.12114](https://arxiv.org/pdf/2406.12114)).
- **Counter-example on gaps:** a ModernBERT student distilled from an NLI teacher reached only 62.5–77.5% teacher agreement on a 60-class taxonomy. That is a preliminary preprint with tiny evaluation sets ([arXiv 2610.02516](https://arxiv.org/html/2610.02516)).

**Best-practice recipe for docrooter:**
1. **Teacher prompt.** Give the taxonomy path plus the element text and context (section heading path, element type, neighbouring elements). Ask for a label, **candidates if unsure**, and a short rationale. Use a permissive open-weight teacher (§5).
2. **Confidence filtering.** Combine:
   - the TELEClass similarity gap between the element embedding and the enriched class embeddings;
   - teacher agreement: teacher label ∈ top-k by similarity;
   - the teacher's candidate-set size, where 1 means confident.
   Keep the top ~75% for the first student round, as in TELEClass.
3. **Noise handling.** Use CanDist-style candidate re-normalisation, or simply train on soft targets over the candidate set. Run a small-loss / confident-learning pass ([Northcutt et al.](https://arxiv.org/abs/1911.00068)) to flag likely mislabels for re-query.
4. **Active learning loop.** After round 1, send the teacher the student's most uncertain elements: high entropy, small top-2 margin, or disagreement with the similarity gap. Stratify by document type and class. Repeat 2–3 rounds.
5. **Budget [estimate, from CanDist and TELEClass scales]:**
   - About **100–300 teacher labels per leaf class** to start, giving 5–15k for a 50-class element taxonomy, plus 1–2k active-learning queries per round.
   - At least 5 LLM-generated pseudo-elements per path for rare classes.
   - Plot a learning curve at 25/50/100% of labels before buying more. No source gives a universal per-class number.
6. **Human gold.** About 50 adjudicated examples per class, kept for evaluation and dial calibration. Never train on them.
7. **Expected gap.** Expect the student to match or exceed the teacher on in-distribution data for coarse taxonomies (CanDist, LLMaAA). Expect a deficit on fine-grained, long-tail or out-of-domain classes (arXiv 2610.02516, TELEClass vs GPT-4 on Amazon). Report per-class F1 against both the teacher and the human gold.

---

## 9. Structure-aware chunking and section-aware retrieval

- **Element-type chunking on financial reports** ([Jimeno Yepes et al., arXiv 2402.05131](https://arxiv.org/html/2402.05131v3), FinanceBench):
  - Element-based chunks raised manual Q&A accuracy to **53.2%**, vs 48.2% for the best fixed-size chunking (512 tokens), with a chunk count between the 512- and 256-token baselines (20.8k vs 16.0k and 32.1k).
  - Page-retrieval accuracy was *lower* for single element strategies (67.4%) than for 256-token chunks (73.1%).
  - Aggregating chunks reached 84.4% page accuracy with 44% fewer chunks than base aggregation.
- **Semantic chunking is not reliably worth its cost** ([Qu et al., arXiv 2410.13070](https://arxiv.org/html/2410.13070)):
  - Fixed-size chunking won 3 of 5 evidence-retrieval sets, and answer quality was mostly tied.
  - Breakpoint chunking helped only on artificially stitched, topic-diverse documents (Miracl F1@5 81.9 vs 69.5).
- **Contextual retrieval (summaries or context as index)** ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)): prepending an LLM-written, chunk-specific context cut top-20 retrieval failures by:
  - 35% with contextual embeddings;
  - 49% when contextual BM25 is added;
  - 67% when reranking is also added.
- **RAPTOR, hierarchical summary trees** ([arXiv 2401.18059](https://arxiv.org/abs/2401.18059)): +20 points absolute accuracy on QuALITY with GPT-4.
- **Late chunking** ([arXiv 2409.04701](https://arxiv.org/html/2409.04701v3)): encode the whole document, then pool per chunk. That gives about +1.5–1.9 nDCG@10 on BEIR (2.7–3.6% relative), with the most benefit for small chunks. This is directly enabled by UltraBERT-Embed's 32K context.
- **Parent-child and hierarchical retrieval:** used widely (e.g. [H-RAG, SemEval-2026](https://arxiv.org/pdf/2605.00631)). I found no controlled 2025–26 benchmark that isolates heading-path-aware chunking against parent-child retrieval **[gap]**.

**Recommendation:**
- **Element and section structure as units.** Expose docrooter elements as retrieval units, each carrying its heading path, element type and section label. Index the text as `heading path + element text`, which is cheap contextualisation with no LLM.
- **Context expansion.** Return parent sections or neighbours at read time.
- **Optional extras:** an LLM context or summary per section (the contextual-retrieval pattern), and late-chunked embeddings from UltraBERT-Embed.
- **Tables:** keep each table whole, plus a one-line description (the "Prefix & Table Description" variant won in 2402.05131).
- **Tie to the dial.** The dial selects elements, and the chunker groups selected elements into contiguous section spans for the LLM.

---

## 10. Open questions and unverified items
- PyMuPDF Layout licence history (AGPL vs PolyForm-NC by version); whether egret ONNX exports exist; heron and OCR speed on M1.
- Licence of DocLayout-YOLO weights; licence of unstructured's yolox weights.
- Licences for SASC, the S2ORC rhetorical-section dataset, MAUD, ACORD and PubMed-RCT.
- Whether CourtListener bulk data includes RECAP PDFs; GovInfo API's actual rate limit.
- Whether Jev Output may be used for evaluation or calibration (not training). Ask TypeSafe.
- Apple Vision OCR accuracy vs RapidOCR and OnnxTR on documents. This needs an in-house benchmark.
- All **[estimate]** thresholds and memory figures need measuring on the M1 under the process-wide MPS cap.
