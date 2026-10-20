# Installation

docrooter needs Python 3.10 or newer. The core install is small (`pydantic`, `numpy`, `typer`,
`rich`); models, PDF parsing, LLM backends and the playground are optional extras.

docrooter 1.0 is installed from GitHub; it is not on PyPI yet.

```bash
pip install "git+https://github.com/LandoCole/docrooter@v1.0.0"                       # core
pip install "docrooter[llm,pdf] @ git+https://github.com/LandoCole/docrooter@v1.0.0"  # + LLMs, PDFs
pip install "docrooter[all] @ git+https://github.com/LandoCole/docrooter@v1.0.0"      # everything
```

The core is enough for Markdown, the taxonomies, the `keyword` backend and the CLI; `llm` adds
the OpenAI-compatible backend that `docrooter.classify` is meant to deploy with. Elsewhere these
docs write `pip install 'docrooter[pdf]'`: until docrooter is on PyPI, use the form above with
the same extras. With [uv](https://docs.astral.sh/uv/):
`uv add "docrooter[pdf,models] @ git+https://github.com/LandoCole/docrooter@v1.0.0"`.

## Extras

| Extra | Adds | Licence of what it adds | Needed for |
| --- | --- | --- | --- |
| *(none)* | `pydantic`, `numpy`, `typer`, `rich` | MIT / BSD | Markdown and text loading, taxonomies, the `keyword` backend, the CLI |
| `pdf` | `pymupdf4llm`, which installs PyMuPDF | **AGPL-3.0** (or a commercial licence from Artifex) | Loading PDFs with PyMuPDF, the default and fastest engine |
| `pdf-mit` | `pdfplumber` | MIT (its dependencies pdfminer.six and pypdfium2: MIT, Apache-2.0 / BSD-3) | Loading PDFs with a permissively licensed engine |
| `models` | `torch`, `transformers>=5`, `sentence-transformers>=6`, `scikit-learn`, `joblib`, `datasets` | BSD / Apache-2.0 (torch: a mix of permissive licences) | The `embedding`, `nli`, `mlm-choice` and `trained` backends, `docrooter train`, `docrooter models` |
| `llm` | `httpx` | BSD | The `openai` backend (any OpenAI-compatible server, hosted or a local stand-in), the deployment path of `docrooter.classify` |
| `jev` | `typesafe-sdk` | MIT client; the Jev API is a hosted service with its own terms | The `jev` backend (TypeSafe Jev, a hosted API) |
| `ui` | `fastapi`, `uvicorn`, `python-multipart` | MIT / BSD / Apache-2.0 | `docrooter serve`, the local [playground](../guides/playground.md) |
| `ocr` | `rapidocr`, `onnxruntime`, `pypdfium2` | Apache-2.0 / MIT / BSD-3 and Apache-2.0; rapidocr also pulls in `opencv-python-headless` (Apache-2.0, bundles **FFmpeg, LGPL-2.1**) and `shapely` (BSD-3, bundles **GEOS, LGPL-2.1**) | The `rapidocr` engine in element mode, and page rendering without PyMuPDF |
| `ocr-mac` | `ocrmac` (macOS only) | MIT | The `ocrmac` engine (Apple Vision) |
| `layout` | `onnxruntime`, `pypdfium2`, `huggingface_hub` | MIT / BSD-3 and Apache-2.0 / Apache-2.0 | The `heron` layout model of the `accurate` profile (no torch) |
| `teach-mlx` | `mlx-lm` (Apple silicon, Python 3.11+ only) | MIT | The default teacher for [training your own element tagger](../guides/train-your-tagger.md) (experimental): Qwen3-4B-Instruct-2507 (Apache-2.0), 4-bit, on MLX. Not part of `all`, because it installs on one platform only |
| `bench` | `datasets`, `pandas` | Apache-2.0 / BSD | The benchmark scripts, `DocumentResult.to_dataframe()` |
| `all` | `models`, `pdf`, `pdf-mit`, `llm`, `jev`, `ui`, `ocr`, `ocr-mac`, `layout` | as above | Everything except `bench` |

docrooter itself is Apache-2.0. The Tesseract engine needs no extra: docrooter calls the system
`tesseract` binary. The licence column summarises the main packages each extra pulls in, as declared in their package metadata; it is not legal advice. No default install and no `ocr`, `ocr-mac` or `layout` extra pulls in a GPL or AGPL package, but the `ocr` extra installs two LGPL-2.1 libraries inside other wheels (FFmpeg in OpenCV, GEOS in Shapely), whose terms apply if you redistribute them. Model weights carry
their own licences, listed on their Hugging Face pages. Check both before you distribute
anything.

If you call something that needs an extra you have not installed, the error names the command to
run, for example `pip install 'docrooter[models]'`.

## Choosing a PDF engine

PyMuPDF (extra `pdf`) is licensed under the **AGPL-3.0**, unlike docrooter. It is imported only
when a PDF is loaded, but read the AGPL terms before you distribute software or run a network
service that includes it. For a permissive stack install `pdf-mit` instead:

- `load(path)` uses PyMuPDF when it is installed and pdfplumber otherwise (`engine="auto"`).
- `load(path, engine="pdfplumber")` forces pdfplumber.
- pdfplumber is about 6 times slower and scores about the same on the development corpus, but
  failed silently on one very large PDF in the held-out set; see the
  [FAQ](../faq.md#agpl-or-mit-which-pdf-engine).

Neither engine does OCR: a scanned PDF without a text layer gives an empty document from
`load()`. OCR belongs to element mode (see [OCR and layout models](../guides/ocr-and-layout.md)).

## Models and the first run

The default backend is the embedding model `dmis-lab/UltraBERT-Embed-150M`. The first run
downloads it (about 600 MB) into the Hugging Face cache. It ships its own modelling code, which
transformers only runs with `trust_remote_code=True`; docrooter enables that only for models
whose built-in recipe pins a commit, so the code that runs is the code at that commit. See
[Backends](../concepts/backends.md#trust_remote_code) for the details and for other models.

To prepare a machine that will run offline:

```bash
docrooter models pull dmis-lab/UltraBERT-Embed-150M     # the pinned revision
HF_HUB_OFFLINE=1 docrooter doctype reports/             # later, without network access
```

Model backends pick the device in the order CUDA, MPS (Apple silicon), CPU, and stay within a
memory budget of 8 GiB by default; see [Memory and performance](../guides/memory-performance.md).

## Check the install

```bash
docrooter --version
docrooter --help
```

## From a release wheel

Each [GitHub release](https://github.com/LandoCole/docrooter/releases) carries the wheel and
the sdist. Download `docrooter-1.0.0-py3-none-any.whl` and install the file, with extras if you
need them: `pip install "./docrooter-1.0.0-py3-none-any.whl[llm]"`.

## From source

```bash
git clone https://github.com/LandoCole/docrooter && cd docrooter
uv sync --extra pdf --extra models
PYTHONPATH=src uv run docrooter --help
```

`uv sync` makes the environment match exactly what you ask for, so repeat the same `--extra`
flags every time. On macOS, prefix commands with `PYTHONPATH=src` (the editable install's `.pth`
file can get a hidden flag that Python then skips). The
[contributing guide](../about/contributing.md) has the full development setup.
