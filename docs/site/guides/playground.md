# The playground

`docrooter serve` starts a local web page for looking at one document in detail: drop in a file,
watch each stage of the pipeline run with its timing, and inspect the table of contents, the
document type and the section labels.

![The docrooter playground after a run: the pipeline stages as a timed waterfall, then the table
of contents](../../assets/playground.png)

*The Federal Reserve's July 2024 Monetary Policy Report (79 pages) with the `keyword` backend,
which needs no model and only checks the pipeline: it calls this report a proxy statement.*

```bash
pip install 'docrooter[ui]'
docrooter serve                    # http://127.0.0.1:8765
docrooter serve -b keyword --open  # pick the preselected backend, open the browser
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--host` | `127.0.0.1` | Address to bind. Anything other than a loopback address prints a warning. |
| `--port` | `8765` | Port to listen on |
| `-b/--backend` | `keyword` | The backend selected when the page opens; it can be switched on the page |
| `--open` | off | Open the page in the browser once the server is up |
| `--env-file` | none | Read missing environment variables (such as `TYPESAFE_API_KEY`) from this file |

## Using it

1. **Choose the options** in the left rail: the model (backend), the PDF engine, what the type
   is decided from (the whole document's view, its sections, or both) and whether to label
   sections with canonical names.
2. **Drop a file** (`.pdf`, `.md` or `.txt`, up to 50 MB) on the drop zone, or click to choose
   one.
3. **Watch the timeline.** Each stage appears as it starts and fills in as it ends, as a
   waterfall with its duration and details: pages read, lines, headings found per level,
   tokens in the view, the model's device, the type chosen. The clock shows the wall time.
4. **Read the results** in four tabs:
    - *Contents*: the table of contents as a tree, with page numbers, heading score chips and
      flags such as `low_heading_confidence`.
    - *Document type*: the chosen type and the probabilities of the five likeliest types.
    - *Section labels*: the canonical label of each section, next to its heading.
    - *Raw JSON*: the whole result, with a *Download JSON* button.

Recent runs stay in the sidebar while the server runs (the last 20, in memory only). *Run the
last file again* repeats a run with different options.

The `keyword` backend is selected by default so the page works without models; switch to the
default embedding model or `jev` for real results. The first run with a model includes loading
it (the `model_load` stage); later runs reuse it, and switching backends releases the previous
one. The `jev` backend appears only when `TYPESAFE_API_KEY` is set.

## Limits and safety

The playground is a local tool for one user. Its guard rails:

- It binds `127.0.0.1` by default and rejects requests whose `Host` header is not the address it
  serves, which stops DNS-rebinding attacks from a web page in your browser.
- It loads no external assets (fonts, scripts, styles), so it works offline.
- Uploads are capped at 50 MB, stored in a temporary directory and deleted as soon as they have
  been processed. Results stay in memory only.
- One worker processes runs one at a time, so memory stays bounded; at most 8 documents can
  wait. PDFs are read up to 2,000 pages.
- Model backends obey the same [memory budget](memory-performance.md) as the library: set
  `DOCROOTER_MEMORY_BUDGET_GB` before starting it.

It has no authentication. Do not expose it to a network you do not trust; read the
[security policy](../about/security.md) before you change `--host`.

## The API behind it

The page is one static HTML file and a small script, talking to a JSON API you can also call
yourself (for example from a notebook, while the server runs):

| Endpoint | Does |
| --- | --- |
| `POST /api/runs` | Multipart `file`, plus `backend`, `engine`, `mode` and `normalize`; returns `{"run_id": ...}` |
| `GET /api/runs/{id}/events` | Server-Sent Events: one JSON [event](../concepts/events.md) per stage, ending with `{"stage": "job", "status": "end"}` |
| `GET /api/runs/{id}` | The run's final result |
| `GET /api/runs` | The recent runs |
| `GET /api/backends` | The available backends, engines and modes, and the upload limit |

The API is meant for the page; it is not a stable interface and may change between releases.
