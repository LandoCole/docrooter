# Security policy

## Supported versions

Security fixes go into the latest release. Older releases are not patched; upgrade instead.

| Version | Supported |
| --- | --- |
| 1.0.x (latest) | yes |
| 0.x | no |
| `main` (unreleased) | fixes land here first |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.** Use GitHub's private vulnerability
reporting instead: open the repository's **Security** tab, choose **Report a vulnerability**, and
describe the problem (or go straight to
<https://github.com/LandoCole/docrooter/security/advisories/new>).

Please include:

- the docrooter version (`pip show docrooter`) and Python version, and your operating system;
- what you did, what you expected and what happened, ideally with a minimal input (a small PDF, a
  taxonomy file, a command line);
- the impact as you see it.

What to expect: we aim to acknowledge a report within a few working days, to tell you whether we
consider it a vulnerability, and to agree on a fix and a disclosure date with you. We credit
reporters in the release notes unless they prefer not to be named. docrooter is maintained by
volunteers, so these are goals, not guarantees.

Problems in a dependency (PyMuPDF, pdfplumber, torch, transformers, FastAPI, ...) should go to that
project; tell us as well if docrooter's use of it makes the problem worse.

## Security-relevant design notes

What docrooter does about security, and what it leaves to you.

### Remote model code (`trust_remote_code`)

Some Hugging Face models, such as the default UltraBERT, ship their own modelling code, which
transformers runs only with `trust_remote_code=True`. docrooter turns that on only for models whose
built-in recipe pins a commit, so the code that runs is the code at that commit and not whatever the
repository holds later. For any other model that needs remote code the load fails until you pass
`trust_remote_code=True` yourself, which you should do only after reading the code. Pinned
revisions are used for the downloads too (`docrooter models pull` fetches the pinned revision
unless you give `--revision`).

### Trained bundles are pickles

A model bundle written by `docrooter train` or `train()` stores its per-node heads as `joblib`
files, which are Python pickles. **Loading a pickle can run arbitrary code.** Load only bundles you
trained yourself or got from a source you trust, whether through `TrainedModel.load(path)`, the
`trained:<path>` backend spec or `docrooter eval -b trained:...`. Treat a bundle like an executable.
Taxonomy and dataset files (JSON, CSV, JSONL) are parsed as data and cannot run code.

### API keys

- Keys are read from the environment: `TYPESAFE_API_KEY` for `jev`, and `DOCROOTER_LLM_API_KEY` or
  `OPENAI_API_KEY` for `openai`. You can also pass `api_key=` explicitly.
- docrooter does not read `.env` files on its own. `docrooter serve --env-file FILE` reads the
  variables the environment lacks from a file you name, and never overrides one that is set.
  Keep that file out of version control (it is in `.gitignore` here).
- Keys are never logged or printed. They are stripped of whitespace and redacted from error
  messages and from the text of any exception a backend raises.
- Never put a key in code, a taxonomy file or a committed configuration.
- A hosted backend sends your document text to its provider (TypeSafe for `jev`, the endpoint you
  configure for `openai`). Use a local backend, or a server you run, for confidential documents.

### The playground (`docrooter serve`)

The playground is a development tool for one user on one machine; it has no accounts or
authentication.

- It binds `127.0.0.1` by default. Binding another address prints a warning, because anyone who
  can reach the port can then upload documents and make the machine run models.
- A host guard rejects (HTTP 400) any request whose `Host` header is not the address it is served on,
  which stops DNS-rebinding attacks from a web page you happen to visit. A cross-site form post
  (an `Origin` that is not this host) is refused with 403.
- It serves no external assets: no CDN scripts, fonts or images.
- Uploads are capped at 50 MB, and the cap is enforced while the body streams, not after it has
  been buffered. At most 8 documents wait behind the running one (more get 503), and a PDF is read
  to at most 2,000 pages.
- Markdown and text uploads are capped at 1 MB, and a run keeps at most 5,000 TOC entries, so a
  heading-dense text file cannot make one run use gigabytes of memory.
- Uploaded files are written to the system temporary directory and deleted after processing;
  files left behind by a killed server are swept (after an hour) when the next one starts.
- Error messages shown in the page are length-limited.

### Untrusted PDFs

PDF parsers read complicated, attacker-controlled files, and PyMuPDF and pdfplumber are
third-party code that can have vulnerabilities of their own.

- docrooter reads PDFs natively and does not run OCR. It extracts text, fonts, positions and the
  outline, and nothing in a document is executed by docrooter itself.
- Bound the work a hostile or huge file can cause: `load_pdf(path, max_pages=N)` and
  `docrooter doctype --max-pages N` read at most `N` pages, and `--chunk-size` keeps the memory of a
  large corpus bounded. The playground applies a page cap of its own.
- For documents from untrusted sources, keep the PDF libraries up to date, and run docrooter in a
  sandbox (a container or an unprivileged user without network access) if the risk matters to you.
  `docrooter doctype --workers N` parses PDFs in separate processes, which also contains a crash.

### Supply chain

The core install is small (`pydantic`, `numpy`, `typer`, `rich`); everything else is an optional
extra. Releases are published to PyPI from GitHub Actions with trusted publishing, so no API token is
stored in the repository. PyMuPDF (the `pdf` extra) is AGPL-3.0, which is a licensing matter rather
than a security one; see the README.
