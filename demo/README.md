# docrooter live demo

The interactive page behind [landonthornton.com/docrooter/](https://landonthornton.com/docrooter/):
pick a rung of the context ladder and see Claude Haiku 5.5's answer, the exact tokens and the
cost.

- `index.html` is the whole page (no build step). Samples load instantly from `data/`.
- `data/` holds the samples and the benchmark summary, made by `build_samples.py` from the
  committed benchmark results. It makes no model calls, and only uses documents whose licence
  allows redistribution (GAO reports, CC BY 4.0 arXiv papers and CUAD contracts, U.S. federal
  court opinions, RFCs).
- `api/docrooter_demo.py` is the live upload: a Vercel Python function that runs docrooter on the
  uploaded file with Claude Haiku 5.5 (`ANTHROPIC_API_KEY` from the environment), limited per
  visitor to 10 runs a minute and 100 a day.

Run it locally (live uploads need `ANTHROPIC_API_KEY`; the samples work without it):

```bash
PYTHONPATH=src python demo/serve.py
```

Rebuild the samples after a new benchmark run (needs the benchmark PDFs in the cache):

```bash
PYTHONPATH=src python demo/build_samples.py
```

The site copies this folder with its own `tools/sync_docrooter.py`, which swaps in the site's
licensed font and pins the function's `docrooter` dependency to a commit of this repository.
Edit the demo here, not in the site.
